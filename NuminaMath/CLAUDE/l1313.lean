import Mathlib

namespace cookies_for_guests_l1313_131341

/-- Given a total number of cookies and cookies per guest, calculate the number of guests --/
def calculate_guests (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_cookies / cookies_per_guest

/-- Theorem: Given 10 total cookies and 2 cookies per guest, the number of guests is 5 --/
theorem cookies_for_guests : calculate_guests 10 2 = 5 := by
  sorry

end cookies_for_guests_l1313_131341


namespace fixed_point_on_line_l1313_131329

/-- The line (a-1)x + ay + 3 = 0 passes through the point (3, -3) for any real a -/
theorem fixed_point_on_line (a : ℝ) : (a - 1) * 3 + a * (-3) + 3 = 0 := by
  sorry

end fixed_point_on_line_l1313_131329


namespace inequality_theorem_l1313_131385

-- Define the functions p and q
variable (p q : ℝ → ℝ)

-- Define the theorem
theorem inequality_theorem 
  (h1 : Differentiable ℝ p) 
  (h2 : Differentiable ℝ q)
  (h3 : p 0 = q 0)
  (h4 : p 0 > 0)
  (h5 : ∀ x ∈ Set.Icc 0 1, deriv p x * Real.sqrt (deriv q x) = Real.sqrt 2) :
  ∀ x ∈ Set.Icc 0 1, p x + 2 * q x > 3 * x := by
sorry


end inequality_theorem_l1313_131385


namespace function_inequality_l1313_131346

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_ineq : ∀ x, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2013 < Real.exp 2013 * f 0 := by
  sorry

end function_inequality_l1313_131346


namespace brenda_banana_pudding_trays_l1313_131301

/-- Proof that Brenda can make 3 trays of banana pudding given the conditions --/
theorem brenda_banana_pudding_trays :
  ∀ (cookies_per_tray : ℕ) 
    (cookies_per_box : ℕ) 
    (cost_per_box : ℚ) 
    (total_spent : ℚ),
  cookies_per_tray = 80 →
  cookies_per_box = 60 →
  cost_per_box = 7/2 →
  total_spent = 14 →
  (total_spent / cost_per_box * cookies_per_box) / cookies_per_tray = 3 :=
by
  sorry

#check brenda_banana_pudding_trays

end brenda_banana_pudding_trays_l1313_131301


namespace max_sum_with_length_constraint_l1313_131380

-- Define the length function
def length (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem max_sum_with_length_constraint :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ length x + length y = 16 ∧ 
  ∀ (a b : ℕ), a > 1 → b > 1 → length a + length b = 16 → 
  a + 3 * b ≤ x + 3 * y ∧ x + 3 * y = 98306 :=
sorry

end max_sum_with_length_constraint_l1313_131380


namespace horner_operations_l1313_131383

-- Define the polynomial coefficients
def coeffs : List ℝ := [8, 7, 6, 5, 4, 3, 2]

-- Define Horner's method
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

-- Define a function to count operations in Horner's method
def count_operations (coeffs : List ℝ) : ℕ × ℕ :=
  (coeffs.length - 1, coeffs.length - 1)

-- Theorem statement
theorem horner_operations :
  let (mults, adds) := count_operations coeffs
  mults = 6 ∧ adds = 6 :=
sorry

end horner_operations_l1313_131383


namespace max_expensive_price_is_11000_l1313_131370

/-- Represents a company's product line -/
structure ProductLine where
  num_products : ℕ
  average_price : ℝ
  min_price : ℝ
  num_below_threshold : ℕ
  price_threshold : ℝ

/-- The maximum possible price for the most expensive product -/
def max_expensive_price (pl : ProductLine) : ℝ :=
  let total_price := pl.num_products * pl.average_price
  let min_price_sum := pl.num_below_threshold * pl.min_price
  let remaining_price := total_price - min_price_sum
  let remaining_products := pl.num_products - pl.num_below_threshold
  remaining_price - (remaining_products - 1) * pl.price_threshold

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_price_is_11000 (c : ProductLine) 
  (h1 : c.num_products = 20)
  (h2 : c.average_price = 1200)
  (h3 : c.min_price = 400)
  (h4 : c.num_below_threshold = 10)
  (h5 : c.price_threshold = 1000) :
  max_expensive_price c = 11000 := by
  sorry


end max_expensive_price_is_11000_l1313_131370


namespace hiking_problem_l1313_131368

/-- Hiking problem statement -/
theorem hiking_problem (endpoint_distance : ℝ) (speed_ratio : ℝ) (head_start : ℝ) (meet_time : ℝ) 
  (planned_time : ℝ) (early_arrival : ℝ) :
  endpoint_distance = 7.5 →
  speed_ratio = 1.5 →
  head_start = 0.75 →
  meet_time = 0.5 →
  planned_time = 1 →
  early_arrival = 1/6 →
  ∃ (speed_a speed_b actual_time : ℝ),
    speed_a = 4.5 ∧
    speed_b = 3 ∧
    actual_time = 4/3 ∧
    speed_a = speed_ratio * speed_b ∧
    (speed_a - speed_b) * meet_time = head_start ∧
    endpoint_distance / speed_b - early_arrival = planned_time + (endpoint_distance - speed_b * planned_time) / speed_a :=
by sorry


end hiking_problem_l1313_131368


namespace complex_number_location_l1313_131361

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I)^2 = 1 - Complex.I) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_location_l1313_131361


namespace pascal_triangle_row20_element5_l1313_131387

theorem pascal_triangle_row20_element5 : Nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row20_element5_l1313_131387


namespace parabola_equation_l1313_131314

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  focus : ℝ × ℝ
  opens_left : Bool

/-- The standard form equation of a parabola. -/
def standard_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = -4 * p.focus.1 * x

theorem parabola_equation (p : Parabola) 
  (h1 : p.focus = (-3, 0)) 
  (h2 : p.opens_left = true) : 
  standard_equation p = fun x y => y^2 = -12 * x := by
sorry

end parabola_equation_l1313_131314


namespace arccos_equation_solution_l1313_131396

theorem arccos_equation_solution :
  ∃ x : ℝ, x = -1/3 ∧ Real.arccos (3*x) - Real.arccos (2*x) = π/6 :=
by sorry

end arccos_equation_solution_l1313_131396


namespace possible_ordering_l1313_131335

theorem possible_ordering (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (positive : a > 0 ∧ b > 0 ∧ c > 0)
  (eq : a^2 + c^2 = 2*b*c) :
  b > a ∧ a > c :=
sorry

end possible_ordering_l1313_131335


namespace function_properties_l1313_131366

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < c then c * x + 1 else 3 * x^4 + x^2 * c

theorem function_properties (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 1) (h3 : f c c^2 = 9/8) :
  c = 1/2 ∧ ∀ x, f (1/2) x < 2 ↔ 0 < x ∧ x < 2/3 := by
  sorry

end function_properties_l1313_131366


namespace modulus_of_z_l1313_131372

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_z_l1313_131372


namespace sphere_only_all_circular_views_l1313_131392

/-- Enumeration of common geometric bodies -/
inductive GeometricBody
  | Cone
  | Cylinder
  | Sphere
  | HollowCylinder

/-- Definition for a view of a geometric body being circular -/
def isCircularView (body : GeometricBody) (view : String) : Prop := sorry

/-- Theorem stating that only a sphere has all circular views -/
theorem sphere_only_all_circular_views (body : GeometricBody) :
  (isCircularView body "front" ∧ 
   isCircularView body "side" ∧ 
   isCircularView body "top") ↔ 
  body = GeometricBody.Sphere :=
sorry

end sphere_only_all_circular_views_l1313_131392


namespace range_of_m_l1313_131322

def is_hyperbola (m : ℝ) : Prop := (m + 2) * (m - 3) > 0

def no_positive_roots (m : ℝ) : Prop :=
  m = 0 ∨ (m ≠ 0 ∧ (∀ x : ℝ, x > 0 → m * x^2 + (m + 3) * x + 4 ≠ 0))

theorem range_of_m (m : ℝ) :
  (is_hyperbola m ∨ no_positive_roots m) ∧
  ¬(is_hyperbola m ∧ no_positive_roots m) →
  m < -2 ∨ (0 ≤ m ∧ m ≤ 3) :=
by sorry

end range_of_m_l1313_131322


namespace men_in_club_l1313_131307

theorem men_in_club (total : ℕ) (attendees : ℕ) (h_total : total = 30) (h_attendees : attendees = 18) :
  ∃ (men women : ℕ),
    men + women = total ∧
    men + (women / 3) = attendees ∧
    men = 12 := by
  sorry

end men_in_club_l1313_131307


namespace project_distribution_count_l1313_131320

/-- The number of ways to distribute 8 projects among 4 companies -/
def distribute_projects : ℕ :=
  -- Total projects
  let total := 8
  -- Projects for each company
  let company_A := 3
  let company_B := 1
  let company_C := 2
  let company_D := 2
  -- The actual calculation would go here
  1680

/-- Theorem stating that the number of ways to distribute the projects is 1680 -/
theorem project_distribution_count : distribute_projects = 1680 := by
  sorry

end project_distribution_count_l1313_131320


namespace geometric_sequence_eighth_term_l1313_131336

/-- Given a geometric sequence of positive numbers where the fifth term is 16 and the eleventh term is 2, 
    prove that the eighth term is 4√2. -/
theorem geometric_sequence_eighth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a > 0) 
  (h2 : r > 0) 
  (h3 : a * r^4 = 16) 
  (h4 : a * r^10 = 2) : 
  a * r^7 = 4 * Real.sqrt 2 := by
sorry

end geometric_sequence_eighth_term_l1313_131336


namespace find_M_l1313_131309

theorem find_M (p q r s M : ℚ) 
  (sum_eq : p + q + r + s = 100)
  (p_eq : p + 10 = M)
  (q_eq : q - 5 = M)
  (r_eq : 10 * r = M)
  (s_eq : s / 2 = M) :
  M = 1050 / 41 := by
  sorry

end find_M_l1313_131309


namespace graph_relationship_l1313_131363

theorem graph_relationship (x : ℝ) : |x^2 - 3/2*x + 3| ≥ x^2 + 3/2*x + 3 := by
  sorry

end graph_relationship_l1313_131363


namespace three_distinct_values_l1313_131349

/-- The number of distinct values possible when evaluating 3^(3^(3^3)) with different parenthesizations -/
def num_distinct_values : ℕ := 3

/-- The original expression 3^(3^(3^3)) -/
def original_expr : ℕ := 3^(3^(3^3))

theorem three_distinct_values :
  ∃ (a b : ℕ), a ≠ b ∧ a ≠ original_expr ∧ b ≠ original_expr ∧
  (∀ (x : ℕ), x ≠ a ∧ x ≠ b ∧ x ≠ original_expr →
    ¬∃ (e₁ e₂ e₃ : ℕ → ℕ → ℕ), x = e₁ 3 (e₂ 3 (e₃ 3 3))) ∧
  num_distinct_values = 3 :=
sorry

end three_distinct_values_l1313_131349


namespace sum_of_fractions_l1313_131327

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + 
  (1 / (8 * 9 : ℚ)) = 7 / 18 := by
  sorry

end sum_of_fractions_l1313_131327


namespace rent_increase_problem_l1313_131315

/-- Given a group of 4 friends paying rent, where:
  - The initial average rent is $800
  - After one person's rent increases by 20%, the new average is $870
  This theorem proves that the original rent of the person whose rent increased was $1400. -/
theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800)
  (h2 : new_average = 870) (h3 : num_friends = 4) (h4 : increase_percentage = 0.2) :
  ∃ (original_rent : ℝ), 
    original_rent * (1 + increase_percentage) = 
      num_friends * new_average - (num_friends - 1) * initial_average ∧
    original_rent = 1400 :=
by sorry

end rent_increase_problem_l1313_131315


namespace bamboo_volume_proof_l1313_131388

theorem bamboo_volume_proof (a : ℕ → ℚ) :
  (∀ i : ℕ, i < 8 → a (i + 1) - a i = a (i + 2) - a (i + 1)) →  -- arithmetic progression
  a 1 + a 2 + a 3 + a 4 = 3 →                                   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →                                         -- sum of last 3 terms
  a 5 + a 6 = 31/9 := by
sorry

end bamboo_volume_proof_l1313_131388


namespace fraction_multiplication_addition_l1313_131360

theorem fraction_multiplication_addition : (2 / 9 : ℚ) * (5 / 6 : ℚ) + (1 / 18 : ℚ) = (13 / 54 : ℚ) := by
  sorry

end fraction_multiplication_addition_l1313_131360


namespace cans_per_bag_l1313_131399

theorem cans_per_bag (total_cans : ℕ) (num_bags : ℕ) (h1 : total_cans = 20) (h2 : num_bags = 4) :
  total_cans / num_bags = 5 := by
  sorry

end cans_per_bag_l1313_131399


namespace common_area_rotated_squares_l1313_131381

/-- The area of the region common to two squares with side length 2, 
    where one is rotated about a vertex by an angle θ such that cos θ = 3/5 -/
theorem common_area_rotated_squares (θ : Real) : 
  θ.cos = 3/5 → 
  (2 : Real) > 0 → 
  (4 * θ.cos * θ.sin : Real) = 48/25 := by
  sorry

end common_area_rotated_squares_l1313_131381


namespace correct_fraction_l1313_131384

theorem correct_fraction (number : ℕ) (x y : ℕ) (h1 : number = 192) 
  (h2 : (5 : ℚ) / 6 * number = x / y * number + 100) : x / y = (5 : ℚ) / 16 := by
  sorry

end correct_fraction_l1313_131384


namespace alternating_squares_sum_l1313_131398

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end alternating_squares_sum_l1313_131398


namespace system_solution_l1313_131308

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 4 * y = -7) ∧ 
  (7 * x - 3 * y = 5) ∧ 
  (x = 41 / 19) ∧ 
  (y = 64 / 19) := by
  sorry

end system_solution_l1313_131308


namespace cube_root_negative_equals_negative_cube_root_l1313_131371

theorem cube_root_negative_equals_negative_cube_root (x : ℝ) (h : x > 0) :
  ((-x : ℝ) ^ (1/3 : ℝ)) = -(x ^ (1/3 : ℝ)) :=
sorry

end cube_root_negative_equals_negative_cube_root_l1313_131371


namespace probability_theorem_l1313_131319

/-- A rectangle with dimensions 3 × 2 units -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- 10 points evenly spaced along the perimeter of the rectangle -/
def num_points : ℕ := 10

/-- The probability of selecting two points one unit apart -/
def probability_one_unit_apart (rect : Rectangle) : ℚ :=
  2 / 9

/-- Theorem stating the probability of selecting two points one unit apart -/
theorem probability_theorem (rect : Rectangle) 
  (h1 : rect.length = 3) 
  (h2 : rect.width = 2) : 
  probability_one_unit_apart rect = 2 / 9 := by
  sorry

end probability_theorem_l1313_131319


namespace samantha_birth_year_l1313_131317

def mathLeagueYear (n : ℕ) : ℕ := 1995 + 2 * (n - 1)

theorem samantha_birth_year :
  (∀ n : ℕ, mathLeagueYear n = 1995 + 2 * (n - 1)) →
  mathLeagueYear 5 - 13 = 1990 :=
by sorry

end samantha_birth_year_l1313_131317


namespace mans_walking_speed_l1313_131344

/-- Proves that given a man who walks a certain distance in 5 hours and runs the same distance at 15 kmph in 36 minutes, his walking speed is 1.8 kmph. -/
theorem mans_walking_speed 
  (walking_time : ℝ) 
  (running_speed : ℝ) 
  (running_time_minutes : ℝ) :
  walking_time = 5 →
  running_speed = 15 →
  running_time_minutes = 36 →
  (walking_time * (running_speed * (running_time_minutes / 60))) / walking_time = 1.8 :=
by sorry

end mans_walking_speed_l1313_131344


namespace rectangle_property_l1313_131337

-- Define the rectangle's properties
def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 3

-- Define the area and perimeter functions
def area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def perimeter (x : ℝ) : ℝ := 2 * (rectangle_length x + rectangle_width x)

-- State the theorem
theorem rectangle_property :
  ∃ x : ℝ, x > 0 ∧ area x = 3 * perimeter x ∧ Real.sqrt ((9 + Real.sqrt 153) / 4 - x) < 0.001 :=
by sorry

end rectangle_property_l1313_131337


namespace school_capacity_l1313_131312

/-- Given a school with the following properties:
  * It has 15 classrooms
  * One-third of the classrooms have 30 desks each
  * The rest of the classrooms have 25 desks each
  * Only one student can sit at one desk
  This theorem proves that the school can accommodate 400 students. -/
theorem school_capacity :
  let total_classrooms : ℕ := 15
  let desks_per_large_classroom : ℕ := 30
  let desks_per_small_classroom : ℕ := 25
  let large_classrooms : ℕ := total_classrooms / 3
  let small_classrooms : ℕ := total_classrooms - large_classrooms
  let total_capacity : ℕ := large_classrooms * desks_per_large_classroom +
                            small_classrooms * desks_per_small_classroom
  total_capacity = 400 := by
  sorry

end school_capacity_l1313_131312


namespace last_two_digits_sum_l1313_131300

theorem last_two_digits_sum (n : ℕ) : (9^25 + 13^25) % 100 = 42 := by
  sorry

end last_two_digits_sum_l1313_131300


namespace smallest_cube_factor_l1313_131325

theorem smallest_cube_factor (n : ℕ) (h : n = 1512) :
  (∃ (y : ℕ), y > 0 ∧ n * 49 = y^3) ∧
  (∀ (x : ℕ), x > 0 ∧ x < 49 → ¬∃ (y : ℕ), y > 0 ∧ n * x = y^3) :=
by sorry

end smallest_cube_factor_l1313_131325


namespace lamp_marked_price_l1313_131342

/-- The marked price of a lamp given initial price, purchase discount, desired gain, and sales discount -/
def marked_price (initial_price : ℚ) (purchase_discount : ℚ) (desired_gain : ℚ) (sales_discount : ℚ) : ℚ :=
  let cost_price := initial_price * (1 - purchase_discount)
  let selling_price := cost_price * (1 + desired_gain)
  selling_price / (1 - sales_discount)

theorem lamp_marked_price :
  marked_price 40 (1/5) (1/4) (3/20) = 800/17 := by
  sorry

end lamp_marked_price_l1313_131342


namespace circle_equation_proof_l1313_131323

theorem circle_equation_proof (x y : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ x y, x^2 + y^2 + 1 = 2*x + 4*y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  (∃ (h k : ℝ), ∀ x y, x^2 + y^2 + 1 = 2*x + 4*y ↔ (x - h)^2 + (y - k)^2 = 4) :=
by sorry

#check circle_equation_proof

end circle_equation_proof_l1313_131323


namespace cos_20_cos_25_minus_sin_20_sin_25_l1313_131303

theorem cos_20_cos_25_minus_sin_20_sin_25 :
  Real.cos (20 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) -
  Real.sin (20 * Real.pi / 180) * Real.sin (25 * Real.pi / 180) =
  Real.sqrt 2 / 2 := by
sorry

end cos_20_cos_25_minus_sin_20_sin_25_l1313_131303


namespace pure_imaginary_magnitude_l1313_131313

theorem pure_imaginary_magnitude (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 9) (m^2 + 2*m - 3)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 12 := by
  sorry

end pure_imaginary_magnitude_l1313_131313


namespace numerator_increase_percentage_l1313_131389

theorem numerator_increase_percentage (P : ℝ) : 
  (5 * (1 + P / 100)) / (7 * (1 - 10 / 100)) = 20 / 21 → P = 20 := by
  sorry

end numerator_increase_percentage_l1313_131389


namespace intersection_of_A_and_B_l1313_131310

-- Define sets A and B
def A : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 = 0}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {(3/5, -9/5)} := by
  sorry

end intersection_of_A_and_B_l1313_131310


namespace john_supermarket_spending_l1313_131356

theorem john_supermarket_spending : 
  ∀ (total : ℚ),
  (1 / 2 : ℚ) * total + (1 / 3 : ℚ) * total + (1 / 10 : ℚ) * total + 5 = total →
  total = 75 := by
sorry

end john_supermarket_spending_l1313_131356


namespace geometric_sequence_product_l1313_131328

/-- Given a geometric sequence {a_n} where a₄ = 4, prove that a₃a₅ = 16 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) →  -- geometric sequence condition
  a 4 = 4 →                                        -- given condition
  a 3 * a 5 = 16 :=                                -- conclusion to prove
by sorry

end geometric_sequence_product_l1313_131328


namespace cos_two_theta_value_l1313_131348

theorem cos_two_theta_value (θ : Real) 
  (h : Real.exp (Real.log 2 * (1 - 3/2 + 3 * Real.cos θ)) + 3 = Real.exp (Real.log 2 * (2 + Real.cos θ))) :
  Real.cos (2 * θ) = -1/2 := by
sorry

end cos_two_theta_value_l1313_131348


namespace range_of_a_l1313_131333

-- Define the propositions p and q
def p (a x : ℝ) : Prop := 3 * a < x ∧ x < a
def q (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (a < 0) →
  (∀ x : ℝ, ¬(p a x) → ¬(q x)) →
  (∃ x : ℝ, ¬(p a x) ∧ q x) →
  -2/3 ≤ a ∧ a < 0 :=
sorry

end range_of_a_l1313_131333


namespace positive_A_value_l1313_131352

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- State the theorem
theorem positive_A_value (A : ℝ) (h : hash A 3 = 130) : A = 11 := by
  sorry

end positive_A_value_l1313_131352


namespace optimal_characterization_l1313_131379

def Ω : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 2008}

def better (p q : ℝ × ℝ) : Prop := p.1 ≤ q.1 ∧ p.2 ≥ q.2

def optimal (q : ℝ × ℝ) : Prop :=
  q ∈ Ω ∧ ∀ p ∈ Ω, ¬(better p q ∧ p ≠ q)

theorem optimal_characterization (q : ℝ × ℝ) :
  optimal q ↔ q.1^2 + q.2^2 = 2008 ∧ q.1 ≤ 0 ∧ q.2 ≥ 0 :=
sorry

end optimal_characterization_l1313_131379


namespace textbook_weight_difference_l1313_131353

theorem textbook_weight_difference :
  let chemistry_weight : Float := 7.12
  let geometry_weight : Float := 0.62
  (chemistry_weight - geometry_weight) = 6.50 := by
  sorry

end textbook_weight_difference_l1313_131353


namespace zeros_after_one_in_power_l1313_131340

theorem zeros_after_one_in_power (n : ℕ) (h : 10000 = 10^4) :
  10000^50 = 10^200 := by
  sorry

end zeros_after_one_in_power_l1313_131340


namespace grasshopper_final_position_l1313_131386

/-- The number of positions in the circular arrangement -/
def num_positions : ℕ := 6

/-- The number of jumps the grasshopper makes -/
def num_jumps : ℕ := 100

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The final position of the grasshopper after num_jumps -/
def final_position : ℕ := (sum_first_n num_jumps) % num_positions + 1

theorem grasshopper_final_position :
  final_position = 5 := by sorry

end grasshopper_final_position_l1313_131386


namespace equation_equality_l1313_131326

theorem equation_equality (a b : ℝ) : (a - b)^3 * (b - a)^4 = (a - b)^7 := by
  sorry

end equation_equality_l1313_131326


namespace no_simultaneous_overtake_l1313_131345

/-- Proves that there is no time when Teena is simultaneously 25 miles ahead of Yoe and 10 miles ahead of Lona -/
theorem no_simultaneous_overtake :
  ¬ ∃ t : ℝ, t > 0 ∧ 
  (85 * t - 60 * t = 25 + 17.5) ∧ 
  (85 * t - 70 * t = 10 + 20) :=
sorry

end no_simultaneous_overtake_l1313_131345


namespace number_of_lineups_l1313_131378

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of regular players in the starting lineup
def regular_players : ℕ := 11

-- Define the number of goalies in the starting lineup
def goalies : ℕ := 1

-- Theorem stating the number of different starting lineups
theorem number_of_lineups : 
  (total_players.choose goalies) * ((total_players - goalies).choose regular_players) = 222768 := by
  sorry

end number_of_lineups_l1313_131378


namespace expression_evaluation_l1313_131357

theorem expression_evaluation : 
  let x := Real.sqrt ((9^9 + 3^12) / (9^5 + 3^13))
  ∃ ε > 0, abs (x - 15.3) < ε ∧ ε < 0.1 :=
by sorry

end expression_evaluation_l1313_131357


namespace arithmetic_sequence_problem_l1313_131394

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 6th term of the arithmetic sequence is 11 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_a2 : a 2 = 3)
    (h_sum : a 3 + a 5 = 14) :
  a 6 = 11 := by
  sorry


end arithmetic_sequence_problem_l1313_131394


namespace wall_bricks_proof_l1313_131347

/-- Represents the number of bricks in the wall -/
def wall_bricks : ℕ := 127

/-- Bea's time to build the wall alone in hours -/
def bea_time : ℚ := 8

/-- Ben's time to build the wall alone in hours -/
def ben_time : ℚ := 12

/-- Bea's break time in minutes per hour -/
def bea_break : ℚ := 10

/-- Ben's break time in minutes per hour -/
def ben_break : ℚ := 15

/-- Decrease in output when working together in bricks per hour -/
def output_decrease : ℕ := 12

/-- Time taken to complete the wall when working together in hours -/
def combined_time : ℚ := 6

/-- Bea's effective working time per hour in minutes -/
def bea_effective_time : ℚ := 60 - bea_break

/-- Ben's effective working time per hour in minutes -/
def ben_effective_time : ℚ := 60 - ben_break

theorem wall_bricks_proof :
  let bea_rate : ℚ := wall_bricks / (bea_time * bea_effective_time / 60)
  let ben_rate : ℚ := wall_bricks / (ben_time * ben_effective_time / 60)
  let combined_rate : ℚ := bea_rate + ben_rate - output_decrease
  combined_rate * combined_time = wall_bricks :=
by sorry

end wall_bricks_proof_l1313_131347


namespace exactly_one_sick_probability_l1313_131391

/-- The probability of an employee being sick on any given day -/
def prob_sick : ℚ := 1 / 40

/-- The probability of an employee not being sick on any given day -/
def prob_not_sick : ℚ := 1 - prob_sick

/-- The probability of exactly one out of three employees being sick -/
def prob_one_sick_out_of_three : ℚ :=
  3 * prob_sick * prob_not_sick * prob_not_sick

theorem exactly_one_sick_probability :
  prob_one_sick_out_of_three = 4563 / 64000 := by sorry

end exactly_one_sick_probability_l1313_131391


namespace quadrilateral_front_view_solids_l1313_131393

-- Define the possible solid figures
inductive SolidFigure
  | Cone
  | Cylinder
  | TriangularPyramid
  | RectangularPrism

-- Define a predicate for having a quadrilateral front view
def has_quadrilateral_front_view (s : SolidFigure) : Prop :=
  match s with
  | SolidFigure.Cylinder => True
  | SolidFigure.RectangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids (s : SolidFigure) :
  has_quadrilateral_front_view s ↔ (s = SolidFigure.Cylinder ∨ s = SolidFigure.RectangularPrism) :=
by sorry

end quadrilateral_front_view_solids_l1313_131393


namespace power_division_rule_l1313_131306

theorem power_division_rule (a : ℝ) : a^4 / a^3 = a := by
  sorry

end power_division_rule_l1313_131306


namespace other_number_l1313_131321

theorem other_number (x : ℝ) : 
  0.5 > x ∧ 0.5 - x = 0.16666666666666669 → x = 0.3333333333333333 := by
  sorry

end other_number_l1313_131321


namespace one_and_two_thirds_of_x_is_36_l1313_131358

theorem one_and_two_thirds_of_x_is_36 (x : ℝ) : (5/3) * x = 36 → x = 21.6 := by
  sorry

end one_and_two_thirds_of_x_is_36_l1313_131358


namespace fractional_equation_simplification_l1313_131367

theorem fractional_equation_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 3) ↔ (2 * x = 3 - 6 * x + 6) :=
by sorry

end fractional_equation_simplification_l1313_131367


namespace machine_work_time_l1313_131377

/-- Given machines A, B, and C, where B takes 3 hours and C takes 6 hours to complete a job,
    and all three machines together take 4/3 hours, prove that A takes 4 hours alone. -/
theorem machine_work_time (time_B time_C time_ABC : ℝ) (time_A : ℝ) : 
  time_B = 3 → 
  time_C = 6 → 
  time_ABC = 4/3 → 
  1/time_A + 1/time_B + 1/time_C = 1/time_ABC → 
  time_A = 4 := by
sorry

end machine_work_time_l1313_131377


namespace two_numbers_sum_diff_product_l1313_131376

theorem two_numbers_sum_diff_product : ∃ (x y : ℝ), 
  x + y = 24 ∧ x - y = 8 ∧ x * y > 100 := by
  sorry

end two_numbers_sum_diff_product_l1313_131376


namespace absolute_value_equality_l1313_131395

theorem absolute_value_equality (m : ℝ) : |m| = |-7| → m = 7 ∨ m = -7 := by
  sorry

end absolute_value_equality_l1313_131395


namespace win_sector_area_l1313_131359

/-- The area of the WIN sector on a circular spinner with given radius and win probability -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 8) (h_p : p = 3/7) :
  p * π * r^2 = (192 * π) / 7 := by
  sorry

end win_sector_area_l1313_131359


namespace unique_five_digit_number_l1313_131330

theorem unique_five_digit_number : ∃! n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧ 
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (∀ i, 0 ≤ i ∧ i < 5 → (n / 10^i) % 10 ≠ 0) ∧
  ((n % 1000) = 7 * (n / 100)) ∧
  n = 12946 := by
sorry

end unique_five_digit_number_l1313_131330


namespace problem_solution_l1313_131364

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + 2 * y - 6 = 0
def equation2 (x y m : ℝ) : Prop := x - 2 * y + m * x + 5 = 0

theorem problem_solution :
  -- Part 1: Positive integer solutions
  (∀ x y : ℕ+, equation1 x y ↔ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 1)) ∧
  -- Part 2: Value of m when x + y = 0
  (∀ x y m : ℝ, x + y = 0 → equation1 x y → equation2 x y m → m = -13/6) ∧
  -- Part 3: Fixed solution regardless of m
  (∀ m : ℝ, equation2 0 (5/2) m) := by
sorry

end problem_solution_l1313_131364


namespace value_of_3x_minus_y_l1313_131375

-- Define the augmented matrix
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℚ := !![2, 1, 5; 1, -2, 0]

-- Define the system of equations
def system_equations (x y : ℚ) : Prop :=
  2 * x + y = 5 ∧ x - 2 * y = 0

-- Theorem statement
theorem value_of_3x_minus_y :
  ∃ x y : ℚ, system_equations x y → 3 * x - y = 5 := by
  sorry

end value_of_3x_minus_y_l1313_131375


namespace polynomial_sum_of_terms_l1313_131339

def polynomial (x : ℝ) : ℝ := 4 * x^2 - 3 * x - 2

def term1 (x : ℝ) : ℝ := 4 * x^2
def term2 (x : ℝ) : ℝ := -3 * x
def term3 : ℝ := -2

theorem polynomial_sum_of_terms :
  ∀ x : ℝ, polynomial x = term1 x + term2 x + term3 := by
  sorry

end polynomial_sum_of_terms_l1313_131339


namespace max_value_a_l1313_131374

theorem max_value_a : ∃ (a : ℝ) (b : ℤ), 
  (a * b^2) / (a + 2 * ↑b) = 2019 ∧ 
  ∀ (a' : ℝ) (b' : ℤ), (a' * b'^2) / (a' + 2 * ↑b') = 2019 → a' ≤ a :=
by sorry

end max_value_a_l1313_131374


namespace complement_of_A_l1313_131302

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 1} := by sorry

end complement_of_A_l1313_131302


namespace no_single_digit_fraction_l1313_131311

theorem no_single_digit_fraction :
  ¬ ∃ (n : ℕ+) (a b : ℕ),
    1 ≤ a ∧ a < 10 ∧
    1 ≤ b ∧ b < 10 ∧
    (1234 - n) * b = (6789 - n) * a :=
by sorry

end no_single_digit_fraction_l1313_131311


namespace dog_park_ratio_l1313_131397

theorem dog_park_ratio (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_ear_dogs : ℕ) :
  pointy_ear_dogs = total_dogs / 5 →
  pointy_ear_dogs = 6 →
  spotted_dogs = 15 →
  (spotted_dogs : ℚ) / total_dogs = 1 / 2 := by
  sorry

end dog_park_ratio_l1313_131397


namespace circle_and_line_intersection_l1313_131351

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y : ℝ) : Prop := 2*x - y + 2 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  -- Circle C passes through intersection points of parabola and coordinate axes
  (∃ x₁ x₂ x₃ y₃ : ℝ, 
    parabola x₁ 0 ∧ parabola x₂ 0 ∧ parabola 0 y₃ ∧
    circle_C x₁ 0 ∧ circle_C x₂ 0 ∧ circle_C 0 y₃) →
  -- Line intersects circle C at two points
  (∃ A B : ℝ × ℝ, 
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ A ≠ B) →
  -- Distance between intersection points is 6√5/5
  ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (6 * Real.sqrt 5) / 5 :=
by
  sorry


end circle_and_line_intersection_l1313_131351


namespace investment_calculation_l1313_131373

theorem investment_calculation (total : ℝ) (ratio : ℝ) (mutual_funds : ℝ) (bonds : ℝ) :
  total = 240000 ∧ 
  mutual_funds = ratio * bonds ∧ 
  ratio = 6 ∧ 
  total = mutual_funds + bonds →
  mutual_funds = 205714.29 := by
  sorry

end investment_calculation_l1313_131373


namespace dusting_team_combinations_l1313_131331

theorem dusting_team_combinations (n : ℕ) (k : ℕ) : n = 5 → k = 3 → Nat.choose n k = 10 := by
  sorry

end dusting_team_combinations_l1313_131331


namespace petya_vasya_meeting_l1313_131305

/-- The number of lampposts along the alley -/
def num_lampposts : ℕ := 100

/-- The lamppost where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamppost where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

theorem petya_vasya_meeting :
  ∀ (petya_speed vasya_speed : ℚ),
    petya_speed > 0 →
    vasya_speed > 0 →
    (petya_observed - 1 : ℚ) / petya_speed = (num_lampposts - vasya_observed : ℚ) / vasya_speed →
    (meeting_point - 1 : ℚ) / petya_speed = (num_lampposts - meeting_point : ℚ) / vasya_speed :=
by sorry

end petya_vasya_meeting_l1313_131305


namespace jelly_bean_probability_l1313_131324

theorem jelly_bean_probability (p_r p_o p_y p_g : ℝ) :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 := by
sorry

end jelly_bean_probability_l1313_131324


namespace washing_time_is_seven_hours_l1313_131382

/-- Calculates the number of cycles needed for a given number of items and capacity per cycle -/
def cycles_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- Calculates the total washing time in minutes -/
def total_washing_time (shirts pants sweaters jeans socks scarves : ℕ) 
  (regular_capacity sock_capacity scarf_capacity : ℕ)
  (regular_time sock_time scarf_time : ℕ) : ℕ :=
  let regular_cycles := cycles_needed shirts regular_capacity + 
                        cycles_needed pants regular_capacity + 
                        cycles_needed sweaters regular_capacity + 
                        cycles_needed jeans regular_capacity
  let sock_cycles := cycles_needed socks sock_capacity
  let scarf_cycles := cycles_needed scarves scarf_capacity
  regular_cycles * regular_time + sock_cycles * sock_time + scarf_cycles * scarf_time

theorem washing_time_is_seven_hours :
  total_washing_time 18 12 17 13 10 8 15 10 5 45 30 60 = 7 * 60 := by
  sorry

end washing_time_is_seven_hours_l1313_131382


namespace rio_persimmon_picking_l1313_131304

/-- Given the conditions of Rio's persimmon picking, calculate the average number of persimmons
    she must pick from each of the last 5 trees to achieve her desired overall average. -/
theorem rio_persimmon_picking (first_pick : ℕ) (first_trees : ℕ) (remaining_trees : ℕ) (desired_avg : ℚ) :
  first_pick = 12 →
  first_trees = 5 →
  remaining_trees = 5 →
  desired_avg = 4 →
  (desired_avg * (first_trees + remaining_trees) - first_pick) / remaining_trees = 28/5 := by
  sorry

end rio_persimmon_picking_l1313_131304


namespace M_intersect_N_eq_one_two_left_closed_l1313_131332

/-- The set M of real numbers x such that (x + 3)(x - 2) < 0 -/
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}

/-- The set N of real numbers x such that 1 ≤ x ≤ 3 -/
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

/-- The theorem stating that the intersection of M and N is equal to the interval [1, 2) -/
theorem M_intersect_N_eq_one_two_left_closed :
  M ∩ N = Set.Ioc 1 2 := by sorry

end M_intersect_N_eq_one_two_left_closed_l1313_131332


namespace chefs_wage_difference_l1313_131365

/-- Proves that the difference between the total hourly wage of 3 managers
    and the total hourly wage of 3 chefs is $3.9375, given the specified conditions. -/
theorem chefs_wage_difference (manager_wage : ℝ) (num_chefs num_dishwashers : ℕ) :
  manager_wage = 8.5 →
  num_chefs = 3 →
  num_dishwashers = 4 →
  let first_dishwasher_wage := manager_wage / 2
  let dishwasher_wages := [
    first_dishwasher_wage,
    first_dishwasher_wage + 1.5,
    first_dishwasher_wage + 3,
    first_dishwasher_wage + 4.5
  ]
  let chef_wages := (List.take num_chefs dishwasher_wages).map (λ w => w * 1.25)
  (3 * manager_wage - chef_wages.sum) = 3.9375 := by
  sorry

end chefs_wage_difference_l1313_131365


namespace robin_seeds_count_robin_seeds_is_150_l1313_131362

theorem robin_seeds_count : ℕ → ℕ → Prop :=
  fun (robin_bushes sparrow_bushes : ℕ) =>
    (robin_bushes = sparrow_bushes + 5) →
    (5 * robin_bushes = 6 * sparrow_bushes) →
    (5 * robin_bushes = 150)

/-- The number of seeds hidden by the robin is 150 -/
theorem robin_seeds_is_150 : ∃ (robin_bushes sparrow_bushes : ℕ),
  robin_seeds_count robin_bushes sparrow_bushes :=
by
  sorry

#check robin_seeds_is_150

end robin_seeds_count_robin_seeds_is_150_l1313_131362


namespace base_representation_five_digits_l1313_131316

theorem base_representation_five_digits (b' : ℕ+) : 
  (∃ (a b c d e : ℕ), a ≠ 0 ∧ 216 = a*(b'^4) + b*(b'^3) + c*(b'^2) + d*(b'^1) + e ∧ 
   a < b' ∧ b < b' ∧ c < b' ∧ d < b' ∧ e < b') ↔ b' = 3 :=
sorry

end base_representation_five_digits_l1313_131316


namespace right_triangle_sin_A_l1313_131355

theorem right_triangle_sin_A (A B C : ℝ) : 
  -- ABC is a right triangle
  A + B + C = Real.pi ∧ A = Real.pi / 2 →
  -- sin B = 3/5
  Real.sin B = 3 / 5 →
  -- sin C = 4/5
  Real.sin C = 4 / 5 →
  -- sin A = 1
  Real.sin A = 1 := by
sorry

end right_triangle_sin_A_l1313_131355


namespace f_five_values_l1313_131318

def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y^2) = f (x^2 - y) + 4 * (f x) * y^2

theorem f_five_values (f : ℝ → ℝ) (h : FunctionProperty f) : 
  f 5 = 0 ∨ f 5 = 25 := by sorry

end f_five_values_l1313_131318


namespace right_triangle_hypotenuse_l1313_131343

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
  a + b + c = 40 ∧  -- perimeter condition
  (1/2) * a * b = 24 ∧  -- area condition
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  c = 18.8 := by
  sorry


end right_triangle_hypotenuse_l1313_131343


namespace root_negative_implies_inequality_l1313_131369

theorem root_negative_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 2*a + 4 = 0 ∧ x < 0) → (a-3)*(a-4) > 0 := by
  sorry

end root_negative_implies_inequality_l1313_131369


namespace sin_cos_transformation_given_condition_l1313_131334

theorem sin_cos_transformation (x : ℝ) :
  4 * Real.sin x * Real.cos x = 2 * Real.sin (2 * x + π / 6) :=
by
  sorry

-- Additional theorem to represent the given condition
theorem given_condition (x : ℝ) :
  Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * x - π / 3) :=
by
  sorry

end sin_cos_transformation_given_condition_l1313_131334


namespace parallel_lines_a_value_l1313_131350

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of a when two given lines are parallel -/
theorem parallel_lines_a_value : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = 3 * x + a / 3 ↔ y = (a - 3) * x + 2) → a = 6 :=
by sorry

end parallel_lines_a_value_l1313_131350


namespace f_iterated_property_l1313_131338

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the iteration of f
def iterate_f (p q : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n+1, x => iterate_f p q n (f p q x)

theorem f_iterated_property (p q : ℝ) 
  (h : ∀ x ∈ Set.Icc 1 3, |f p q x| ≤ 1/2) :
  iterate_f p q 2017 ((3 + Real.sqrt 7) / 2) = (3 - Real.sqrt 7) / 2 := by
  sorry

end f_iterated_property_l1313_131338


namespace student_position_l1313_131390

theorem student_position (total_students : ℕ) (position_from_back : ℕ) (position_from_front : ℕ) :
  total_students = 27 →
  position_from_back = 13 →
  position_from_front = total_students - position_from_back + 1 →
  position_from_front = 15 :=
by sorry

end student_position_l1313_131390


namespace quadratic_inequality_solution_set_l1313_131354

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) :
  Set.Ioo (-1/2 : ℝ) (-1/3) = {x : ℝ | b*x^2 - a*x - 1 > 0} := by
  sorry

end quadratic_inequality_solution_set_l1313_131354
