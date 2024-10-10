import Mathlib

namespace relationship_abc_l4103_410389

theorem relationship_abc : 
  let a : ℝ := (0.6 : ℝ) ^ (2/5 : ℝ)
  let b : ℝ := (0.4 : ℝ) ^ (2/5 : ℝ)
  let c : ℝ := (0.4 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end relationship_abc_l4103_410389


namespace tom_barbados_cost_l4103_410320

/-- The total cost Tom has to pay for his trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let insurance_payment := medical_cost * insurance_coverage
  let out_of_pocket_medical := medical_cost - insurance_payment
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost Tom has to pay -/
theorem tom_barbados_cost : 
  total_cost 10 45 250 (4/5) 1200 = 1340 := by sorry

end tom_barbados_cost_l4103_410320


namespace line_slope_l4103_410398

/-- A straight line in the xy-plane with y-intercept 4 and passing through (199, 800) has slope 4 -/
theorem line_slope (m : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + 4) ∧ f 199 = 800) → m = 4 := by
  sorry

end line_slope_l4103_410398


namespace lucys_cake_packs_l4103_410399

/-- Lucy's grocery shopping problem -/
theorem lucys_cake_packs (cookies chocolate total : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) :
  total - (cookies + chocolate) = 22 := by
  sorry

end lucys_cake_packs_l4103_410399


namespace intersection_of_M_and_N_l4103_410348

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -2 < x ∧ x < -1} := by
  sorry

end intersection_of_M_and_N_l4103_410348


namespace power_difference_evaluation_l4103_410305

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16246775 := by
  sorry

end power_difference_evaluation_l4103_410305


namespace quadratic_minimum_l4103_410313

/-- The quadratic function f(x) = x^2 + 4x - 5 has a minimum value of -9 at x = -2 -/
theorem quadratic_minimum : ∃ (f : ℝ → ℝ), 
  (∀ x, f x = x^2 + 4*x - 5) ∧ 
  (∀ x, f x ≥ f (-2)) ∧
  f (-2) = -9 := by
  sorry

end quadratic_minimum_l4103_410313


namespace cubic_sum_coefficients_l4103_410334

/-- A cubic function f(x) = ax^3 + bx^2 + cx + d -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_sum_coefficients (a b c d : ℝ) :
  (∀ x, cubic_function a b c d (x + 2) = 2 * x^3 - x^2 + 5 * x + 3) →
  a + b + c + d = -5 := by
  sorry

end cubic_sum_coefficients_l4103_410334


namespace subset_implies_m_value_l4103_410396

theorem subset_implies_m_value (A B : Set ℝ) (m : ℝ) : 
  A = {-1} →
  B = {x : ℝ | x^2 + m*x - 3 = 1} →
  A ⊆ B →
  m = -3 := by sorry

end subset_implies_m_value_l4103_410396


namespace susana_viviana_vanilla_ratio_l4103_410378

/-- Prove that the ratio of Susana's vanilla chips to Viviana's vanilla chips is 3:4 -/
theorem susana_viviana_vanilla_ratio :
  let viviana_chocolate := susana_chocolate + 5
  let viviana_vanilla := 20
  let susana_chocolate := 25
  let total_chips := 90
  let susana_vanilla := total_chips - viviana_chocolate - susana_chocolate - viviana_vanilla
  (susana_vanilla : ℚ) / viviana_vanilla = 3 / 4 := by
  sorry

end susana_viviana_vanilla_ratio_l4103_410378


namespace complement_A_intersect_B_l4103_410333

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 2*x + 5)}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 4}

-- Define the universal set U
def U : Type := ℝ

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioo (-1 : ℝ) (2 : ℝ) :=
sorry

end complement_A_intersect_B_l4103_410333


namespace banana_bunches_l4103_410383

theorem banana_bunches (total_bananas : ℕ) (known_bunches : ℕ) (known_bananas_per_bunch : ℕ) 
  (unknown_bunches : ℕ) (h1 : total_bananas = 83) (h2 : known_bunches = 6) 
  (h3 : known_bananas_per_bunch = 8) (h4 : unknown_bunches = 5) : 
  (total_bananas - known_bunches * known_bananas_per_bunch) / unknown_bunches = 7 := by
  sorry

end banana_bunches_l4103_410383


namespace product_sum_difference_l4103_410366

theorem product_sum_difference (x y : ℝ) : x * y = 23 ∧ x + y = 24 → |x - y| = 22 := by
  sorry

end product_sum_difference_l4103_410366


namespace water_moles_equal_cao_moles_l4103_410306

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product : String

-- Define the molar quantities
structure MolarQuantities where
  cao_moles : ℝ
  h2o_moles : ℝ
  caoh2_moles : ℝ

-- Define the problem parameters
def cao_mass : ℝ := 168
def cao_molar_mass : ℝ := 56.08
def target_caoh2_moles : ℝ := 3

-- Define the reaction
def calcium_hydroxide_reaction : Reaction :=
  { reactant1 := "CaO", reactant2 := "H2O", product := "Ca(OH)2" }

-- Theorem statement
theorem water_moles_equal_cao_moles 
  (reaction : Reaction) 
  (quantities : MolarQuantities) :
  reaction = calcium_hydroxide_reaction →
  quantities.caoh2_moles = target_caoh2_moles →
  quantities.cao_moles = cao_mass / cao_molar_mass →
  quantities.h2o_moles = quantities.cao_moles :=
by sorry

end water_moles_equal_cao_moles_l4103_410306


namespace cross_product_example_l4103_410384

/-- The cross product of two 3D vectors -/
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => v 1 * w 2 - v 2 * w 1
  | 1 => v 2 * w 0 - v 0 * w 2
  | 2 => v 0 * w 1 - v 1 * w 0

/-- The first vector -/
def v : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -3
  | 1 => 4
  | 2 => 5

/-- The second vector -/
def w : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2
  | 1 => -1
  | 2 => 4

theorem cross_product_example : cross_product v w = fun i =>
  match i with
  | 0 => 21
  | 1 => 22
  | 2 => -5 := by sorry

end cross_product_example_l4103_410384


namespace vector_magnitude_l4103_410397

/-- The magnitude of the vector (-3, 4) is 5. -/
theorem vector_magnitude : Real.sqrt ((-3)^2 + 4^2) = 5 := by
  sorry

end vector_magnitude_l4103_410397


namespace pen_count_is_31_l4103_410392

/-- The number of pens after a series of events --/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

/-- Theorem stating that given the initial conditions, the final number of pens is 31 --/
theorem pen_count_is_31 : final_pen_count 5 20 2 19 = 31 := by
  sorry

end pen_count_is_31_l4103_410392


namespace power_equation_l4103_410315

theorem power_equation (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 := by
  sorry

end power_equation_l4103_410315


namespace inequality_solution_range_l4103_410343

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) →
  a ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end inequality_solution_range_l4103_410343


namespace simplify_radicals_l4103_410304

theorem simplify_radicals (y z : ℝ) (h : y ≥ 0 ∧ z ≥ 0) : 
  Real.sqrt (32 * y) * Real.sqrt (75 * z) * Real.sqrt (14 * y) = 40 * y * Real.sqrt (21 * z) := by
  sorry

end simplify_radicals_l4103_410304


namespace spheres_radius_l4103_410316

/-- A configuration of spheres in a unit cube -/
structure SpheresInCube where
  /-- The radius of each sphere -/
  radius : ℝ
  /-- The number of spheres is 8 -/
  num_spheres : Nat
  num_spheres_eq : num_spheres = 8
  /-- The cube is a unit cube -/
  cube_edge : ℝ
  cube_edge_eq : cube_edge = 1
  /-- Each sphere touches three adjacent spheres -/
  touches_adjacent : True
  /-- Spheres are inscribed in trihedral angles -/
  inscribed_in_angles : True

/-- The radius of spheres in the specific configuration is 1/4 -/
theorem spheres_radius (config : SpheresInCube) : config.radius = 1/4 := by
  sorry

end spheres_radius_l4103_410316


namespace train_length_l4103_410302

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 12 → speed_kmh * (1000 / 3600) * time_s = 240 := by
  sorry

#check train_length

end train_length_l4103_410302


namespace three_digit_equation_solutions_l4103_410318

theorem three_digit_equation_solutions :
  ∀ x y z : ℕ,
  (100 ≤ x ∧ x ≤ 999) ∧
  (100 ≤ y ∧ y ≤ 999) ∧
  (100 ≤ z ∧ z ≤ 999) ∧
  (17 * x + 15 * y - 28 * z = 61) ∧
  (19 * x - 25 * y + 12 * z = 31) →
  ((x = 265 ∧ y = 372 ∧ z = 358) ∨
   (x = 525 ∧ y = 740 ∧ z = 713)) :=
by sorry

end three_digit_equation_solutions_l4103_410318


namespace stamps_per_page_l4103_410350

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 924) 
  (h2 : book2 = 1386) 
  (h3 : book3 = 1848) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 462 := by
  sorry

end stamps_per_page_l4103_410350


namespace writing_time_for_three_books_l4103_410380

/-- Calculates the number of days required to write multiple books given the daily writing rate and book length. -/
def days_to_write_books (pages_per_day : ℕ) (pages_per_book : ℕ) (num_books : ℕ) : ℕ :=
  (pages_per_book * num_books) / pages_per_day

/-- Theorem stating that it takes 60 days to write 3 books of 400 pages each at a rate of 20 pages per day. -/
theorem writing_time_for_three_books :
  days_to_write_books 20 400 3 = 60 := by
  sorry

end writing_time_for_three_books_l4103_410380


namespace line_points_k_value_l4103_410357

/-- Given a line represented by equations x = 2y + 5 and z = 3x - 4,
    and two points (m, n, p) and (m + 4, n + k, p + 3) lying on this line,
    prove that k = 2 -/
theorem line_points_k_value
  (m n p k : ℝ)
  (point1_on_line : m = 2 * n + 5 ∧ p = 3 * m - 4)
  (point2_on_line : (m + 4) = 2 * (n + k) + 5 ∧ (p + 3) = 3 * (m + 4) - 4) :
  k = 2 :=
by sorry

end line_points_k_value_l4103_410357


namespace male_listeners_count_l4103_410321

/-- Represents the survey results of radio station XYZ -/
structure SurveyResults where
  total_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ
  total_non_listeners : ℕ

/-- Calculates the number of male listeners given the survey results -/
def male_listeners (survey : SurveyResults) : ℕ :=
  survey.total_listeners - survey.female_listeners

/-- Theorem stating that the number of male listeners is 85 -/
theorem male_listeners_count (survey : SurveyResults) 
  (h1 : survey.total_listeners = 160)
  (h2 : survey.female_listeners = 75) :
  male_listeners survey = 85 := by
  sorry

#eval male_listeners { total_listeners := 160, female_listeners := 75, male_non_listeners := 85, total_non_listeners := 180 }

end male_listeners_count_l4103_410321


namespace fixed_point_on_curve_l4103_410329

-- Define the curve equation
def curve_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Theorem statement
theorem fixed_point_on_curve :
  ∀ k : ℝ, k ≠ -1 → curve_equation k 1 (-3) :=
by
  sorry

end fixed_point_on_curve_l4103_410329


namespace triangle_properties_l4103_410395

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.a) * Real.cos t.B - t.b * Real.cos t.A = 0)
  (h2 : t.a + t.c = 6)
  (h3 : t.b = 2 * Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
by sorry

end triangle_properties_l4103_410395


namespace range_of_a_lower_bound_of_f_l4103_410372

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : f a 1 < 3 → -2/3 < a ∧ a < 4/3 := by sorry

-- Theorem for the lower bound of f(x)
theorem lower_bound_of_f (a x : ℝ) : a ≥ 1 → f a x ≥ 2 := by sorry

end range_of_a_lower_bound_of_f_l4103_410372


namespace find_n_l4103_410330

theorem find_n : ∃ n : ℝ, (256 : ℝ)^(1/4) = 4^n ∧ n = 1 := by sorry

end find_n_l4103_410330


namespace hash_solution_l4103_410385

/-- Definition of the # operation -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that 2 is the number that satisfies x # 3 = 12 -/
theorem hash_solution : ∃ x : ℝ, hash x 3 = 12 ∧ x = 2 := by sorry

end hash_solution_l4103_410385


namespace total_amount_shared_l4103_410375

/-- Given that x gets 25% more than y, y gets 20% more than z, and z's share is 400,
    prove that the total amount shared between x, y, and z is 1480. -/
theorem total_amount_shared (x y z : ℝ) : 
  x = 1.25 * y → y = 1.2 * z → z = 400 → x + y + z = 1480 := by
  sorry

end total_amount_shared_l4103_410375


namespace walking_speed_problem_l4103_410363

/-- The speed of person P in miles per hour -/
def speed_P : ℝ := 7.5

/-- The speed of person Q in miles per hour -/
def speed_Q : ℝ := speed_P + 3

/-- The distance between Town X and Town Y in miles -/
def distance : ℝ := 90

/-- The distance from the meeting point to Town Y in miles -/
def meeting_distance : ℝ := 15

theorem walking_speed_problem :
  (distance - meeting_distance) / speed_P = (distance + meeting_distance) / speed_Q :=
sorry

end walking_speed_problem_l4103_410363


namespace probability_of_common_books_l4103_410358

def total_books : ℕ := 12
def books_chosen : ℕ := 6
def books_in_common : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books books_in_common * 
   Nat.choose (total_books - books_in_common) (books_chosen - books_in_common) * 
   Nat.choose (total_books - books_chosen) (books_chosen - books_in_common)) / 
  (Nat.choose total_books books_chosen * Nat.choose total_books books_chosen) = 50 / 116 := by
  sorry

end probability_of_common_books_l4103_410358


namespace range_of_f_l4103_410353

-- Define the function f
def f (x : ℝ) : ℝ := (x^3 - 3*x + 1)^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end range_of_f_l4103_410353


namespace initial_members_count_l4103_410371

/-- The number of initial earning members in a family -/
def initial_members : ℕ := sorry

/-- The initial average monthly income of the family -/
def initial_average : ℕ := 735

/-- The new average monthly income after one member died -/
def new_average : ℕ := 590

/-- The income of the deceased member -/
def deceased_income : ℕ := 1170

/-- Theorem stating the number of initial earning members -/
theorem initial_members_count : initial_members = 4 := by
  sorry

end initial_members_count_l4103_410371


namespace lattice_triangle_area_bound_l4103_410342

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Checks if a point is on the edge of a triangle -/
def isOnEdge (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Counts the number of lattice points inside a triangle -/
def interiorPointCount (t : LatticeTriangle) : ℕ := sorry

/-- Counts the number of lattice points on the edges of a triangle -/
def boundaryPointCount (t : LatticeTriangle) : ℕ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : LatticeTriangle) : ℚ := sorry

/-- Theorem: The area of a lattice triangle with exactly one interior lattice point is at most 9/2 -/
theorem lattice_triangle_area_bound (t : LatticeTriangle) 
  (h : interiorPointCount t = 1) : 
  triangleArea t ≤ 9/2 := by sorry

end lattice_triangle_area_bound_l4103_410342


namespace smallest_two_digit_prime_with_reversed_prime_ending_in_3_l4103_410377

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_reversed_prime_ending_in_3 :
  ∃ (p : ℕ), is_two_digit p ∧ 
             Nat.Prime p ∧
             Nat.Prime (reverse_digits p) ∧
             reverse_digits p % 10 = 3 ∧
             (∀ (q : ℕ), is_two_digit q → 
                         Nat.Prime q → 
                         Nat.Prime (reverse_digits q) → 
                         reverse_digits q % 10 = 3 → 
                         p ≤ q) ∧
             p = 13 :=
by sorry

end smallest_two_digit_prime_with_reversed_prime_ending_in_3_l4103_410377


namespace triangle_property_l4103_410317

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  (t.a - 5)^2 + |t.b - 12| + (t.c - 13)^2 = 0

-- Define what it means to be a right triangle with c as hypotenuse
def is_right_triangle_with_c_hypotenuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) (h : satisfies_condition t) :
  is_right_triangle_with_c_hypotenuse t :=
sorry

end triangle_property_l4103_410317


namespace purely_imaginary_roots_l4103_410326

theorem purely_imaginary_roots (z : ℂ) (k : ℝ) : 
  (∀ r : ℂ, 20 * r^2 + 6 * Complex.I * r - k = 0 → ∃ b : ℝ, r = Complex.I * b) ↔ k = 9/5 := by
  sorry

end purely_imaginary_roots_l4103_410326


namespace elaines_earnings_increase_l4103_410308

-- Define Elaine's earnings last year
variable (E : ℝ)

-- Define the percentage increase in earnings
variable (P : ℝ)

-- Theorem statement
theorem elaines_earnings_increase :
  -- Last year's rent spending
  (0.20 * E) > 0 →
  -- This year's rent spending is 143.75% of last year's
  (0.25 * (E * (1 + P / 100))) = (1.4375 * (0.20 * E)) →
  -- Conclusion: Earnings increased by 15%
  P = 15 := by
sorry

end elaines_earnings_increase_l4103_410308


namespace accidental_addition_l4103_410347

theorem accidental_addition (x : ℕ) : x + 65 = 125 → x + 95 = 155 := by
  sorry

end accidental_addition_l4103_410347


namespace min_value_of_f_l4103_410309

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 6*x - 8

-- Theorem stating that f(x) achieves its minimum when x = -3
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -3 := by
  sorry

end min_value_of_f_l4103_410309


namespace odd_power_function_l4103_410370

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m

theorem odd_power_function (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f m x = -f m (-x)) →
  f m (m + 1) = 8 :=
by sorry

end odd_power_function_l4103_410370


namespace cafe_tables_l4103_410369

-- Define the seating capacity in base 8
def seating_capacity_base8 : ℕ := 312

-- Define the number of people per table
def people_per_table : ℕ := 3

-- Define the function to convert from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Theorem statement
theorem cafe_tables :
  (base8_to_base10 seating_capacity_base8) / people_per_table = 67 := by
  sorry

end cafe_tables_l4103_410369


namespace triangle_properties_l4103_410301

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (BD : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  c * Real.sin ((A + C) / 2) = b * Real.sin C ∧
  BD = 1 ∧
  b = Real.sqrt 3 ∧
  BD * (a * Real.sin C) = b * c * Real.sin (π / 2) →
  B = π / 3 ∧ 
  a + b + c = 3 + Real.sqrt 3 := by
sorry

end triangle_properties_l4103_410301


namespace log_equation_solution_l4103_410339

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ (x : ℝ), log (2^x) (3^20) = log (2^(x+3)) (3^2020) → x = 3/100 :=
by sorry

end log_equation_solution_l4103_410339


namespace smallest_share_amount_l4103_410381

def total_amount : ℝ := 500
def give_away_percentage : ℝ := 0.60
def friend_count : ℕ := 5
def shares : List ℝ := [0.30, 0.25, 0.20, 0.15, 0.10]

theorem smallest_share_amount :
  let amount_to_distribute := total_amount * give_away_percentage
  let smallest_share := shares.minimum?
  smallest_share.map (λ s => s * amount_to_distribute) = some 30 := by sorry

end smallest_share_amount_l4103_410381


namespace landmark_visit_sequences_l4103_410332

theorem landmark_visit_sequences (n : Nat) (h : n = 5) : 
  (List.permutations (List.range n)).length = 120 := by
  sorry

end landmark_visit_sequences_l4103_410332


namespace min_distance_to_circle_l4103_410312

theorem min_distance_to_circle (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  (∃ (min : ℝ), min = Real.sqrt 5 - 1 ∧ 
    ∀ (u v : ℝ), u^2 + v^2 = 1 → 
      Real.sqrt ((u - 1)^2 + (v - 2)^2) ≥ min) :=
by sorry

end min_distance_to_circle_l4103_410312


namespace a_range_l4103_410354

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (a : ℝ) :
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end a_range_l4103_410354


namespace bounded_figure_at_most_one_center_no_figure_exactly_two_centers_finite_set_at_most_three_almost_centers_l4103_410331

-- Define a type for figures
structure Figure where
  isBounded : Bool

-- Define a type for sets of points
structure PointSet where
  isFinite : Bool

-- Define a function to count centers of symmetry
def countCentersOfSymmetry (f : Figure) : Nat :=
  sorry

-- Define a function to count almost centers of symmetry
def countAlmostCentersOfSymmetry (s : PointSet) : Nat :=
  sorry

-- Theorem 1: A bounded figure has at most one center of symmetry
theorem bounded_figure_at_most_one_center (f : Figure) (h : f.isBounded = true) :
  countCentersOfSymmetry f ≤ 1 :=
sorry

-- Theorem 2: No figure can have exactly two centers of symmetry
theorem no_figure_exactly_two_centers (f : Figure) :
  countCentersOfSymmetry f ≠ 2 :=
sorry

-- Theorem 3: A finite set of points has at most 3 almost centers of symmetry
theorem finite_set_at_most_three_almost_centers (s : PointSet) (h : s.isFinite = true) :
  countAlmostCentersOfSymmetry s ≤ 3 :=
sorry

end bounded_figure_at_most_one_center_no_figure_exactly_two_centers_finite_set_at_most_three_almost_centers_l4103_410331


namespace largest_n_binomial_equality_l4103_410319

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n)) ∧
  (∀ m : ℕ, m > 7 → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m) :=
by sorry

end largest_n_binomial_equality_l4103_410319


namespace dodecagon_enclosure_l4103_410394

theorem dodecagon_enclosure (m : ℕ) (n : ℕ) : 
  m = 12 →
  (360 : ℝ) / n = (180 : ℝ) - (m - 2 : ℝ) * 180 / m →
  n = 6 :=
sorry

end dodecagon_enclosure_l4103_410394


namespace tan_315_degrees_l4103_410300

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l4103_410300


namespace point_p_coordinates_l4103_410391

/-- Given points A, B, C in ℝ³ and a point P such that vector AP is half of vector CB,
    prove that P has the specified coordinates. -/
theorem point_p_coordinates (A B C P : ℝ × ℝ × ℝ) : 
  A = (2, -1, 2) → 
  B = (4, 5, -1) → 
  C = (-2, 2, 3) → 
  P - A = (1/2 : ℝ) • (B - C) → 
  P = (5, 1/2, 0) := by
sorry


end point_p_coordinates_l4103_410391


namespace all_lines_through_single_point_l4103_410364

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for lines
structure Line where
  color : Color

-- Define a type for points
structure Point where

-- Define the plane
structure Plane where
  lines : Finset Line
  points : Set Point

-- Define the property that no lines are parallel
def NoParallelLines (p : Plane) : Prop :=
  ∀ l1 l2 : Line, l1 ∈ p.lines → l2 ∈ p.lines → l1 ≠ l2 → ∃ pt : Point, pt ∈ p.points

-- Define the property that through each intersection point of same-color lines passes a line of the other color
def IntersectionProperty (p : Plane) : Prop :=
  ∀ pt : Point, pt ∈ p.points →
    ∀ l1 l2 : Line, l1 ∈ p.lines → l2 ∈ p.lines → l1.color = l2.color →
      ∃ l3 : Line, l3 ∈ p.lines ∧ l3.color ≠ l1.color

-- The main theorem
theorem all_lines_through_single_point (p : Plane) 
  (h1 : Finite p.lines)
  (h2 : NoParallelLines p)
  (h3 : IntersectionProperty p) :
  ∃ pt : Point, ∀ l : Line, l ∈ p.lines → pt ∈ p.points :=
sorry

end all_lines_through_single_point_l4103_410364


namespace arithmetic_calculations_l4103_410327

theorem arithmetic_calculations :
  ((-10) + (-7) - 3 + 2 = -18) ∧
  ((-2)^3 / 4 - (-1)^2023 + |(-6)| * (-1) = -7) ∧
  ((1/3 - 1/4 + 5/6) * (-24) = -22) := by
  sorry

end arithmetic_calculations_l4103_410327


namespace balloon_count_l4103_410340

/-- The number of blue balloons after a series of events --/
def total_balloons (joan_initial : ℕ) (joan_popped : ℕ) (jessica_initial : ℕ) (jessica_inflated : ℕ) (peter_initial : ℕ) (peter_deflated : ℕ) : ℕ :=
  (joan_initial - joan_popped) + (jessica_initial + jessica_inflated) + (peter_initial - peter_deflated)

/-- Theorem stating the total number of balloons after the given events --/
theorem balloon_count :
  total_balloons 9 5 2 3 4 2 = 11 := by
  sorry

#eval total_balloons 9 5 2 3 4 2

end balloon_count_l4103_410340


namespace factorial_ratio_l4103_410328

theorem factorial_ratio (N : ℕ) (h : N > 1) :
  (Nat.factorial (N^2 - 1)) / ((Nat.factorial (N + 1))^2) = 
  (Nat.factorial (N - 1)) / (N + 1) :=
sorry

end factorial_ratio_l4103_410328


namespace sum_coordinates_point_D_l4103_410345

/-- Given a point N which is the midpoint of segment CD, and point C,
    prove that the sum of coordinates of point D is 5. -/
theorem sum_coordinates_point_D (N C D : ℝ × ℝ) : 
  N = (3, 5) →
  C = (1, 10) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 5 := by
  sorry

end sum_coordinates_point_D_l4103_410345


namespace bailey_shot_percentage_l4103_410393

theorem bailey_shot_percentage (total_shots : ℕ) (scored_shots : ℕ) 
  (h1 : total_shots = 8) (h2 : scored_shots = 6) : 
  (1 - scored_shots / total_shots) * 100 = 25 := by
  sorry

end bailey_shot_percentage_l4103_410393


namespace triangle_inequality_expression_negative_l4103_410356

/-- Given a triangle with side lengths a, b, and c, 
    the expression a^2 - c^2 - 2ab + b^2 is always negative. -/
theorem triangle_inequality_expression_negative 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 - c^2 - 2*a*b + b^2 < 0 := by
  sorry

end triangle_inequality_expression_negative_l4103_410356


namespace test_results_l4103_410390

/-- The probability of exactly two people meeting the standard in a test where
    A, B, and C have independent probabilities of 2/5, 3/4, and 1/2 respectively. -/
def prob_two_meet_standard : ℚ := 17/40

/-- The most likely number of people to meet the standard in the test. -/
def most_likely_number : ℕ := 2

/-- Probabilities of A, B, and C meeting the standard -/
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/2

theorem test_results :
  (prob_two_meet_standard = prob_A * prob_B * (1 - prob_C) +
                            prob_A * (1 - prob_B) * prob_C +
                            (1 - prob_A) * prob_B * prob_C) ∧
  (most_likely_number = 2) := by
  sorry

end test_results_l4103_410390


namespace x_value_at_y_25_l4103_410324

/-- The constant ratio between (4x - 5) and (2y + 20) -/
def k : ℚ := (4 * 1 - 5) / (2 * 5 + 20)

/-- Theorem stating that given the constant ratio k and the initial condition,
    x equals 2/3 when y equals 25 -/
theorem x_value_at_y_25 (x y : ℚ) 
  (h1 : (4 * x - 5) / (2 * y + 20) = k) 
  (h2 : x = 1 → y = 5) :
  y = 25 → x = 2/3 := by
  sorry

end x_value_at_y_25_l4103_410324


namespace country_y_total_exports_l4103_410361

/-- Proves that the total yearly exports of country Y are $127.5 million given the specified conditions -/
theorem country_y_total_exports :
  ∀ (total_exports : ℝ),
  (0.2 * total_exports * (1/6) = 4.25) →
  total_exports = 127.5 := by
sorry

end country_y_total_exports_l4103_410361


namespace monthly_snake_feeding_cost_l4103_410349

/-- Proves that the monthly cost per snake is $10, given Harry's pet ownership and feeding costs. -/
theorem monthly_snake_feeding_cost (num_geckos num_iguanas num_snakes : ℕ)
  (gecko_cost iguana_cost : ℚ) (total_annual_cost : ℚ) :
  num_geckos = 3 →
  num_iguanas = 2 →
  num_snakes = 4 →
  gecko_cost = 15 →
  iguana_cost = 5 →
  total_annual_cost = 1140 →
  (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * 10) * 12 = total_annual_cost :=
by sorry

end monthly_snake_feeding_cost_l4103_410349


namespace sum_of_common_elements_l4103_410386

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- The sequence of common elements between the arithmetic and geometric progressions -/
def common_sequence (n : ℕ) : ℕ := 40 * 4^n

theorem sum_of_common_elements :
  (Finset.range 10).sum common_sequence = 13981000 := by
  sorry

end sum_of_common_elements_l4103_410386


namespace arithmetic_seq_problem_l4103_410360

def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_seq_problem (a : ℕ → ℚ) :
  is_arithmetic_seq a →
  a 4 + a 5 + a 6 + a 7 = 56 →
  a 4 * a 7 = 187 →
  ((∃ a₁ d, ∀ n, a n = a₁ + (n - 1) * d) ∧ 
   ((a 1 = 5 ∧ ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 2) ∨
    (a 1 = 23 ∧ ∃ d, ∀ n, a n = 23 + (n - 1) * d ∧ d = -2))) :=
by sorry

end arithmetic_seq_problem_l4103_410360


namespace steve_height_l4103_410388

/-- Converts feet and inches to total inches -/
def feet_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Calculates final height after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_to_inches initial_feet initial_inches + growth

theorem steve_height :
  final_height 5 6 6 = 72 := by sorry

end steve_height_l4103_410388


namespace faucet_filling_time_l4103_410307

/-- Given that five faucets can fill a 150-gallon tub in 10 minutes,
    prove that ten faucets will fill a 50-gallon tub in 100 seconds. -/
theorem faucet_filling_time 
  (fill_rate : ℝ)  -- Rate at which one faucet fills in gallons per minute
  (h1 : 5 * fill_rate * 10 = 150)  -- Five faucets fill 150 gallons in 10 minutes
  : 10 * fill_rate * (100 / 60) = 50  -- Ten faucets fill 50 gallons in 100 seconds
  := by sorry

end faucet_filling_time_l4103_410307


namespace equal_distribution_proof_l4103_410352

theorem equal_distribution_proof (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / 3 = 115 :=
by
  sorry

end equal_distribution_proof_l4103_410352


namespace not_like_terms_example_l4103_410379

/-- Definition of a monomial -/
structure Monomial (α : Type*) [CommRing α] :=
  (coeff : α)
  (vars : List (α × ℕ))

/-- Definition of like terms -/
def are_like_terms {α : Type*} [CommRing α] (m1 m2 : Monomial α) : Prop :=
  m1.vars.map Prod.fst = m2.vars.map Prod.fst ∧
  m1.vars.map Prod.snd = m2.vars.map Prod.snd

/-- The main theorem -/
theorem not_like_terms_example {α : Type*} [CommRing α] :
  ¬ are_like_terms 
    (Monomial.mk 7 [(a, 2), (n, 1)])
    (Monomial.mk (-9) [(a, 1), (n, 2)]) :=
sorry

end not_like_terms_example_l4103_410379


namespace number_satisfying_condition_l4103_410310

theorem number_satisfying_condition : ∃ x : ℝ, 0.65 * x = 0.8 * x - 21 ∧ x = 140 := by
  sorry

end number_satisfying_condition_l4103_410310


namespace same_roots_implies_a_equals_five_l4103_410376

theorem same_roots_implies_a_equals_five :
  ∀ (a : ℝ),
  (∀ x : ℝ, (|x|^2 - 3*|x| + 2 = 0) ↔ (x^4 - a*x^2 + 4 = 0)) →
  a = 5 :=
by sorry

end same_roots_implies_a_equals_five_l4103_410376


namespace charm_bracelet_profit_l4103_410323

/-- Calculates the profit from selling charm bracelets -/
theorem charm_bracelet_profit
  (string_cost : ℕ)
  (bead_cost : ℕ)
  (selling_price : ℕ)
  (bracelets_sold : ℕ)
  (h1 : string_cost = 1)
  (h2 : bead_cost = 3)
  (h3 : selling_price = 6)
  (h4 : bracelets_sold = 25) :
  (selling_price * bracelets_sold) - ((string_cost + bead_cost) * bracelets_sold) = 50 :=
by sorry

end charm_bracelet_profit_l4103_410323


namespace cube_painted_faces_l4103_410382

/-- Calculates the number of unit cubes with exactly one painted side in a painted cube of given side length -/
def painted_faces (side_length : ℕ) : ℕ :=
  if side_length ≤ 2 then 0
  else 6 * (side_length - 2)^2

/-- The problem statement -/
theorem cube_painted_faces :
  painted_faces 5 = 54 := by
  sorry

end cube_painted_faces_l4103_410382


namespace profit_percentage_formula_l4103_410336

theorem profit_percentage_formula (C S M n : ℝ) (P : ℝ) 
  (h1 : S > 0) 
  (h2 : C > 0)
  (h3 : n > 0)
  (h4 : M = (2 / n) * C) 
  (h5 : P = (M / S) * 100) :
  P = 200 / (n + 2) := by
sorry

end profit_percentage_formula_l4103_410336


namespace number_to_add_divisibility_l4103_410338

theorem number_to_add_divisibility (p q : ℕ) (n m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p = 563 → q = 839 → n = 1398547 → m = 18284 →
  (p * q) ∣ (n + m) :=
by sorry

end number_to_add_divisibility_l4103_410338


namespace geometric_sequence_first_term_l4103_410322

/-- Given a geometric sequence {aₙ}, prove that if a₃ = 16 and a₄ = 8, then a₁ = 64. -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- Definition of geometric sequence
  a 3 = 16 →                                -- Condition: a₃ = 16
  a 4 = 8 →                                 -- Condition: a₄ = 8
  a 1 = 64 :=                               -- Conclusion: a₁ = 64
by sorry

end geometric_sequence_first_term_l4103_410322


namespace hockey_arena_seating_l4103_410344

/-- The minimum number of rows required to seat students in a hockey arena --/
def min_rows (seats_per_row : ℕ) (total_students : ℕ) (max_students_per_school : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rows required for the given conditions --/
theorem hockey_arena_seating 
  (seats_per_row : ℕ) 
  (total_students : ℕ) 
  (max_students_per_school : ℕ) 
  (h1 : seats_per_row = 168)
  (h2 : total_students = 2016)
  (h3 : max_students_per_school = 45)
  (h4 : ∀ (school : ℕ), school ≤ total_students → school ≤ max_students_per_school) :
  min_rows seats_per_row total_students max_students_per_school = 16 :=
sorry

end hockey_arena_seating_l4103_410344


namespace complex_absolute_value_product_l4103_410314

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 6 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 18 * Real.sqrt 6 := by
  sorry

end complex_absolute_value_product_l4103_410314


namespace tenth_term_of_arithmetic_sequence_l4103_410351

/-- An arithmetic sequence with given second and third terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end tenth_term_of_arithmetic_sequence_l4103_410351


namespace count_even_one_matrices_l4103_410346

/-- The number of m × n matrices with entries 0 or 1, where the number of 1's in each row and column is even -/
def evenOneMatrices (m n : ℕ) : ℕ :=
  2^((m-1)*(n-1))

/-- Theorem stating that the number of m × n matrices with entries 0 or 1, 
    where the number of 1's in each row and column is even, is 2^((m-1)(n-1)) -/
theorem count_even_one_matrices (m n : ℕ) :
  evenOneMatrices m n = 2^((m-1)*(n-1)) := by
  sorry

end count_even_one_matrices_l4103_410346


namespace gcd_and_sum_of_1729_and_867_l4103_410374

theorem gcd_and_sum_of_1729_and_867 :
  (Nat.gcd 1729 867 = 1) ∧ (1729 + 867 = 2596) := by
  sorry

end gcd_and_sum_of_1729_and_867_l4103_410374


namespace factorization_cubic_factorization_fifth_power_l4103_410311

-- We don't need to prove the first part as no specific factorization was provided

-- Prove the factorization of x^3 + 2x^2 + 4x + 3
theorem factorization_cubic (x : ℝ) : 
  x^3 + 2*x^2 + 4*x + 3 = (x + 1) * (x^2 + x + 3) := by
sorry

-- Prove the factorization of x^5 - 1
theorem factorization_fifth_power (x : ℝ) : 
  x^5 - 1 = (x - 1) * (x^4 + x^3 + x^2 + x + 1) := by
sorry

end factorization_cubic_factorization_fifth_power_l4103_410311


namespace journey_distance_l4103_410303

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = total_time * (speed1 + speed2) / 2 ∧ 
    distance = 224 := by
  sorry

end journey_distance_l4103_410303


namespace count_integers_satisfying_inequality_l4103_410387

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset Int),
    (∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0) ∧
    (∀ n, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 → n ∈ S) ∧
    S.card = 8 := by
  sorry

end count_integers_satisfying_inequality_l4103_410387


namespace ac_squared_gt_bc_squared_implies_a_gt_b_l4103_410355

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) :
  a * c^2 > b * c^2 → a > b := by sorry

end ac_squared_gt_bc_squared_implies_a_gt_b_l4103_410355


namespace plane_equation_correct_l4103_410368

def plane_equation (x y z : ℝ) : ℝ := 3 * x - y + 2 * z - 11

theorem plane_equation_correct :
  ∃ (A B C D : ℤ),
    (∀ (s t : ℝ),
      plane_equation (2 + 2*s - 2*t) (3 - 2*s) (4 - s + 3*t) = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    ∀ (x y z : ℝ), A * x + B * y + C * z + D = plane_equation x y z := by
  sorry

end plane_equation_correct_l4103_410368


namespace jame_tear_frequency_l4103_410337

/-- Represents the number of times Jame tears cards per week -/
def tear_frequency (cards_per_tear : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) (num_weeks : ℕ) : ℕ :=
  (cards_per_deck * num_decks) / (cards_per_tear * num_weeks)

/-- Theorem stating that Jame tears cards 3 times a week given the conditions -/
theorem jame_tear_frequency :
  let cards_per_tear := 30
  let cards_per_deck := 55
  let num_decks := 18
  let num_weeks := 11
  tear_frequency cards_per_tear cards_per_deck num_decks num_weeks = 3 := by
  sorry


end jame_tear_frequency_l4103_410337


namespace y_intercept_of_line_l4103_410335

/-- The y-intercept of the line 2x + 7y = 35 is (0, 5) -/
theorem y_intercept_of_line (x y : ℝ) :
  2 * x + 7 * y = 35 → y = 5 ∧ x = 0 := by
  sorry

end y_intercept_of_line_l4103_410335


namespace factor_implies_absolute_value_l4103_410362

/-- Given a polynomial 3x^4 - mx^2 + nx - p with factors (x-3) and (x+4), 
    prove that |m+2n-4p| = 20 -/
theorem factor_implies_absolute_value (m n p : ℤ) : 
  (∃ (a b : ℤ), (3 * X^4 - m * X^2 + n * X - p) = 
    (X - 3) * (X + 4) * (a * X^2 + b * X + (3 * a - 4 * b))) →
  |m + 2*n - 4*p| = 20 := by
  sorry


end factor_implies_absolute_value_l4103_410362


namespace solutions_equality_l4103_410359

-- Define a as a positive real number
variable (a : ℝ) (ha : a > 0)

-- Define the condition that 10 < a^x < 100 has exactly five solutions in natural numbers
def has_five_solutions (a : ℝ) : Prop :=
  (∃ (s : Finset ℕ), s.card = 5 ∧ ∀ x : ℕ, x ∈ s ↔ (10 < a^x ∧ a^x < 100))

-- Theorem statement
theorem solutions_equality (h : has_five_solutions a) :
  ∃ (s : Finset ℕ), s.card = 5 ∧ ∀ x : ℕ, x ∈ s ↔ (100 < a^x ∧ a^x < 1000) :=
sorry

end solutions_equality_l4103_410359


namespace acme_horseshoe_problem_l4103_410365

/-- Acme's horseshoe manufacturing problem -/
theorem acme_horseshoe_problem (initial_outlay : ℝ) : 
  let cost_per_set : ℝ := 20.75
  let selling_price : ℝ := 50
  let num_sets : ℕ := 950
  let profit : ℝ := 15337.5
  let revenue : ℝ := selling_price * num_sets
  let total_cost : ℝ := initial_outlay + cost_per_set * num_sets
  profit = revenue - total_cost →
  initial_outlay = 12450 := by
  sorry

#check acme_horseshoe_problem

end acme_horseshoe_problem_l4103_410365


namespace pool_filling_time_l4103_410325

theorem pool_filling_time (faster_pipe_rate : ℝ) (slower_pipe_rate : ℝ) :
  faster_pipe_rate = 1 / 9 →
  slower_pipe_rate = faster_pipe_rate / 1.25 →
  1 / (faster_pipe_rate + slower_pipe_rate) = 5 := by
  sorry

end pool_filling_time_l4103_410325


namespace rectangle_dimensions_l4103_410373

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3 > 0) →
  (2*x - 1 > 0) →
  (x + 3) * (2*x - 1) = 12*x + 5 →
  x = (7 + Real.sqrt 113) / 4 :=
by sorry

end rectangle_dimensions_l4103_410373


namespace container_volume_maximized_l4103_410341

/-- The total length of the steel bar used to make the container frame -/
def total_length : ℝ := 14.8

/-- The function representing the volume of the container -/
def volume (width : ℝ) : ℝ :=
  width * (width + 0.5) * (3.2 - 2 * width)

/-- The width that maximizes the container's volume -/
def optimal_width : ℝ := 1

theorem container_volume_maximized :
  ∀ w : ℝ, 0 < w → w < 1.6 → volume w ≤ volume optimal_width :=
sorry

end container_volume_maximized_l4103_410341


namespace imaginary_part_of_z_l4103_410367

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l4103_410367
