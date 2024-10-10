import Mathlib

namespace square_root_domain_only_five_satisfies_l186_18604

theorem square_root_domain (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) ↔ x ≥ 4 :=
sorry

theorem only_five_satisfies : 
  (∃ y : ℝ, y^2 = 5 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 0 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 1 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 2 - 4) :=
sorry

end square_root_domain_only_five_satisfies_l186_18604


namespace intersection_points_count_l186_18615

/-- A line in the 2D plane --/
inductive Line
  | General (a b c : ℝ) : Line  -- ax + by + c = 0
  | Vertical (x : ℝ) : Line     -- x = k
  | Horizontal (y : ℝ) : Line   -- y = k

/-- Check if a point (x, y) is on a line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  match l with
  | General a b c => a * x + b * y + c = 0
  | Vertical k => x = k
  | Horizontal k => y = k

/-- The set of lines given in the problem --/
def problem_lines : List Line :=
  [Line.General 3 (-1) (-1), Line.General 1 2 (-5), Line.Vertical 3, Line.Horizontal 1]

/-- A point is an intersection point if it's contained in at least two distinct lines --/
def is_intersection_point (x y : ℝ) (lines : List Line) : Prop :=
  ∃ l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ l1.contains x y ∧ l2.contains x y

/-- The theorem to be proved --/
theorem intersection_points_count :
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    (∀ x y : ℝ, is_intersection_point x y problem_lines ↔ (x, y) = p1 ∨ (x, y) = p2) :=
  sorry

end intersection_points_count_l186_18615


namespace largest_two_digit_multiple_of_3_and_5_l186_18694

theorem largest_two_digit_multiple_of_3_and_5 : 
  ∃ n : ℕ, n = 90 ∧ 
  n ≥ 10 ∧ n < 100 ∧ 
  n % 3 = 0 ∧ n % 5 = 0 ∧
  ∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m % 3 = 0 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end largest_two_digit_multiple_of_3_and_5_l186_18694


namespace union_equals_M_l186_18682

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≠ 0}

theorem union_equals_M : M ∪ N = M := by sorry

end union_equals_M_l186_18682


namespace max_balls_theorem_l186_18658

/-- The maximum number of balls that can be counted while maintaining at least 90% red balls -/
def max_balls : ℕ := 210

/-- The proportion of red balls in the first 50 counted -/
def initial_red_ratio : ℚ := 49 / 50

/-- The proportion of red balls in each subsequent batch of 8 -/
def subsequent_red_ratio : ℚ := 7 / 8

/-- The minimum required proportion of red balls -/
def min_red_ratio : ℚ := 9 / 10

/-- Theorem stating that max_balls is the maximum number of balls that can be counted
    while maintaining at least 90% red balls -/
theorem max_balls_theorem (n : ℕ) :
  n ≤ max_balls ↔
  (∃ x : ℕ, n = 50 + 8 * x ∧
    (initial_red_ratio * 50 + subsequent_red_ratio * 8 * x) / n ≥ min_red_ratio) :=
sorry

end max_balls_theorem_l186_18658


namespace equation_solution_l186_18623

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end equation_solution_l186_18623


namespace feathers_per_pound_is_300_l186_18661

/-- Represents the number of feathers in a goose -/
def goose_feathers : ℕ := 3600

/-- Represents the number of pillows that can be stuffed with one goose's feathers -/
def pillows_per_goose : ℕ := 6

/-- Represents the number of pounds of feathers needed for each pillow -/
def pounds_per_pillow : ℕ := 2

/-- Calculates the number of feathers in a pound of goose feathers -/
def feathers_per_pound : ℕ := goose_feathers / (pillows_per_goose * pounds_per_pillow)

theorem feathers_per_pound_is_300 : feathers_per_pound = 300 := by
  sorry

end feathers_per_pound_is_300_l186_18661


namespace quadratic_factorization_sum_l186_18677

theorem quadratic_factorization_sum (a w c d : ℝ) : 
  (∀ x, 6 * x^2 + x - 12 = (a * x + w) * (c * x + d)) →
  |a| + |w| + |c| + |d| = 12 := by
  sorry

end quadratic_factorization_sum_l186_18677


namespace abs_inequality_l186_18655

theorem abs_inequality (x : ℝ) : |5 - x| > 6 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 11 :=
sorry

end abs_inequality_l186_18655


namespace length_of_diagonal_l186_18638

/-- Given two triangles AOC and BOD sharing a vertex O, with specified side lengths,
    prove that the length of AC is √1036/7 -/
theorem length_of_diagonal (AO BO CO DO BD : ℝ) (x : ℝ) 
    (h1 : AO = 3)
    (h2 : CO = 5)
    (h3 : BO = 7)
    (h4 : DO = 6)
    (h5 : BD = 11)
    (h6 : x = Real.sqrt (AO^2 + CO^2 - 2*AO*CO*(BO^2 + DO^2 - BD^2)/(2*BO*DO))) :
  x = Real.sqrt 1036 / 7 := by
  sorry

end length_of_diagonal_l186_18638


namespace max_value_complex_expression_l186_18663

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (max_val : ℝ), max_val = 24 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 2 →
    Complex.abs ((w - 2)^3 * (w + 2)) ≤ max_val :=
sorry

end max_value_complex_expression_l186_18663


namespace binary_101111011_equals_379_l186_18688

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101111011_equals_379 :
  binary_to_decimal [true, true, false, true, true, true, true, false, true] = 379 := by
  sorry

end binary_101111011_equals_379_l186_18688


namespace simplify_sum_of_square_roots_l186_18643

theorem simplify_sum_of_square_roots : 
  Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_sum_of_square_roots_l186_18643


namespace range_of_r_l186_18644

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ r x = y) ↔ y ≥ 16 :=
by sorry

end range_of_r_l186_18644


namespace investment_growth_l186_18612

/-- Calculates the final amount after simple interest --/
def finalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, the final amount after 5 years is $350 --/
theorem investment_growth (principal : ℝ) (amount_after_2_years : ℝ) :
  principal = 200 →
  amount_after_2_years = 260 →
  finalAmount principal ((amount_after_2_years - principal) / (principal * 2)) 5 = 350 :=
by
  sorry


end investment_growth_l186_18612


namespace fixed_point_of_exponential_function_l186_18628

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 3
  f 2 = -2 := by sorry

end fixed_point_of_exponential_function_l186_18628


namespace period_length_divides_totient_l186_18695

-- Define L(m) as the period length of the decimal expansion of 1/m
def L (m : ℕ) : ℕ := sorry

-- State the theorem
theorem period_length_divides_totient (m : ℕ) (h : Nat.gcd m 10 = 1) : 
  L m ∣ Nat.totient m := by sorry

end period_length_divides_totient_l186_18695


namespace order_of_roots_l186_18698

theorem order_of_roots (m n p : ℝ) : 
  m = (1/3)^(1/5) → n = (1/4)^(1/3) → p = (1/5)^(1/4) → n < p ∧ p < m := by
  sorry

end order_of_roots_l186_18698


namespace min_time_to_return_l186_18611

/-- Given a circular track and a person's walking pattern, calculate the minimum time to return to the starting point. -/
theorem min_time_to_return (track_length : ℝ) (speed : ℝ) (t1 t2 t3 : ℝ) : 
  track_length = 400 →
  speed = 6000 / 60 →
  t1 = 1 →
  t2 = 3 →
  t3 = 5 →
  (min_time : ℝ) * speed = track_length - ((t1 - t2 + t3) * speed) →
  min_time = 1 := by
  sorry

#check min_time_to_return

end min_time_to_return_l186_18611


namespace imaginary_part_of_complex_division_l186_18679

theorem imaginary_part_of_complex_division (i : ℂ) : i * i = -1 → 
  Complex.im ((4 - 3 * i) / i) = -4 := by
  sorry

end imaginary_part_of_complex_division_l186_18679


namespace c_range_l186_18680

theorem c_range (a b c : ℝ) 
  (ha : 6 < a ∧ a < 10) 
  (hb : a / 2 ≤ b ∧ b ≤ 2 * a) 
  (hc : c = a + b) : 
  9 < c ∧ c < 30 := by
  sorry

end c_range_l186_18680


namespace sphere_radius_is_6_l186_18603

/-- The radius of a sphere whose surface area is equal to the curved surface area of a right circular cylinder with height and diameter both 12 cm. -/
def sphere_radius : ℝ := 6

/-- The height of the cylinder. -/
def cylinder_height : ℝ := 12

/-- The diameter of the cylinder. -/
def cylinder_diameter : ℝ := 12

/-- The theorem stating that the radius of the sphere is 6 cm. -/
theorem sphere_radius_is_6 :
  sphere_radius = 6 ∧
  4 * Real.pi * sphere_radius ^ 2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height :=
sorry

end sphere_radius_is_6_l186_18603


namespace percentage_difference_l186_18674

theorem percentage_difference (x y z : ℝ) : 
  x = 1.2 * y ∧ x = 0.36 * z → y = 0.3 * z :=
by sorry

end percentage_difference_l186_18674


namespace system_solution_l186_18671

theorem system_solution (a : ℝ) (x y z : ℝ) :
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = a^2) →
  (x^3 + y^3 + z^3 = a^3) →
  ((x = a ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = a ∧ z = 0) ∨
   (x = 0 ∧ y = 0 ∧ z = a)) :=
by sorry

end system_solution_l186_18671


namespace existence_of_finite_set_with_1993_unit_distance_neighbors_l186_18609

theorem existence_of_finite_set_with_1993_unit_distance_neighbors :
  ∃ (A : Set (ℝ × ℝ)), Set.Finite A ∧
    ∀ X ∈ A, ∃ (Y : Fin 1993 → ℝ × ℝ),
      (∀ i, Y i ∈ A) ∧
      (∀ i j, i ≠ j → Y i ≠ Y j) ∧
      (∀ i, dist X (Y i) = 1) :=
sorry

end existence_of_finite_set_with_1993_unit_distance_neighbors_l186_18609


namespace intersection_of_M_and_N_l186_18619

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {-3, 3, 4} := by
  sorry

end intersection_of_M_and_N_l186_18619


namespace profit_percentage_at_marked_price_l186_18601

theorem profit_percentage_at_marked_price 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0)
  (h3 : 0.8 * marked_price = 1.2 * cost_price) : 
  (marked_price - cost_price) / cost_price = 0.5 := by
sorry

end profit_percentage_at_marked_price_l186_18601


namespace min_value_of_t_l186_18683

theorem min_value_of_t (x y t : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 3 * x + y + x * y - 13 = 0) 
  (h2 : ∃ (t : ℝ), t ≥ 2 * y + x) : 
  ∀ t, t ≥ 2 * y + x → t ≥ 8 * Real.sqrt 2 - 7 :=
by sorry

end min_value_of_t_l186_18683


namespace cistern_wet_surface_area_l186_18608

/-- The total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 9 6 2.25 = 121.5 := by
  sorry

end cistern_wet_surface_area_l186_18608


namespace parallel_linear_functions_touch_theorem_l186_18639

/-- Two linear functions that are parallel but not parallel to the coordinate axes -/
structure ParallelLinearFunctions where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The condition that (f(x))^2 touches 20g(x) -/
def touches_condition_1 (f : ParallelLinearFunctions) : Prop :=
  ∃! x : ℝ, (f.a * x + f.b)^2 = 20 * (f.a * x + f.c)

/-- The condition that (g(x))^2 touches f(x)/A -/
def touches_condition_2 (f : ParallelLinearFunctions) (A : ℝ) : Prop :=
  ∃! x : ℝ, (f.a * x + f.c)^2 = (f.a * x + f.b) / A

/-- The main theorem -/
theorem parallel_linear_functions_touch_theorem (f : ParallelLinearFunctions) :
  touches_condition_1 f → (touches_condition_2 f A ↔ A = -1/20) :=
sorry

end parallel_linear_functions_touch_theorem_l186_18639


namespace emus_per_pen_l186_18631

/-- Proves that the number of emus in each pen is 6 -/
theorem emus_per_pen (num_pens : ℕ) (eggs_per_week : ℕ) (h1 : num_pens = 4) (h2 : eggs_per_week = 84) : 
  (eggs_per_week / 7 * 2) / num_pens = 6 := by
  sorry

#check emus_per_pen

end emus_per_pen_l186_18631


namespace min_slope_tangent_line_l186_18630

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 3

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x : ℝ, f' x ≥ f' x₀) ∧ 
    y₀ = f x₀ ∧
    (∀ x : ℝ, 3*x - y₀ = 1 → f x = 3*x - 1) :=
sorry

end min_slope_tangent_line_l186_18630


namespace sum_greater_than_two_l186_18620

theorem sum_greater_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 := by
  sorry

end sum_greater_than_two_l186_18620


namespace unique_number_satisfying_conditions_l186_18690

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def has_even_number_of_factors (n : ℕ) : Prop :=
  Even (Nat.card (Nat.divisors n))

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
    ((has_even_number_of_factors n ∨ n > 50) ∧
     ¬(has_even_number_of_factors n ∧ n > 50)) ∧
    ((Odd n ∨ n > 60) ∧ ¬(Odd n ∧ n > 60)) ∧
    ((Even n ∨ n > 70) ∧ ¬(Even n ∧ n > 70)) ∧
    n = 64 :=
by
  sorry

#check unique_number_satisfying_conditions

end unique_number_satisfying_conditions_l186_18690


namespace sqrt_rational_l186_18657

theorem sqrt_rational (a b c : ℚ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : Real.sqrt a + Real.sqrt b = c) : 
  ∃ (q r : ℚ), Real.sqrt a = q ∧ Real.sqrt b = r := by
sorry

end sqrt_rational_l186_18657


namespace age_difference_l186_18673

/-- Given the ages of three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 17 →
  b = 6 →
  a - b = 2 := by
  sorry

end age_difference_l186_18673


namespace smallest_n_for_coprime_subset_l186_18685

def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 100}

theorem smallest_n_for_coprime_subset : 
  ∃ (n : Nat), n = 75 ∧ 
  (∀ (A : Set Nat), A ⊆ S → A.Finite → A.ncard ≥ n → 
    ∃ (a b c d : Nat), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
    Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ 
    Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime c d) ∧
  (∀ (m : Nat), m < 75 → 
    ∃ (B : Set Nat), B ⊆ S ∧ B.Finite ∧ B.ncard = m ∧
    ¬(∃ (a b c d : Nat), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ d ∈ B ∧ 
      Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ 
      Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime c d)) :=
by sorry

end smallest_n_for_coprime_subset_l186_18685


namespace hyperbola_properties_l186_18621

-- Define the given hyperbola
def given_hyperbola (x y : ℝ) : Prop := x^2 - 2*y^2 = 2

-- Define the desired hyperbola
def desired_hyperbola (x y : ℝ) : Prop := y^2/2 - x^2/4 = 1

-- Define a function to represent the asymptotes
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote x y ↔ (∃ k : ℝ, given_hyperbola x y ∧ k ≠ 0 ∧ y = k*x)) ∧
  (∀ x y : ℝ, asymptote x y ↔ (∃ k : ℝ, desired_hyperbola x y ∧ k ≠ 0 ∧ y = k*x)) ∧
  desired_hyperbola 2 (-2) :=
sorry

end hyperbola_properties_l186_18621


namespace car_selling_price_l186_18625

/-- Calculates the selling price of a car given its purchase price, repair cost, and profit percentage. -/
theorem car_selling_price (purchase_price repair_cost : ℕ) (profit_percent : ℚ) :
  purchase_price = 42000 →
  repair_cost = 13000 →
  profit_percent = 17272727272727273 / 100000000000000000 →
  (purchase_price + repair_cost) * (1 + profit_percent) = 64500 := by
  sorry

end car_selling_price_l186_18625


namespace average_age_decrease_l186_18684

theorem average_age_decrease (initial_average : ℝ) : 
  let original_total := 10 * initial_average
  let new_total := original_total - 44 + 14
  let new_average := new_total / 10
  initial_average - new_average = 3 := by
sorry

end average_age_decrease_l186_18684


namespace inequality_solution_set_l186_18656

def solution_set (x : ℝ) : Prop := x ≥ 0 ∨ x ≤ -2

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x + 2) ≥ 0 ↔ solution_set x :=
by sorry

end inequality_solution_set_l186_18656


namespace distance_between_points_l186_18697

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 12)
  let p2 : ℝ × ℝ := (10, 0)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 193 := by
  sorry

end distance_between_points_l186_18697


namespace modulus_of_complex_l186_18687

theorem modulus_of_complex : Complex.abs (7/4 - 3*I) = (Real.sqrt 193)/4 := by sorry

end modulus_of_complex_l186_18687


namespace circle_equation_and_line_intersection_l186_18602

/-- Represents a circle with center on the x-axis -/
structure CircleOnXAxis where
  center : ℤ
  radius : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def is_tangent (circle : CircleOnXAxis) (line : Line) : Prop :=
  (|line.a * circle.center + line.c| / Real.sqrt (line.a^2 + line.b^2)) = circle.radius

def intersects_circle (circle : CircleOnXAxis) (line : Line) : Prop :=
  ∃ x y : ℝ, line.a * x + line.b * y + line.c = 0 ∧
             (x - circle.center)^2 + y^2 = circle.radius^2

theorem circle_equation_and_line_intersection
  (circle : CircleOnXAxis)
  (tangent_line : Line)
  (h_radius : circle.radius = 5)
  (h_tangent : is_tangent circle tangent_line)
  (h_tangent_eq : tangent_line.a = 4 ∧ tangent_line.b = 3 ∧ tangent_line.c = -29) :
  (∃ equation : ℝ → ℝ → Prop, ∀ x y, equation x y ↔ (x - 1)^2 + y^2 = 25) ∧
  (∀ a : ℝ, a > 0 →
    let intersecting_line : Line := { a := a, b := -1, c := 5 }
    intersects_circle circle intersecting_line ↔ a > 5/12) :=
sorry

end circle_equation_and_line_intersection_l186_18602


namespace max_value_theorem_l186_18664

theorem max_value_theorem (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ (max_x : ℝ), max_x = 1/2 ∧
  ∀ y, 0 < y ∧ y < 1 → x * (3 - 3 * x) ≤ max_x * (3 - 3 * max_x) :=
sorry

end max_value_theorem_l186_18664


namespace rectangle_dissection_theorem_l186_18629

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle -/
structure Triangle

/-- Represents a pentagon -/
structure Pentagon

/-- Represents a set of shapes that can be rearranged -/
structure ShapeSet where
  triangles : Finset Triangle
  pentagon : Pentagon

theorem rectangle_dissection_theorem (initial : Rectangle) (final : Rectangle) 
  (h_initial : initial.width = 4 ∧ initial.height = 6)
  (h_final : final.width = 3 ∧ final.height = 8)
  (h_area_preservation : initial.width * initial.height = final.width * final.height) :
  ∃ (pieces : ShapeSet), 
    pieces.triangles.card = 2 ∧ 
    (∃ (arrangement : ShapeSet → Rectangle), arrangement pieces = final) :=
sorry

end rectangle_dissection_theorem_l186_18629


namespace trajectory_is_ellipse_l186_18669

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the moving circle M
def M (x y r : ℝ) : Prop := ∃ (center : ℝ × ℝ), center = (x, y) ∧ r > 0

-- Define internal tangency of M to C₁
def internalTangent (x y r : ℝ) : Prop :=
  M x y r ∧ C₁ (x + r) y

-- Define external tangency of M to C₂
def externalTangent (x y r : ℝ) : Prop :=
  M x y r ∧ C₂ (x - r) y

-- Theorem statement
theorem trajectory_is_ellipse :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), internalTangent x y r ∧ externalTangent x y r) →
    x^2 / 16 + y^2 / 15 = 1 :=
by sorry

end trajectory_is_ellipse_l186_18669


namespace f_min_at_three_l186_18670

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3 -/
theorem f_min_at_three : ∀ x : ℝ, f x ≥ f 3 := by sorry

end f_min_at_three_l186_18670


namespace quadratic_sum_l186_18614

/-- A quadratic function f(x) = ax^2 + bx + c passing through (-2,0) and (4,0) with maximum value 54 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  QuadraticFunction a b c (-2) = 0 →
  QuadraticFunction a b c 4 = 0 →
  (∀ x, QuadraticFunction a b c x ≤ 54) →
  (∃ x, QuadraticFunction a b c x = 54) →
  a + b + c = 54 := by
  sorry

end quadratic_sum_l186_18614


namespace smallest_third_altitude_l186_18618

/-- An isosceles triangle with integer altitudes -/
structure IsoscelesTriangle where
  -- The length of the equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The altitude to the equal sides
  altitude_to_equal_side : ℝ
  -- The altitude to the base
  altitude_to_base : ℝ
  -- Constraint: the triangle is isosceles
  isosceles : side > 0
  -- Constraint: altitudes are positive
  altitude_to_equal_side_pos : altitude_to_equal_side > 0
  altitude_to_base_pos : altitude_to_base > 0
  -- Constraint: altitudes are integers
  altitude_to_equal_side_int : ∃ n : ℤ, altitude_to_equal_side = n
  altitude_to_base_int : ∃ n : ℤ, altitude_to_base = n

/-- The theorem stating the smallest possible value for the third altitude -/
theorem smallest_third_altitude (t : IsoscelesTriangle) 
  (h1 : t.altitude_to_equal_side = 15)
  (h2 : t.altitude_to_base = 5) :
  ∃ h : ℝ, h ≥ 5 ∧ 
  (∀ h' : ℝ, (∃ n : ℤ, h' = n) → 
    (2 * t.side * t.base = t.altitude_to_equal_side * t.base + t.altitude_to_base * t.side) → 
    h' ≥ h) := by
  sorry

end smallest_third_altitude_l186_18618


namespace count_grid_paths_l186_18616

/-- The number of paths from (0,0) to (m,n) in a grid with only right and up steps -/
def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of paths from (0,0) to (m,n) is (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) :
  grid_paths m n = Nat.choose (m + n) m := by sorry

end count_grid_paths_l186_18616


namespace josh_found_seven_marbles_l186_18653

/-- The number of marbles Josh had initially -/
def initial_marbles : ℕ := 21

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := 28

/-- The number of marbles Josh found -/
def found_marbles : ℕ := current_marbles - initial_marbles

theorem josh_found_seven_marbles :
  found_marbles = 7 :=
by sorry

end josh_found_seven_marbles_l186_18653


namespace sqrt_inequality_l186_18632

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end sqrt_inequality_l186_18632


namespace train_crossing_time_l186_18689

theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 431.25)
  (h3 : platform_crossing_time = 39) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / platform_crossing_time
  train_length / train_speed = 16 := by sorry

end train_crossing_time_l186_18689


namespace xyz_inequality_l186_18659

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  Real.sqrt (1 + 8 * x) + Real.sqrt (1 + 8 * y) + Real.sqrt (1 + 8 * z) ≥ 9 := by
  sorry

end xyz_inequality_l186_18659


namespace circular_track_length_l186_18622

/-- The length of a circular track given cycling conditions. -/
theorem circular_track_length
  (ivanov_initial_speed : ℝ)
  (petrov_speed : ℝ)
  (track_length : ℝ)
  (h1 : 2 * ivanov_initial_speed - 2 * petrov_speed = 3 * track_length)
  (h2 : 3 * ivanov_initial_speed + 10 - 3 * petrov_speed = 7 * track_length) :
  track_length = 4 := by
sorry

end circular_track_length_l186_18622


namespace base_seven_5432_equals_1934_l186_18696

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base_seven_5432_equals_1934 : 
  base_seven_to_ten [2, 3, 4, 5] = 1934 := by
  sorry

end base_seven_5432_equals_1934_l186_18696


namespace quadratic_equation_real_root_l186_18637

theorem quadratic_equation_real_root (m : ℝ) : 
  ∃ x : ℝ, x^2 - (m + 1) * x + (3 * m - 6) = 0 := by
  sorry

end quadratic_equation_real_root_l186_18637


namespace number_of_nests_l186_18691

theorem number_of_nests (birds : ℕ) (nests : ℕ) : 
  birds = 6 → birds = nests + 3 → nests = 3 := by
  sorry

end number_of_nests_l186_18691


namespace smallest_number_of_blocks_l186_18640

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions --/
def calculateBlocksNeeded (wall : WallDimensions) (blockHeight : ℕ) (evenRowBlocks : ℕ) (oddRowBlocks : ℕ) : ℕ :=
  let oddRows := (wall.height + 1) / 2
  let evenRows := wall.height / 2
  oddRows * oddRowBlocks + evenRows * evenRowBlocks

/-- Theorem stating the smallest number of blocks needed for the wall --/
theorem smallest_number_of_blocks 
  (wall : WallDimensions)
  (blockHeight : ℕ)
  (block2ft : BlockDimensions)
  (block1_5ft : BlockDimensions)
  (block1ft : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 7)
  (h3 : blockHeight = 1)
  (h4 : block2ft.length = 2)
  (h5 : block1_5ft.length = 3/2)
  (h6 : block1ft.length = 1)
  (h7 : block2ft.height = blockHeight)
  (h8 : block1_5ft.height = blockHeight)
  (h9 : block1ft.height = blockHeight) :
  calculateBlocksNeeded wall blockHeight 61 60 = 423 :=
sorry

end smallest_number_of_blocks_l186_18640


namespace theater_revenue_l186_18675

theorem theater_revenue (total_seats : ℕ) (adult_price child_price : ℕ) (child_tickets : ℕ) :
  total_seats = 80 →
  adult_price = 12 →
  child_price = 5 →
  child_tickets = 63 →
  (total_seats = child_tickets + (total_seats - child_tickets)) →
  child_tickets * child_price + (total_seats - child_tickets) * adult_price = 519 := by
  sorry

end theater_revenue_l186_18675


namespace stadium_empty_seats_l186_18662

/-- The number of empty seats in a stadium -/
def empty_seats (total_seats people_present : ℕ) : ℕ :=
  total_seats - people_present

/-- Theorem: In a stadium with 92 seats and 47 people present, there are 45 empty seats -/
theorem stadium_empty_seats :
  empty_seats 92 47 = 45 := by
  sorry

end stadium_empty_seats_l186_18662


namespace f_below_tangent_and_inequality_l186_18692

-- Define the function f(x) = (2-x)e^x
noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Define the tangent line l(x) = x + 2
def l (x : ℝ) : ℝ := x + 2

theorem f_below_tangent_and_inequality (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ l x) ∧
  (f (1 / n - 1 / (n + 1)) + (1 / Real.exp 2) * f (2 - 1 / n) ≤ 2 + 1 / n) := by
  sorry

end f_below_tangent_and_inequality_l186_18692


namespace candy_bar_cost_l186_18648

/-- The cost of a candy bar given initial and remaining amounts --/
theorem candy_bar_cost (initial : ℕ) (remaining : ℕ) (cost : ℕ) :
  initial = 4 →
  remaining = 3 →
  initial = remaining + cost →
  cost = 1 := by
  sorry

end candy_bar_cost_l186_18648


namespace factor_expression_l186_18633

theorem factor_expression (x : ℝ) : 45 * x + 30 = 15 * (3 * x + 2) := by
  sorry

end factor_expression_l186_18633


namespace number_between_24_and_28_l186_18647

def possibleNumbers : List ℕ := [20, 23, 26, 29]

theorem number_between_24_and_28 (n : ℕ) 
  (h1 : n > 24) 
  (h2 : n < 28) 
  (h3 : n ∈ possibleNumbers) : 
  n = 26 := by
  sorry

end number_between_24_and_28_l186_18647


namespace unique_solution_for_equation_l186_18617

theorem unique_solution_for_equation :
  ∀ x y : ℕ+,
    x > y →
    (x - y : ℕ+) ^ (x * y : ℕ) = x ^ (y : ℕ) * y ^ (x : ℕ) →
    x = 4 ∧ y = 2 := by
  sorry

end unique_solution_for_equation_l186_18617


namespace degree_of_g_l186_18660

/-- Given a polynomial f(x) = -7x^4 + 3x^3 + x - 5 and another polynomial g(x) such that 
    the degree of f(x) + g(x) is 2, prove that the degree of g(x) is 4. -/
theorem degree_of_g (f g : Polynomial ℝ) : 
  f = -7 * X^4 + 3 * X^3 + X - 5 →
  (f + g).degree = 2 →
  g.degree = 4 := by
  sorry

end degree_of_g_l186_18660


namespace solve_systems_of_equations_l186_18634

theorem solve_systems_of_equations :
  -- System 1
  (∃ (x y : ℝ), x - y = 3 ∧ 3*x - 8*y = 14 ∧ x = 2 ∧ y = -1) ∧
  -- System 2
  (∃ (x y : ℝ), 3*x + y = 1 ∧ 5*x - 2*y = 9 ∧ x = 1 ∧ y = -2) :=
by sorry

end solve_systems_of_equations_l186_18634


namespace quadratic_form_k_value_l186_18678

theorem quadratic_form_k_value :
  ∃ (a h : ℝ), ∀ x : ℝ, 9 * x^2 - 12 * x = a * (x - h)^2 - 4 :=
by sorry

end quadratic_form_k_value_l186_18678


namespace pyramid_numbers_l186_18652

theorem pyramid_numbers (a b : ℕ) : 
  (42 = a * 6) → 
  (72 = 6 * b) → 
  (504 = 42 * 72) → 
  (a = 7 ∧ b = 12) := by
sorry

end pyramid_numbers_l186_18652


namespace weekly_reading_time_l186_18607

-- Define the daily meditation time
def daily_meditation_time : ℝ := 1

-- Define the daily reading time as twice the meditation time
def daily_reading_time : ℝ := 2 * daily_meditation_time

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem weekly_reading_time : daily_reading_time * days_in_week = 14 := by
  sorry

end weekly_reading_time_l186_18607


namespace total_worth_is_correct_l186_18668

-- Define the given conditions
def initial_packs : ℕ := 4
def new_packs : ℕ := 2
def price_per_pack : ℚ := 2.5
def discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.07

-- Define the function to calculate the total worth
def total_worth : ℚ :=
  let initial_cost := initial_packs * price_per_pack
  let new_cost := new_packs * price_per_pack
  let discount := new_cost * discount_rate
  let discounted_cost := new_cost - discount
  let tax := new_cost * tax_rate
  let total_new_cost := discounted_cost + tax
  initial_cost + total_new_cost

-- State the theorem
theorem total_worth_is_correct : total_worth = 14.6 := by
  sorry

end total_worth_is_correct_l186_18668


namespace solution_equation_one_solution_equation_two_l186_18681

-- Equation 1
theorem solution_equation_one : 
  ∀ x : ℝ, 2 * x^2 - 4 * x - 1 = 0 ↔ x = 1 + Real.sqrt 6 / 2 ∨ x = 1 - Real.sqrt 6 / 2 := by
sorry

-- Equation 2
theorem solution_equation_two :
  ∀ x : ℝ, (x - 1) * (x + 2) = 28 ↔ x = -6 ∨ x = 5 := by
sorry

end solution_equation_one_solution_equation_two_l186_18681


namespace long_video_multiple_is_42_l186_18665

/-- Represents the video release schedule and durations for John's channel --/
structure VideoSchedule where
  short_videos_per_day : Nat
  long_videos_per_day : Nat
  short_video_duration : Nat
  days_per_week : Nat
  total_weekly_duration : Nat

/-- Calculates how many times longer the long video is compared to a short video --/
def long_video_multiple (schedule : VideoSchedule) : Nat :=
  let total_short_duration := schedule.short_videos_per_day * schedule.short_video_duration * schedule.days_per_week
  let long_video_duration := schedule.total_weekly_duration - total_short_duration
  long_video_duration / (schedule.long_videos_per_day * schedule.days_per_week * schedule.short_video_duration)

theorem long_video_multiple_is_42 (schedule : VideoSchedule) 
  (h1 : schedule.short_videos_per_day = 2)
  (h2 : schedule.long_videos_per_day = 1)
  (h3 : schedule.short_video_duration = 2)
  (h4 : schedule.days_per_week = 7)
  (h5 : schedule.total_weekly_duration = 112) :
  long_video_multiple schedule = 42 := by
  sorry

#eval long_video_multiple {
  short_videos_per_day := 2,
  long_videos_per_day := 1,
  short_video_duration := 2,
  days_per_week := 7,
  total_weekly_duration := 112
}

end long_video_multiple_is_42_l186_18665


namespace range_of_t_l186_18605

/-- Given a set A containing 1 and a real number t, 
    the range of t is all real numbers except 1 -/
theorem range_of_t (t : ℝ) (A : Set ℝ) (h : A = {1, t}) : 
  {x : ℝ | x ≠ 1} = {x : ℝ | ∃ y ∈ A, y = x ∧ y ≠ 1} := by
sorry

end range_of_t_l186_18605


namespace weight_of_person_a_l186_18666

/-- Given the average weights of different groups and the relationship between individuals' weights,
    prove that the weight of person A is 80 kg. -/
theorem weight_of_person_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 8 →
  (b + c + d + e) / 4 = 79 →
  a = 80 := by
sorry

end weight_of_person_a_l186_18666


namespace carnival_game_cost_per_play_l186_18646

/-- Represents the carnival game scenario -/
structure CarnivalGame where
  budget : ℚ
  red_points : ℕ
  green_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ
  total_points_possible : ℕ

/-- Calculates the cost per play for the carnival game -/
def cost_per_play (game : CarnivalGame) : ℚ :=
  game.budget / game.games_played

/-- Theorem stating that the cost per play is $1.50 for the given scenario -/
theorem carnival_game_cost_per_play :
  let game : CarnivalGame := {
    budget := 3,
    red_points := 2,
    green_points := 3,
    games_played := 2,
    red_buckets_hit := 4,
    green_buckets_hit := 5,
    total_points_possible := 38
  }
  cost_per_play game = 3/2 := by
  sorry

end carnival_game_cost_per_play_l186_18646


namespace mike_remaining_nickels_l186_18651

/-- Given Mike's initial number of nickels and the number of nickels his dad borrowed,
    calculate the number of nickels Mike has left. -/
def nickels_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Mike has 12 nickels remaining after his dad's borrowing. -/
theorem mike_remaining_nickels :
  nickels_remaining 87 75 = 12 := by
  sorry

end mike_remaining_nickels_l186_18651


namespace root_existence_l186_18686

theorem root_existence (h1 : Real.log 1.5 < 4/11) (h2 : Real.log 2 > 2/7) :
  ∃ x : ℝ, 1/4 < x ∧ x < 1/2 ∧ Real.log (2*x + 1) = 1 / (3*x + 2) := by
  sorry

end root_existence_l186_18686


namespace probability_JQKA_same_suit_value_l186_18672

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards dealt -/
def CardsDealt : ℕ := 4

/-- Represents the number of Jacks in a standard deck -/
def JacksInDeck : ℕ := 4

/-- Probability of drawing a specific sequence of four cards (Jack, Queen, King, Ace) 
    of the same suit from a standard 52-card deck without replacement -/
def probability_JQKA_same_suit : ℚ :=
  (JacksInDeck : ℚ) / StandardDeck *
  1 / (StandardDeck - 1) *
  1 / (StandardDeck - 2) *
  1 / (StandardDeck - 3)

theorem probability_JQKA_same_suit_value : 
  probability_JQKA_same_suit = 1 / 1624350 := by sorry

end probability_JQKA_same_suit_value_l186_18672


namespace equation_proof_l186_18642

theorem equation_proof (x y : ℝ) (h : x - 2*y = -2) : 3 + 2*x - 4*y = -1 := by
  sorry

end equation_proof_l186_18642


namespace rectangular_field_perimeter_l186_18645

theorem rectangular_field_perimeter (a b d A : ℝ) : 
  a = 2 * b →                 -- One side is twice as long as the other
  a * b = A →                 -- Area is A
  a^2 + b^2 = d^2 →           -- Pythagorean theorem for diagonal
  A = 240 →                   -- Area is 240 square meters
  d = 34 →                    -- Diagonal is 34 meters
  2 * (a + b) = 91.2 :=       -- Perimeter is 91.2 meters
by sorry

end rectangular_field_perimeter_l186_18645


namespace arithmetic_mean_scaling_l186_18636

theorem arithmetic_mean_scaling (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  let original_set := [b₁, b₂, b₃, b₃, b₅]
  let scaled_set := original_set.map (· * 3)
  let original_mean := (b₁ + b₂ + b₃ + b₄ + b₅) / 5
  let scaled_mean := (scaled_set.sum) / 5
  scaled_mean = 3 * original_mean := by
sorry


end arithmetic_mean_scaling_l186_18636


namespace div_power_eq_power_l186_18627

theorem div_power_eq_power (a : ℝ) : a^4 / (-a)^2 = a^2 := by
  sorry

end div_power_eq_power_l186_18627


namespace equation_solution_l186_18606

theorem equation_solution : ∃ (x : ℝ), 
  Real.sqrt (9 + Real.sqrt (25 + 5*x)) + Real.sqrt (3 + Real.sqrt (5 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 0.2 := by
  sorry

end equation_solution_l186_18606


namespace al_wins_probability_l186_18650

/-- Represents the possible moves in Rock Paper Scissors -/
inductive Move
| Rock
| Paper
| Scissors

/-- The probability of Bob playing each move -/
def bobProbability : Move → ℚ
| Move.Rock => 1/3
| Move.Paper => 1/3
| Move.Scissors => 1/3

/-- Al's move is Rock -/
def alMove : Move := Move.Rock

/-- Determines if Al wins given Bob's move -/
def alWins (bobMove : Move) : Bool :=
  match bobMove with
  | Move.Scissors => true
  | _ => false

/-- The probability of Al winning -/
def probAlWins : ℚ := bobProbability Move.Scissors

theorem al_wins_probability :
  probAlWins = 1/3 :=
by sorry

end al_wins_probability_l186_18650


namespace initial_number_problem_l186_18693

theorem initial_number_problem (x : ℝ) : 8 * x - 4 = 2.625 → x = 0.828125 := by
  sorry

end initial_number_problem_l186_18693


namespace cost_of_type_b_books_l186_18641

/-- Given a total of 100 books, with 'a' books of type A purchased,
    and type B books costing $6 each, prove that the cost of type B books
    is 6(100 - a) dollars. -/
theorem cost_of_type_b_books (a : ℕ) : ℕ :=
  let total_books : ℕ := 100
  let price_b : ℕ := 6
  let num_b : ℕ := total_books - a
  price_b * num_b

#check cost_of_type_b_books

end cost_of_type_b_books_l186_18641


namespace min_value_sum_squares_l186_18600

theorem min_value_sum_squares (x y z : ℝ) (h : x + y + z = 1) :
  ∃ m : ℝ, m = 4/9 ∧ ∀ a b c : ℝ, a + b + c = 1 → a^2 + b^2 + 4*c^2 ≥ m :=
by sorry

end min_value_sum_squares_l186_18600


namespace particular_solution_correct_l186_18624

/-- The differential equation xy' = y - 1 -/
def diff_eq (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (deriv y x) = y x - 1

/-- The general solution y = Cx + 1 -/
def general_solution (C : ℝ) (x : ℝ) : ℝ :=
  C * x + 1

/-- The particular solution y = 4x + 1 -/
def particular_solution (x : ℝ) : ℝ :=
  4 * x + 1

theorem particular_solution_correct :
  ∀ C : ℝ,
  (∀ x : ℝ, diff_eq x (general_solution C)) →
  general_solution C 1 = 5 →
  ∀ x : ℝ, general_solution C x = particular_solution x :=
by sorry

end particular_solution_correct_l186_18624


namespace sin_330_degrees_l186_18654

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by sorry

end sin_330_degrees_l186_18654


namespace chicken_feathers_l186_18613

theorem chicken_feathers (initial_feathers : ℕ) (cars_dodged : ℕ) : 
  initial_feathers = 5263 →
  cars_dodged = 23 →
  initial_feathers - (cars_dodged * 2) = 5217 := by
  sorry

end chicken_feathers_l186_18613


namespace complex_in_second_quadrant_l186_18635

theorem complex_in_second_quadrant (θ : Real) (h : θ ∈ Set.Ioo (3*Real.pi/4) (5*Real.pi/4)) :
  let z : ℂ := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 := by
sorry

end complex_in_second_quadrant_l186_18635


namespace min_value_expression_l186_18676

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ 2 :=
by sorry

end min_value_expression_l186_18676


namespace canada_population_l186_18626

/-- The number of moose in Canada -/
def moose : ℕ := 1000000

/-- The number of beavers in Canada -/
def beavers : ℕ := 2 * moose

/-- The number of humans in Canada -/
def humans : ℕ := 19 * beavers

/-- Theorem: Given the relationship between moose, beavers, and humans in Canada,
    and a moose population of 1 million, the human population is 38 million. -/
theorem canada_population : humans = 38000000 := by
  sorry

end canada_population_l186_18626


namespace triangle_could_be_isosceles_l186_18649

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  t.c^2 - t.a^2 + t.b^2 = (4*t.a*t.c - 2*t.b*t.c) * Real.cos t.A

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem statement
theorem triangle_could_be_isosceles (t : Triangle) 
  (h : satisfiesCondition t) : 
  ∃ (t' : Triangle), satisfiesCondition t' ∧ isIsosceles t' :=
sorry

end triangle_could_be_isosceles_l186_18649


namespace room_width_calculation_l186_18699

/-- Given a room with the specified dimensions and paving costs, prove that the width is 3.75 meters -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) 
    (h1 : length = 5.5)
    (h2 : total_cost = 24750)
    (h3 : rate_per_sqm = 1200) : 
  total_cost / rate_per_sqm / length = 3.75 := by
  sorry

end room_width_calculation_l186_18699


namespace max_garden_area_l186_18667

theorem max_garden_area (L : ℝ) (h : L > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = L ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + 2*b = L → x*y ≥ a*b ∧
  x*y = L^2/8 :=
sorry

end max_garden_area_l186_18667


namespace garden_vegetables_l186_18610

theorem garden_vegetables (potatoes cucumbers peppers : ℕ) : 
  cucumbers = potatoes - 60 →
  peppers = 2 * cucumbers →
  potatoes + cucumbers + peppers = 768 →
  potatoes = 237 := by
sorry

end garden_vegetables_l186_18610
