import Mathlib

namespace eleven_percent_of_700_is_77_l1229_122908

theorem eleven_percent_of_700_is_77 : (11 / 100) * 700 = 77 := by
  sorry

end eleven_percent_of_700_is_77_l1229_122908


namespace volume_common_tetrahedra_l1229_122974

/-- Given a parallelepiped ABCDA₁B₁C₁D₁ with volume V, the volume of the common part
    of tetrahedra AB₁CD₁ and A₁BC₁D is V/12 -/
theorem volume_common_tetrahedra (V : ℝ) : ℝ :=
  let parallelepiped_volume := V
  let common_volume := V / 12
  common_volume

#check volume_common_tetrahedra

end volume_common_tetrahedra_l1229_122974


namespace youngest_member_age_l1229_122924

theorem youngest_member_age (n : ℕ) (current_avg : ℚ) (birth_avg : ℚ) 
  (h1 : n = 5)
  (h2 : current_avg = 20)
  (h3 : birth_avg = 25/2) :
  (n : ℚ) * current_avg - (n - 1 : ℚ) * birth_avg = 10 := by
  sorry

end youngest_member_age_l1229_122924


namespace internet_speed_calculation_l1229_122976

/-- Represents the internet speed calculation problem -/
theorem internet_speed_calculation 
  (file1 : ℝ) 
  (file2 : ℝ) 
  (file3 : ℝ) 
  (download_time : ℝ) 
  (h1 : file1 = 80) 
  (h2 : file2 = 90) 
  (h3 : file3 = 70) 
  (h4 : download_time = 2) :
  (file1 + file2 + file3) / download_time = 120 := by
  sorry

#check internet_speed_calculation

end internet_speed_calculation_l1229_122976


namespace equal_positive_reals_l1229_122916

theorem equal_positive_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (x*y + 1) / (x + 1) = (y*z + 1) / (y + 1))
  (h2 : (y*z + 1) / (y + 1) = (z*x + 1) / (z + 1)) :
  x = y ∧ y = z := by sorry

end equal_positive_reals_l1229_122916


namespace smallest_multiplier_for_perfect_square_l1229_122996

theorem smallest_multiplier_for_perfect_square : ∃ (k : ℕ+), 
  (∀ (m : ℕ+), (∃ (n : ℕ), 2010 * m = n * n) → k ≤ m) ∧ 
  (∃ (n : ℕ), 2010 * k = n * n) ∧
  k = 2010 := by
  sorry

end smallest_multiplier_for_perfect_square_l1229_122996


namespace five_integers_sum_20_product_420_l1229_122962

theorem five_integers_sum_20_product_420 :
  ∃! (a b c d e : ℕ+),
    a + b + c + d + e = 20 ∧
    a * b * c * d * e = 420 :=
by sorry

end five_integers_sum_20_product_420_l1229_122962


namespace imaginary_number_condition_l1229_122979

theorem imaginary_number_condition (a : ℝ) : 
  let z : ℂ := (a - 2*I) / (2 + I)
  (∃ b : ℝ, z = b * I ∧ b ≠ 0) → a = 1 := by
  sorry

end imaginary_number_condition_l1229_122979


namespace haley_jason_difference_l1229_122970

/-- The number of necklaces Haley has -/
def haley_necklaces : ℕ := 25

/-- The difference between Haley's and Josh's necklaces -/
def haley_josh_diff : ℕ := 15

/-- Represents the relationship between Josh's and Jason's necklaces -/
def josh_jason_ratio : ℚ := 1/2

/-- The number of necklaces Josh has -/
def josh_necklaces : ℕ := haley_necklaces - haley_josh_diff

/-- The number of necklaces Jason has -/
def jason_necklaces : ℕ := (2 * josh_necklaces)

theorem haley_jason_difference : haley_necklaces - jason_necklaces = 5 := by
  sorry

end haley_jason_difference_l1229_122970


namespace cubic_fraction_equals_25_l1229_122984

theorem cubic_fraction_equals_25 (a b : ℝ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3)^2 / (a^2 - a*b + b^2)^2 = 25 := by
  sorry

end cubic_fraction_equals_25_l1229_122984


namespace quadratic_equations_solutions_l1229_122918

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ x₁^2 + 4*x₁ - 5 = 0 ∧ x₂^2 + 4*x₂ - 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -1 ∧ 3*y₁^2 + 2*y₁ = 1 ∧ 3*y₂^2 + 2*y₂ = 1) :=
by
  sorry

end quadratic_equations_solutions_l1229_122918


namespace min_value_expression_l1229_122925

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^2 + 4*x*y + 9*y^2 + 8*y*z + 3*z^2 ≥ 9^(10/9) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 8*y₀*z₀ + 3*z₀^2 = 9^(10/9) :=
by sorry

end min_value_expression_l1229_122925


namespace number_of_subsets_complement_union_l1229_122914

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 3}

-- Define set B
def B : Finset Nat := {1, 2, 4}

-- Theorem statement
theorem number_of_subsets_complement_union (U A B : Finset Nat) : 
  Finset.card (Finset.powerset ((U \ B) ∪ A)) = 8 :=
sorry

end number_of_subsets_complement_union_l1229_122914


namespace unique_solution_implies_a_greater_than_one_l1229_122981

/-- If the equation 2ax^2 - x - 1 = 0 has exactly one solution in the interval (0,1), then a > 1 -/
theorem unique_solution_implies_a_greater_than_one (a : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Ioo 0 1 ∧ 2*a*x^2 - x - 1 = 0) → a > 1 := by
  sorry

end unique_solution_implies_a_greater_than_one_l1229_122981


namespace power_functions_inequality_l1229_122913

theorem power_functions_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) :
  (((x₁ + x₂) / 2) ^ 2 < (x₁^2 + x₂^2) / 2) ∧
  (2 / (x₁ + x₂) < (1 / x₁ + 1 / x₂) / 2) := by
  sorry

end power_functions_inequality_l1229_122913


namespace lines_parallel_if_planes_parallel_and_coplanar_l1229_122978

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (coplanar : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_if_planes_parallel_and_coplanar
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_in_α : subset m α)
  (h_n_in_β : subset n β)
  (h_planes_parallel : parallel α β)
  (h_coplanar : coplanar m n) :
  line_parallel m n :=
sorry

end lines_parallel_if_planes_parallel_and_coplanar_l1229_122978


namespace sum_of_number_and_its_square_l1229_122982

theorem sum_of_number_and_its_square (n : ℕ) : n = 11 → n + n^2 = 132 := by
  sorry

end sum_of_number_and_its_square_l1229_122982


namespace bread_slices_left_l1229_122901

/-- The number of slices of bread Tony uses per sandwich -/
def slices_per_sandwich : ℕ := 2

/-- The number of sandwiches Tony made from Monday to Friday -/
def weekday_sandwiches : ℕ := 5

/-- The number of sandwiches Tony made on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The total number of slices in the loaf Tony started with -/
def initial_slices : ℕ := 22

/-- Theorem stating the number of bread slices left after Tony made sandwiches for the week -/
theorem bread_slices_left : 
  initial_slices - (slices_per_sandwich * (weekday_sandwiches + saturday_sandwiches)) = 8 := by
  sorry

end bread_slices_left_l1229_122901


namespace circle_symmetry_l1229_122947

/-- The equation of a circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Symmetry of a circle with respect to the origin -/
def symmetric_to_origin (c1 c2 : Circle) : Prop :=
  c1.center.1 = -c2.center.1 ∧ c1.center.2 = -c2.center.2 ∧ c1.radius = c2.radius

theorem circle_symmetry (c : Circle) :
  symmetric_to_origin c ⟨(-2, 1), 1⟩ →
  c = ⟨(2, -1), 1⟩ := by
  sorry

end circle_symmetry_l1229_122947


namespace quadratic_minimum_value_l1229_122951

theorem quadratic_minimum_value (c : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, -x^2 - 2*x + c ≥ -5) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, -x^2 - 2*x + c = -5) → 
  c = 3 := by
  sorry

end quadratic_minimum_value_l1229_122951


namespace gcd_digits_bound_l1229_122953

theorem gcd_digits_bound (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 :=
by sorry

end gcd_digits_bound_l1229_122953


namespace last_two_digits_of_floor_fraction_l1229_122950

theorem last_two_digits_of_floor_fraction : ∃ n : ℕ, 
  n ≥ 10^62 - 3 * 10^31 + 8 ∧ 
  n < 10^62 - 3 * 10^31 + 9 ∧ 
  n % 100 = 8 := by
  sorry

end last_two_digits_of_floor_fraction_l1229_122950


namespace solution_set_implies_a_and_b_solution_set_when_a_negative_l1229_122972

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (a - 2) * x - 2

-- Part 1
theorem solution_set_implies_a_and_b :
  ∀ a b : ℝ, (∀ x : ℝ, f a x ≤ b ↔ -2 ≤ x ∧ x ≤ 1) → a = 1 ∧ b = 0 := by sorry

-- Part 2
theorem solution_set_when_a_negative :
  ∀ a : ℝ, a < 0 →
    (∀ x : ℝ, f a x ≥ 0 ↔
      ((-2 < a ∧ a < 0 ∧ 1 ≤ x ∧ x ≤ -2/a) ∨
       (a = -2 ∧ x = 1) ∨
       (a < -2 ∧ -2/a ≤ x ∧ x ≤ 1))) := by sorry

end solution_set_implies_a_and_b_solution_set_when_a_negative_l1229_122972


namespace product_equality_l1229_122912

theorem product_equality : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end product_equality_l1229_122912


namespace harvard_attendance_l1229_122969

theorem harvard_attendance 
  (total_applicants : ℕ) 
  (acceptance_rate : ℚ) 
  (attendance_rate : ℚ) :
  total_applicants = 20000 →
  acceptance_rate = 5 / 100 →
  attendance_rate = 90 / 100 →
  (total_applicants : ℚ) * acceptance_rate * attendance_rate = 900 :=
by sorry

end harvard_attendance_l1229_122969


namespace inequality_proof_l1229_122915

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_condition : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
sorry

end inequality_proof_l1229_122915


namespace car_travel_time_l1229_122939

/-- Proves that a car with given specifications travels for 5 hours -/
theorem car_travel_time (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (fuel_used_ratio : ℝ) :
  speed = 50 →
  fuel_efficiency = 30 →
  tank_capacity = 10 →
  fuel_used_ratio = 0.8333333333333334 →
  (fuel_used_ratio * tank_capacity * fuel_efficiency) / speed = 5 := by
  sorry

end car_travel_time_l1229_122939


namespace rice_profit_l1229_122986

/-- Calculates the profit from selling a sack of rice -/
def calculate_profit (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) : ℝ :=
  weight * price_per_kg - cost

/-- Theorem: The profit from selling a 50kg sack of rice that costs $50 at $1.20 per kg is $10 -/
theorem rice_profit : calculate_profit 50 50 1.20 = 10 := by
  sorry

end rice_profit_l1229_122986


namespace jakes_weight_l1229_122942

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 8 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 278) :
  jake_weight = 188 := by
sorry

end jakes_weight_l1229_122942


namespace geometric_sequence_fifth_term_l1229_122994

/-- A geometric sequence with specified terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_3 : a 3 = -4) 
  (h_7 : a 7 = -16) : 
  a 5 = -8 := by
sorry

end geometric_sequence_fifth_term_l1229_122994


namespace unique_prime_evaluation_l1229_122983

theorem unique_prime_evaluation (T : ℕ) (h : T = 2161) :
  ∃! p : ℕ, Prime p ∧ ∃ n : ℤ, n^4 - 898*n^2 + T - 2160 = p :=
by
  -- The proof goes here
  sorry

end unique_prime_evaluation_l1229_122983


namespace license_plate_combinations_l1229_122988

/-- The number of possible characters for each position in the license plate -/
def numCharOptions : ℕ := 26 + 10

/-- The length of the license plate -/
def plateLength : ℕ := 4

/-- The number of ways to position two identical characters in non-adjacent positions in a 4-character plate -/
def numPairPositions : ℕ := 3

/-- The number of ways to choose characters for the non-duplicate positions -/
def numNonDuplicateChoices : ℕ := numCharOptions * (numCharOptions - 1)

theorem license_plate_combinations :
  numPairPositions * numCharOptions * numNonDuplicateChoices = 136080 := by
  sorry

end license_plate_combinations_l1229_122988


namespace crayon_cost_proof_l1229_122959

/-- The cost of one pack of crayons -/
def pack_cost : ℚ := 5/2

/-- The number of packs Michael initially has -/
def initial_packs : ℕ := 4

/-- The number of packs Michael buys -/
def bought_packs : ℕ := 2

/-- The total value of all packs after purchase -/
def total_value : ℚ := 15

theorem crayon_cost_proof :
  (initial_packs + bought_packs : ℚ) * pack_cost = total_value := by
  sorry

end crayon_cost_proof_l1229_122959


namespace length_of_AB_is_seven_l1229_122992

-- Define the triangle ABC
structure TriangleABC where
  A : Point
  B : Point
  C : Point

-- Define the triangle CBD
structure TriangleCBD where
  C : Point
  B : Point
  D : Point

-- Define the properties of the triangles
def isIsosceles (t : TriangleABC) : Prop := sorry
def isEquilateral (t : TriangleABC) : Prop := sorry
def isIsoscelesCBD (t : TriangleCBD) : Prop := sorry
def perimeterCBD (t : TriangleCBD) : ℝ := sorry
def perimeterABC (t : TriangleABC) : ℝ := sorry
def lengthBD (t : TriangleCBD) : ℝ := sorry
def lengthAB (t : TriangleABC) : ℝ := sorry

theorem length_of_AB_is_seven 
  (abc : TriangleABC) 
  (cbd : TriangleCBD) 
  (h1 : isIsosceles abc)
  (h2 : isEquilateral abc)
  (h3 : isIsoscelesCBD cbd)
  (h4 : perimeterCBD cbd = 24)
  (h5 : perimeterABC abc = 21)
  (h6 : lengthBD cbd = 10) :
  lengthAB abc = 7 := by sorry

end length_of_AB_is_seven_l1229_122992


namespace xy_sum_theorem_l1229_122971

theorem xy_sum_theorem (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_lt_20 : x < 20) (hy_lt_20 : y < 20) 
  (h_eq : x + y + x * y = 99) : x + y = 23 ∨ x + y = 18 :=
sorry

end xy_sum_theorem_l1229_122971


namespace payment_difference_l1229_122987

-- Define the pizza parameters
def total_slices : ℕ := 10
def plain_pizza_cost : ℚ := 12
def double_cheese_cost : ℚ := 4

-- Define the number of slices each person ate
def bob_double_cheese_slices : ℕ := 5
def bob_plain_slices : ℕ := 2
def cindy_plain_slices : ℕ := 3

-- Calculate the total cost of the pizza
def total_pizza_cost : ℚ := plain_pizza_cost + double_cheese_cost

-- Calculate the cost per slice
def cost_per_slice : ℚ := total_pizza_cost / total_slices

-- Calculate Bob's payment
def bob_payment : ℚ := cost_per_slice * (bob_double_cheese_slices + bob_plain_slices)

-- Calculate Cindy's payment
def cindy_payment : ℚ := cost_per_slice * cindy_plain_slices

-- State the theorem
theorem payment_difference : bob_payment - cindy_payment = 6.4 := by
  sorry

end payment_difference_l1229_122987


namespace marks_weekly_reading_time_l1229_122936

/-- Given Mark's daily reading time and weekly increase, prove his total weekly reading time -/
theorem marks_weekly_reading_time 
  (daily_reading_time : ℕ) 
  (weekly_increase : ℕ) 
  (h1 : daily_reading_time = 2)
  (h2 : weekly_increase = 4) :
  daily_reading_time * 7 + weekly_increase = 18 := by
  sorry

#check marks_weekly_reading_time

end marks_weekly_reading_time_l1229_122936


namespace salary_change_l1229_122906

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * (1 + 0.2)
  let final_salary := increased_salary * (1 - 0.2)
  (final_salary - initial_salary) / initial_salary = -0.04 :=
by
  sorry

end salary_change_l1229_122906


namespace chord_length_unit_circle_l1229_122985

/-- The length of the chord cut by a line on a unit circle -/
theorem chord_length_unit_circle (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (1 - d^2) = 8/5 ↔ 
    a = 3 ∧ b = -4 ∧ c = 3 :=
by sorry

end chord_length_unit_circle_l1229_122985


namespace cube_edge_length_is_5_l1229_122973

/-- The edge length of a cube immersed in water --/
def cube_edge_length (base_length base_width water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that the edge length of the cube is 5 cm --/
theorem cube_edge_length_is_5 :
  cube_edge_length 10 5 2.5 = 5 := by sorry

end cube_edge_length_is_5_l1229_122973


namespace rabbit_walk_prob_l1229_122954

/-- A random walk on a rectangular grid. -/
structure RandomWalk where
  width : ℕ
  height : ℕ
  start_x : ℕ
  start_y : ℕ

/-- The probability of ending on the top or bottom edge for a given random walk. -/
noncomputable def prob_top_bottom (walk : RandomWalk) : ℚ :=
  sorry

/-- The specific random walk described in the problem. -/
def rabbit_walk : RandomWalk :=
  { width := 6
    height := 5
    start_x := 2
    start_y := 3 }

/-- The main theorem stating the probability for the specific random walk. -/
theorem rabbit_walk_prob : prob_top_bottom rabbit_walk = 17 / 24 := by
  sorry

end rabbit_walk_prob_l1229_122954


namespace circumscribed_circle_area_l1229_122949

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π square units. -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let R := s / Real.sqrt 3
  π * R^2 = 48 * π := by sorry

end circumscribed_circle_area_l1229_122949


namespace circle_area_increase_l1229_122963

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end circle_area_increase_l1229_122963


namespace vector_relation_l1229_122919

/-- Given points A, B, C, and D in a plane, where BC = 3CD, prove that AD = -1/3 AB + 4/3 AC -/
theorem vector_relation (A B C D : ℝ × ℝ) 
  (h : B - C = 3 * (C - D)) : 
  A - D = -1/3 * (A - B) + 4/3 * (A - C) := by
  sorry

end vector_relation_l1229_122919


namespace sum_of_squares_unique_l1229_122977

theorem sum_of_squares_unique (x y z : ℕ+) : 
  (x : ℕ) + y + z = 24 → 
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 → 
  x^2 + y^2 + z^2 = 216 := by
  sorry

end sum_of_squares_unique_l1229_122977


namespace complex_equation_solution_l1229_122903

theorem complex_equation_solution (z : ℂ) : 
  (3 + Complex.I) * z = 2 - Complex.I → 
  z = (1 / 2 : ℂ) - (1 / 2 : ℂ) * Complex.I ∧ Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_equation_solution_l1229_122903


namespace units_digit_base8_product_l1229_122946

/-- The units digit of a number in base 8 -/
def unitsDigitBase8 (n : ℕ) : ℕ := n % 8

/-- The product of 348 and 76 -/
def product : ℕ := 348 * 76

theorem units_digit_base8_product : unitsDigitBase8 product = 0 := by
  sorry

end units_digit_base8_product_l1229_122946


namespace compare_negative_fractions_l1229_122943

theorem compare_negative_fractions : -10/11 > -11/12 := by
  sorry

end compare_negative_fractions_l1229_122943


namespace binomial_coefficient_formula_l1229_122941

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) :
  Nat.choose n k = n.factorial / ((n - k).factorial * k.factorial) :=
by sorry

end binomial_coefficient_formula_l1229_122941


namespace h_definition_l1229_122993

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 1
def g (x : ℝ) : ℝ := 2 * x + 3

-- Define h as a function that satisfies f(h(x)) = g(x)
noncomputable def h (x : ℝ) : ℝ := sorry

-- State the theorem
theorem h_definition (x : ℝ) : f (h x) = g x → h x = (2 * x + 4) / 3 := by
  sorry

end h_definition_l1229_122993


namespace triangle_rectangle_ratio_l1229_122921

theorem triangle_rectangle_ratio : 
  ∀ (t w : ℝ), 
  t > 0 → w > 0 →
  3 * t = 24 →
  2 * (2 * w) + 2 * w = 24 →
  t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l1229_122921


namespace potato_sack_problem_l1229_122900

theorem potato_sack_problem (original_potatoes : ℕ) : 
  original_potatoes - 69 - (2 * 69) - ((2 * 69) / 3) = 47 → 
  original_potatoes = 300 := by
sorry

end potato_sack_problem_l1229_122900


namespace symmetric_point_sum_l1229_122923

/-- A point is symmetric to the line x+y+1=0 if its symmetric point is also on this line -/
def is_symmetric_point (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y + 1 = 0 ∧ (a + x) / 2 + (b + y) / 2 + 1 = 0

/-- Theorem: If a point (a,b) is symmetric to the line x+y+1=0 and its symmetric point
    is also on this line, then a+b=-1 -/
theorem symmetric_point_sum (a b : ℝ) (h : is_symmetric_point a b) : a + b = -1 := by
  sorry

end symmetric_point_sum_l1229_122923


namespace gcf_lcm_sum_8_12_l1229_122932

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcf_lcm_sum_8_12_l1229_122932


namespace equality_preservation_l1229_122948

theorem equality_preservation (x y : ℝ) : x = y → x - 2 = y - 2 := by
  sorry

end equality_preservation_l1229_122948


namespace polynomial_factor_implies_c_value_l1229_122920

/-- If 2x+5 is a factor of 4x^3 + 19x^2 + cx + 45, then c = 40.5 -/
theorem polynomial_factor_implies_c_value :
  ∀ c : ℝ,
  (∀ x : ℝ, (2 * x + 5) ∣ (4 * x^3 + 19 * x^2 + c * x + 45)) →
  c = 40.5 := by
sorry

end polynomial_factor_implies_c_value_l1229_122920


namespace net_percentage_gain_calculation_l1229_122937

/-- Calculate the net percentage gain from buying and selling glass bowls and ceramic plates --/
theorem net_percentage_gain_calculation 
  (glass_bowls_bought : ℕ) 
  (glass_bowls_price : ℚ) 
  (ceramic_plates_bought : ℕ) 
  (ceramic_plates_price : ℚ) 
  (discount_rate : ℚ) 
  (glass_bowls_sold : ℕ) 
  (glass_bowls_sell_price : ℚ) 
  (ceramic_plates_sold : ℕ) 
  (ceramic_plates_sell_price : ℚ) 
  (glass_bowls_broken : ℕ) 
  (ceramic_plates_broken : ℕ) :
  glass_bowls_bought = 250 →
  glass_bowls_price = 18 →
  ceramic_plates_bought = 150 →
  ceramic_plates_price = 25 →
  discount_rate = 5 / 100 →
  glass_bowls_sold = 200 →
  glass_bowls_sell_price = 25 →
  ceramic_plates_sold = 120 →
  ceramic_plates_sell_price = 32 →
  glass_bowls_broken = 30 →
  ceramic_plates_broken = 10 →
  ∃ (net_percentage_gain : ℚ), 
    abs (net_percentage_gain - (271 / 10000 : ℚ)) < (1 / 1000 : ℚ) := by
  sorry

end net_percentage_gain_calculation_l1229_122937


namespace inverse_exponential_is_logarithm_l1229_122910

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_exponential_is_logarithm (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1) : 
  ∀ x, f a x = Real.log x / Real.log 2 := by
sorry

end inverse_exponential_is_logarithm_l1229_122910


namespace equality_or_power_relation_l1229_122990

theorem equality_or_power_relation (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : x^y = y^x) :
  x = y ∨ ∃ m : ℝ, m > 0 ∧ m ≠ 1 ∧ x = m^(1/(m-1)) ∧ y = m^(m/(m-1)) := by
  sorry

end equality_or_power_relation_l1229_122990


namespace function_inequality_implies_k_range_l1229_122991

theorem function_inequality_implies_k_range (k : ℝ) :
  (∀ x : ℝ, (k * x + 1 > 0) ∨ (x^2 - 1 > 0)) →
  k ∈ Set.Ioo (-1 : ℝ) 1 := by
  sorry

end function_inequality_implies_k_range_l1229_122991


namespace simple_interest_principal_calculation_l1229_122980

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ)
  (h_rate : rate = 0.08)
  (h_time : time = 1)
  (h_interest : interest = 800)
  (h_formula : interest = principal * rate * time) :
  principal = 10000 := by
  sorry

end simple_interest_principal_calculation_l1229_122980


namespace sally_tuesday_shirts_l1229_122955

/-- The number of shirts Sally sewed on Monday -/
def monday_shirts : ℕ := 4

/-- The number of shirts Sally sewed on Wednesday -/
def wednesday_shirts : ℕ := 2

/-- The number of buttons required for each shirt -/
def buttons_per_shirt : ℕ := 5

/-- The total number of buttons needed for all shirts -/
def total_buttons : ℕ := 45

/-- The number of shirts Sally sewed on Tuesday -/
def tuesday_shirts : ℕ := 3

theorem sally_tuesday_shirts :
  tuesday_shirts = (total_buttons - (monday_shirts + wednesday_shirts) * buttons_per_shirt) / buttons_per_shirt :=
by sorry

end sally_tuesday_shirts_l1229_122955


namespace recycling_points_l1229_122958

/-- Calculates the points earned from recycling paper --/
def points_earned (pounds_per_point : ℕ) (chloe_pounds : ℕ) (friends_pounds : ℕ) : ℕ :=
  (chloe_pounds + friends_pounds) / pounds_per_point

/-- Theorem: Given the recycling conditions, the total points earned is 5 --/
theorem recycling_points : points_earned 6 28 2 = 5 := by
  sorry

end recycling_points_l1229_122958


namespace quadratic_function_inequality_l1229_122931

theorem quadratic_function_inequality (a b c : ℝ) (h1 : b > 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (a + b + c) / b ≥ 2 := by
  sorry

end quadratic_function_inequality_l1229_122931


namespace solve_equations_l1229_122934

theorem solve_equations :
  (∀ x : ℝ, 4 * x^2 = 9 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, (1 - 2*x)^3 = 8 ↔ x = -1/2) := by
  sorry

end solve_equations_l1229_122934


namespace walk_distance_proof_l1229_122930

/-- Given a constant walking speed and time, calculates the distance walked. -/
def distance_walked (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that walking at 4 miles per hour for 2 hours results in a distance of 8 miles. -/
theorem walk_distance_proof :
  let speed : ℝ := 4
  let time : ℝ := 2
  distance_walked speed time = 8 := by
sorry

end walk_distance_proof_l1229_122930


namespace fruit_basket_problem_l1229_122917

/-- The number of ways to select a non-empty subset of fruits from a given number of identical apples and oranges, such that at least 2 oranges are selected. -/
def fruitBasketCombinations (apples oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges - 1)

/-- Theorem stating that the number of fruit basket combinations with 4 apples and 12 oranges is 55. -/
theorem fruit_basket_problem :
  fruitBasketCombinations 4 12 = 55 := by
  sorry

#eval fruitBasketCombinations 4 12

end fruit_basket_problem_l1229_122917


namespace integer_sum_problem_l1229_122964

theorem integer_sum_problem (x y : ℤ) : 
  (x = 15 ∨ y = 15) → (4 * x + 3 * y = 150) → (x = 30 ∨ y = 30) := by
sorry

end integer_sum_problem_l1229_122964


namespace smallest_n_square_fourth_power_l1229_122935

/-- The smallest positive integer n such that 5n is a perfect square and 3n is a perfect fourth power is 75. -/
theorem smallest_n_square_fourth_power : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 5 * n = k^2) ∧ 
    (∃ (m : ℕ), 3 * n = m^4)) ∧
  (∀ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 5 * n = k^2) ∧ 
    (∃ (m : ℕ), 3 * n = m^4) → 
    n ≥ 75) := by
  sorry

end smallest_n_square_fourth_power_l1229_122935


namespace complex_parts_of_z_l1229_122965

theorem complex_parts_of_z : ∃ (z : ℂ), z = 2 - 3 * I ∧ z.re = 2 ∧ z.im = -3 := by
  sorry

end complex_parts_of_z_l1229_122965


namespace locus_of_midpoints_is_single_point_l1229_122998

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ
  h : r > 0

/-- A point P inside the circle on its diameter -/
structure InteriorPointOnDiameter (K : Circle) where
  P : ℝ × ℝ
  h₁ : dist P K.O < K.r
  h₂ : ∃ (t : ℝ), P = (K.O.1 + t * K.r, K.O.2) ∨ P = (K.O.1, K.O.2 + t * K.r)

/-- The midpoint of a chord passing through P -/
def midpoint_of_chord (K : Circle) (P : InteriorPointOnDiameter K) (θ : ℝ) : ℝ × ℝ :=
  P.P

/-- The theorem stating that the locus of midpoints is a single point -/
theorem locus_of_midpoints_is_single_point (K : Circle) (P : InteriorPointOnDiameter K) :
  ∀ θ : ℝ, midpoint_of_chord K P θ = P.P :=
sorry

end locus_of_midpoints_is_single_point_l1229_122998


namespace stock_sale_cash_realization_l1229_122952

/-- The cash realized on selling a stock, given the brokerage rate and total amount including brokerage -/
theorem stock_sale_cash_realization (brokerage_rate : ℚ) (total_with_brokerage : ℚ) :
  brokerage_rate = 1 / 400 →
  total_with_brokerage = 106 →
  ∃ cash_realized : ℚ, cash_realized + cash_realized * brokerage_rate = total_with_brokerage ∧
                    cash_realized = 42400 / 401 := by
  sorry

end stock_sale_cash_realization_l1229_122952


namespace paths_in_7x6_grid_l1229_122928

/-- The number of paths in a grid with specified horizontal and vertical steps --/
def numPaths (horizontal vertical : ℕ) : ℕ :=
  Nat.choose (horizontal + vertical) vertical

/-- Theorem stating that the number of paths in a 7x6 grid is 1716 --/
theorem paths_in_7x6_grid :
  numPaths 7 6 = 1716 := by
  sorry

end paths_in_7x6_grid_l1229_122928


namespace octal_subtraction_l1229_122902

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Theorem: 53₈ - 27₈ = 24₈ in base 8 --/
theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 53 - octal_to_decimal 27) = 24 := by sorry

end octal_subtraction_l1229_122902


namespace pencil_arrangement_theorem_l1229_122995

def total_pencils (total_rows : ℕ) (pattern_length : ℕ) (pencils_second_row : ℕ) : ℕ :=
  let pattern_repeats := total_rows / pattern_length
  let pencils_fifth_row := pencils_second_row + pencils_second_row / 2
  let pencil_rows_per_pattern := 2
  pattern_repeats * pencil_rows_per_pattern * (pencils_second_row + pencils_fifth_row)

theorem pencil_arrangement_theorem :
  total_pencils 30 6 76 = 950 := by
  sorry

end pencil_arrangement_theorem_l1229_122995


namespace max_min_sum_on_interval_l1229_122926

def f (x : ℝ) := 2 * x^2 - 6 * x + 1

theorem max_min_sum_on_interval :
  ∃ (m M : ℝ),
    (∀ x ∈ Set.Icc (-1) 1, m ≤ f x ∧ f x ≤ M) ∧
    (∃ x₁ ∈ Set.Icc (-1) 1, f x₁ = m) ∧
    (∃ x₂ ∈ Set.Icc (-1) 1, f x₂ = M) ∧
    M + m = 6 :=
sorry

end max_min_sum_on_interval_l1229_122926


namespace remainder_theorem_l1229_122938

theorem remainder_theorem (x y u v : ℤ) (hx : 0 < x) (hy : 0 < y) (h_div : x = u * y + v) (h_rem : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end remainder_theorem_l1229_122938


namespace prob_different_plants_l1229_122999

/-- The number of distinct plant options available -/
def num_options : ℕ := 4

/-- The probability of two employees choosing different plants -/
def prob_different_choices : ℚ := 3/4

/-- Theorem stating that the probability of two employees choosing different plants
    from four options is 3/4 -/
theorem prob_different_plants :
  (num_options : ℚ)^2 - num_options = (prob_different_choices * num_options^2 : ℚ) := by
  sorry

end prob_different_plants_l1229_122999


namespace digging_project_breadth_l1229_122968

/-- The breadth of the first digging project -/
def breadth_project1 : ℝ := 30

/-- The depth of the first digging project in meters -/
def depth_project1 : ℝ := 100

/-- The length of the first digging project in meters -/
def length_project1 : ℝ := 25

/-- The depth of the second digging project in meters -/
def depth_project2 : ℝ := 75

/-- The length of the second digging project in meters -/
def length_project2 : ℝ := 20

/-- The breadth of the second digging project in meters -/
def breadth_project2 : ℝ := 50

/-- The number of days to complete each project -/
def days_to_complete : ℝ := 12

theorem digging_project_breadth :
  depth_project1 * length_project1 * breadth_project1 =
  depth_project2 * length_project2 * breadth_project2 :=
by sorry

end digging_project_breadth_l1229_122968


namespace possible_a_values_l1229_122922

theorem possible_a_values :
  ∀ (a : ℤ), 
    (∃ (b c : ℤ), ∀ (x : ℤ), (x - a) * (x - 15) + 4 = (x + b) * (x + c)) ↔ 
    (a = 16 ∨ a = 21) := by
sorry

end possible_a_values_l1229_122922


namespace sum_xy_is_zero_l1229_122907

theorem sum_xy_is_zero (x y : ℝ) 
  (h : (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1) : 
  x + y = 0 := by
  sorry

end sum_xy_is_zero_l1229_122907


namespace units_digit_of_factorial_sum_l1229_122909

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

def units_digit (n : ℕ) : ℕ := sorry

theorem units_digit_of_factorial_sum :
  units_digit (sum_factorials 15) = 3 := by sorry

end units_digit_of_factorial_sum_l1229_122909


namespace ceiling_floor_product_range_l1229_122944

theorem ceiling_floor_product_range (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 210 → -15 < y ∧ y < -14 := by sorry

end ceiling_floor_product_range_l1229_122944


namespace function_derivative_existence_l1229_122929

open Set

theorem function_derivative_existence (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a) (h2 : a < b)
  (h3 : ContinuousOn f (Icc a b))
  (h4 : DifferentiableOn ℝ f (Ioo a b)) :
  ∃ c ∈ Ioo a b, deriv f c = 1 / (a - c) + 1 / (b - c) + 1 / (a + b) := by
  sorry

end function_derivative_existence_l1229_122929


namespace units_digit_of_product_first_four_composites_l1229_122933

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_first_four_composites :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end units_digit_of_product_first_four_composites_l1229_122933


namespace hash_12_6_hash_general_form_l1229_122911

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ := 
  r * s + 2 * r

-- Theorem to prove
theorem hash_12_6 : hash 12 6 = 96 := by
  sorry

-- Axioms for the # operation
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

-- Additional theorem to prove the general form
theorem hash_general_form (r s : ℝ) : hash r s = r * s + 2 * r := by
  sorry

end hash_12_6_hash_general_form_l1229_122911


namespace two_numbers_with_given_means_l1229_122956

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (Real.sqrt (a * b) = Real.sqrt 5) → 
  (2 / (1/a + 1/b) = 5/3) → 
  ((a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1)) := by
sorry

end two_numbers_with_given_means_l1229_122956


namespace a_divisibility_a_specific_cases_l1229_122957

def a (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem a_divisibility (n : ℕ) (h : n > 0) :
  (3^n ∣ a n) ∧ ¬(3^(n+1) ∣ a n) :=
sorry

theorem a_specific_cases :
  (3 ∣ a 1) ∧ ¬(9 ∣ a 1) ∧
  (9 ∣ a 2) ∧ ¬(27 ∣ a 2) ∧
  (27 ∣ a 3) ∧ ¬(81 ∣ a 3) :=
sorry

end a_divisibility_a_specific_cases_l1229_122957


namespace smallest_digit_sum_of_difference_l1229_122927

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- A predicate that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10

theorem smallest_digit_sum_of_difference :
  ∀ a b : ℕ,
    100 ≤ a ∧ a < 1000 →
    100 ≤ b ∧ b < 1000 →
    a > b →
    allDigitsDifferent (1000000 * a + b) →
    100 ≤ a - b ∧ a - b < 1000 →
    (∀ D : ℕ, 100 ≤ D ∧ D < 1000 → D = a - b → sumOfDigits D ≥ 9) ∧
    (∃ D : ℕ, 100 ≤ D ∧ D < 1000 ∧ D = a - b ∧ sumOfDigits D = 9) :=
by sorry

end smallest_digit_sum_of_difference_l1229_122927


namespace geometric_sequence_ratio_l1229_122945

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∃ d : ℝ, 3 * a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) →  -- Arithmetic sequence condition
  (∃ q : ℝ, ∀ n, a (n+1) = q * a n) →  -- Geometric sequence definition
  (a 8 + a 9) / (a 6 + a 7) = 9 := by
sorry

end geometric_sequence_ratio_l1229_122945


namespace mean_of_remaining_numbers_l1229_122997

theorem mean_of_remaining_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 →
  (a + b + c + d) / 4 = 88.75 := by
  sorry

end mean_of_remaining_numbers_l1229_122997


namespace problem_solution_l1229_122966

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - a - Real.log x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * g a x

theorem problem_solution :
  (∃ a : ℝ, ∀ x > 0, g a x ≥ 0) ∧
  (∃ a : ℝ, ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧
    (∀ x > 0, (deriv (f a)) x₀ = 0 ∧ f a x ≤ f a x₀)) :=
by
  sorry

end problem_solution_l1229_122966


namespace commodity_profit_optimization_l1229_122960

/-- Represents the monthly sales quantity as a function of price -/
def sales_quantity (x : ℝ) : ℝ := -30 * x + 960

/-- Represents the monthly profit as a function of price -/
def monthly_profit (x : ℝ) : ℝ := (sales_quantity x) * (x - 10)

theorem commodity_profit_optimization (cost_price : ℝ) 
  (h1 : cost_price = 10)
  (h2 : sales_quantity 20 = 360)
  (h3 : sales_quantity 30 = 60) :
  (∃ (optimal_price max_profit : ℝ),
    (∀ x, monthly_profit x ≤ monthly_profit optimal_price) ∧
    monthly_profit optimal_price = max_profit ∧
    optimal_price = 21 ∧
    max_profit = 3630) := by
  sorry

end commodity_profit_optimization_l1229_122960


namespace car_efficiency_before_modification_l1229_122967

/-- Represents the fuel efficiency of a car before and after modification -/
structure CarEfficiency where
  pre_mod : ℝ  -- Fuel efficiency before modification (miles per gallon)
  post_mod : ℝ  -- Fuel efficiency after modification (miles per gallon)
  fuel_capacity : ℝ  -- Fuel tank capacity in gallons
  extra_distance : ℝ  -- Additional distance traveled after modification (miles)

/-- Theorem stating the car's fuel efficiency before modification -/
theorem car_efficiency_before_modification (car : CarEfficiency)
  (h1 : car.post_mod = car.pre_mod / 0.8)
  (h2 : car.fuel_capacity = 15)
  (h3 : car.fuel_capacity * car.post_mod = car.fuel_capacity * car.pre_mod + car.extra_distance)
  (h4 : car.extra_distance = 105) :
  car.pre_mod = 28 := by
  sorry

end car_efficiency_before_modification_l1229_122967


namespace last_two_digits_zero_l1229_122961

theorem last_two_digits_zero (x y : ℕ) : 
  (x^2 + x*y + y^2) % 10 = 0 → (x^2 + x*y + y^2) % 100 = 0 := by
  sorry

end last_two_digits_zero_l1229_122961


namespace subtraction_preserves_inequality_l1229_122905

theorem subtraction_preserves_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end subtraction_preserves_inequality_l1229_122905


namespace geometric_sequence_minimum_value_l1229_122989

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2 ∧
    ∀ k l : ℕ, k > 0 → l > 0 → a k * a l = 16 * (a 1)^2 →
      1 / m + 4 / n ≤ 1 / k + 4 / l :=
by sorry

end geometric_sequence_minimum_value_l1229_122989


namespace power_multiplication_l1229_122904

theorem power_multiplication (m : ℝ) : m^3 * m^2 = m^5 := by
  sorry

end power_multiplication_l1229_122904


namespace complex_multiplication_l1229_122975

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : i * (2 * i + 1) = -2 + i := by
  sorry

end complex_multiplication_l1229_122975


namespace fraction_inequality_l1229_122940

theorem fraction_inequality (a b : ℝ) :
  ((b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0)) → (1 / a < 1 / b) ∧
  (a > 0 ∧ 0 > b) → ¬(1 / a < 1 / b) :=
by sorry

end fraction_inequality_l1229_122940
