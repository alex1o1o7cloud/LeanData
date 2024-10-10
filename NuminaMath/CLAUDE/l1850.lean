import Mathlib

namespace triangle_area_triangle_area_is_eight_l1850_185015

/-- Given two lines with slopes 1/4 and 5/4 intersecting at (1,1), and a vertical line x=5,
    the area of the triangle formed by these three lines is 8. -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∀ (line1 line2 : ℝ → ℝ) (x : ℝ),
      (∀ x, line1 x = 1/4 * x + 3/4) →  -- Equation of line with slope 1/4 passing through (1,1)
      (∀ x, line2 x = 5/4 * x - 1/4) →  -- Equation of line with slope 5/4 passing through (1,1)
      line1 1 = 1 →                     -- Both lines pass through (1,1)
      line2 1 = 1 →
      x = 5 →                           -- The vertical line is x=5
      area = 8                          -- The area of the formed triangle is 8

-- The proof of this theorem
theorem triangle_area_is_eight : triangle_area 8 := by
  sorry

end triangle_area_triangle_area_is_eight_l1850_185015


namespace square_root_sum_implies_product_l1850_185025

theorem square_root_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (40 - x) = 10 →
  (10 + x) * (40 - x) = 625 := by
  sorry

end square_root_sum_implies_product_l1850_185025


namespace circle_properties_l1850_185005

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the bisecting line
def bisecting_line (x y : ℝ) : Prop := x - y = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (∀ x y : ℝ, circle_equation x y → bisecting_line x y) ∧
    (∃ x y : ℝ, circle_equation x y ∧ tangent_line x y) :=
sorry

end circle_properties_l1850_185005


namespace daisy_sales_proof_l1850_185069

/-- The number of daisies sold on the first day -/
def first_day_sales : ℕ := 45

/-- The number of daisies sold on the second day -/
def second_day_sales : ℕ := first_day_sales + 20

/-- The number of daisies sold on the third day -/
def third_day_sales : ℕ := 2 * second_day_sales - 10

/-- The number of daisies sold on the fourth day -/
def fourth_day_sales : ℕ := 120

/-- The total number of daisies sold over 4 days -/
def total_sales : ℕ := 350

theorem daisy_sales_proof :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = total_sales :=
by sorry

end daisy_sales_proof_l1850_185069


namespace bisection_method_for_f_l1850_185012

def f (x : ℝ) := 3 * x^2 - 1

theorem bisection_method_for_f :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo 0 1 ∧ f x₀ = 0 ∧ ∀ x ∈ Set.Ioo 0 1, f x = 0 → x = x₀ →
  let ε : ℝ := 0.05
  let n : ℕ := 5
  let approx : ℝ := 37/64
  (∀ m : ℕ, m < n → 1 / 2^m > ε) ∧
  1 / 2^n ≤ ε ∧
  |approx - x₀| < ε :=
sorry

end bisection_method_for_f_l1850_185012


namespace average_score_is_92_l1850_185042

def brief_scores : List Int := [10, -5, 0, 8, -3]
def xiao_ming_score : Int := 90
def xiao_ming_rank : Nat := 3

def actual_scores : List Int := brief_scores.map (λ x => xiao_ming_score + x)

theorem average_score_is_92 : 
  (actual_scores.sum : ℚ) / actual_scores.length = 92 := by sorry

end average_score_is_92_l1850_185042


namespace min_distance_and_line_equation_l1850_185057

-- Define the line l: x - y + 3 = 0
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C: (x - 1)^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the distance PA
def distance_PA (x y : ℝ) : ℝ := sorry

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := 2*x - 2*y - 1 = 0

-- Theorem statement
theorem min_distance_and_line_equation :
  (∃ (x y : ℝ), point_P x y ∧ 
    (∀ (x' y' : ℝ), point_P x' y' → distance_PA x y ≤ distance_PA x' y')) ∧
  (∀ (x y : ℝ), point_P x y ∧ distance_PA x y = Real.sqrt 7 → line_AB x y) :=
sorry

end min_distance_and_line_equation_l1850_185057


namespace cubic_polynomial_satisfies_conditions_l1850_185070

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => (17/3) * x^3 - 38 * x^2 - (101/3) * x + 185/3
  (q 1 = -5) ∧ (q 2 = 1) ∧ (q 3 = -1) ∧ (q 4 = 23) := by
  sorry

end cubic_polynomial_satisfies_conditions_l1850_185070


namespace third_person_contribution_l1850_185055

theorem third_person_contribution
  (total : ℕ)
  (h_total : total = 1040)
  (x : ℕ)
  (h_brittany : 3 * x = Brittany)
  (h_angela : 3 * Brittany = Angela)
  (h_sum : x + Brittany + Angela = total) :
  x = 80 := by
sorry

end third_person_contribution_l1850_185055


namespace quadratic_root_two_l1850_185064

theorem quadratic_root_two (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 2) ↔ 4 * a + 2 * b + c = 0 := by
  sorry

end quadratic_root_two_l1850_185064


namespace art_marks_calculation_l1850_185090

theorem art_marks_calculation (geography : ℕ) (history_government : ℕ) (computer_science : ℕ) (modern_literature : ℕ) (average : ℚ) :
  geography = 56 →
  history_government = 60 →
  computer_science = 85 →
  modern_literature = 80 →
  average = 70.6 →
  ∃ (art : ℕ), (geography + history_government + art + computer_science + modern_literature : ℚ) / 5 = average ∧ art = 72 :=
by
  sorry

#check art_marks_calculation

end art_marks_calculation_l1850_185090


namespace simultaneous_integers_l1850_185034

theorem simultaneous_integers (t : ℤ) : 
  let x : ℤ := 60 * t + 1
  (∃ (k₁ k₂ k₃ : ℤ), (2 * x + 1) / 3 = k₁ ∧ (3 * x + 1) / 4 = k₂ ∧ (4 * x + 1) / 5 = k₃) ∧
  (∀ (y : ℤ), y ≠ x → ¬(∃ (k₁ k₂ k₃ : ℤ), (2 * y + 1) / 3 = k₁ ∧ (3 * y + 1) / 4 = k₂ ∧ (4 * y + 1) / 5 = k₃)) :=
by sorry

end simultaneous_integers_l1850_185034


namespace sine_equality_theorem_l1850_185098

theorem sine_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.sin (192 * π / 180)) ↔ (n = 12 ∨ n = 168) := by
sorry

end sine_equality_theorem_l1850_185098


namespace elephant_drinking_problem_l1850_185028

/-- The number of days it takes for one elephant to drink a lake dry -/
def days_to_drink_lake (V C K : ℝ) : ℝ :=
  365

/-- Theorem stating the conditions and the result for the elephant drinking problem -/
theorem elephant_drinking_problem (V C K : ℝ) 
  (h1 : 183 * C = V + K)
  (h2 : 37 * 5 * C = V + 5 * K)
  (h3 : V > 0)
  (h4 : C > 0)
  (h5 : K > 0) :
  ∃ (t : ℝ), t * C = V + t * K ∧ t = days_to_drink_lake V C K :=
by
  sorry

#check elephant_drinking_problem

end elephant_drinking_problem_l1850_185028


namespace preimage_of_point_l1850_185079

def f (x y : ℝ) : ℝ × ℝ := (2*x + y, x*y)

theorem preimage_of_point (x₁ y₁ x₂ y₂ : ℝ) :
  f x₁ y₁ = (1/6, -1/6) ∧ f x₂ y₂ = (1/6, -1/6) ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  ((x₁ = 1/4 ∧ y₁ = -1/3) ∨ (x₁ = -1/3 ∧ y₁ = 7/6)) ∧
  ((x₂ = 1/4 ∧ y₂ = -1/3) ∨ (x₂ = -1/3 ∧ y₂ = 7/6)) :=
sorry

end preimage_of_point_l1850_185079


namespace napkin_division_l1850_185020

structure Napkin :=
  (is_square : Bool)
  (folds : Nat)
  (cut_type : String)

def can_divide (n : Napkin) (parts : Nat) : Prop :=
  n.is_square ∧ n.folds = 2 ∧ n.cut_type = "straight" ∧ 
  ((parts = 2 ∨ parts = 3 ∨ parts = 4) ∨ parts ≠ 5)

theorem napkin_division (n : Napkin) (parts : Nat) :
  can_divide n parts ↔ (parts = 2 ∨ parts = 3 ∨ parts = 4) :=
sorry

end napkin_division_l1850_185020


namespace julia_money_left_l1850_185077

theorem julia_money_left (initial_amount : ℚ) : initial_amount = 40 →
  let after_game := initial_amount / 2
  let after_purchases := after_game - (after_game / 4)
  after_purchases = 15 := by
sorry

end julia_money_left_l1850_185077


namespace sin_theta_value_l1850_185087

theorem sin_theta_value (θ : Real) 
  (h1 : 6 * Real.tan θ = 5 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-3 + 2 * Real.sqrt 34) / 5 := by
  sorry

end sin_theta_value_l1850_185087


namespace angle_C_measure_side_ratio_bounds_l1850_185068

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π

variable (t : Triangle)

/-- First theorem: If sin(2C - π/2) = 1/2 and a² + b² < c², then C = 2π/3 -/
theorem angle_C_measure (h1 : sin (2 * t.C - π/2) = 1/2) (h2 : t.a^2 + t.b^2 < t.c^2) :
  t.C = 2*π/3 := by sorry

/-- Second theorem: If C = 2π/3, then 1 < (a + b)/c ≤ 2√3/3 -/
theorem side_ratio_bounds (h : t.C = 2*π/3) :
  1 < (t.a + t.b) / t.c ∧ (t.a + t.b) / t.c ≤ 2 * Real.sqrt 3 / 3 := by sorry

end angle_C_measure_side_ratio_bounds_l1850_185068


namespace tan_alpha_eq_one_third_l1850_185062

theorem tan_alpha_eq_one_third (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end tan_alpha_eq_one_third_l1850_185062


namespace bottle_production_l1850_185010

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 5 such machines will produce 900 bottles in 4 minutes. -/
theorem bottle_production 
  (machines : ℕ) 
  (bottles_per_minute : ℕ) 
  (h1 : machines = 6) 
  (h2 : bottles_per_minute = 270) : 
  (5 : ℕ) * (4 : ℕ) * (bottles_per_minute / machines) = 900 := by
  sorry


end bottle_production_l1850_185010


namespace lcm_gcf_relation_l1850_185082

theorem lcm_gcf_relation (n : ℕ) :
  n ≠ 0 ∧ Nat.lcm n 24 = 48 ∧ Nat.gcd n 24 = 8 → n = 16 := by
  sorry

end lcm_gcf_relation_l1850_185082


namespace remainder_after_adding_2010_l1850_185074

theorem remainder_after_adding_2010 (n : ℤ) (h : n % 6 = 1) : (n + 2010) % 6 = 1 := by
  sorry

end remainder_after_adding_2010_l1850_185074


namespace arithmetic_expression_equals_24_l1850_185053

theorem arithmetic_expression_equals_24 : (8 * 10 - 8) / 3 = 24 := by
  sorry

end arithmetic_expression_equals_24_l1850_185053


namespace least_clock_equivalent_l1850_185036

def clock_equivalent (n : ℕ) : Prop :=
  24 ∣ (n^2 - n)

theorem least_clock_equivalent : 
  ∀ k : ℕ, k > 5 → clock_equivalent k → k ≥ 9 :=
by sorry

end least_clock_equivalent_l1850_185036


namespace factors_of_243_times_5_l1850_185067

-- Define the number we're working with
def n : Nat := 243 * 5

-- Define a function to count the number of distinct positive factors
def countDistinctPositiveFactors (x : Nat) : Nat :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card

-- State the theorem
theorem factors_of_243_times_5 : countDistinctPositiveFactors n = 12 := by
  sorry

end factors_of_243_times_5_l1850_185067


namespace binomial_coefficient_inequality_l1850_185095

theorem binomial_coefficient_inequality (n k h : ℕ) (h1 : n ≥ k + h) :
  Nat.choose n (k + h) ≥ Nat.choose (n - k) h :=
sorry

end binomial_coefficient_inequality_l1850_185095


namespace floor_ratio_property_l1850_185035

theorem floor_ratio_property (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) 
  (h : ∀ n : ℕ, Int.floor (x / y) = Int.floor (n * x) / Int.floor (n * y)) :
  x = y ∨ (∃ (a b : ℕ), x = a ∧ y = b ∧ (a ∣ b ∨ b ∣ a)) :=
sorry

end floor_ratio_property_l1850_185035


namespace train_length_proof_l1850_185075

def train_problem (length1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) (clear_time : ℝ) : Prop :=
  let relative_speed : ℝ := (speed1 + speed2) * (1000 / 3600)
  let total_length : ℝ := relative_speed * clear_time
  let length2 : ℝ := total_length - length1
  length2 = 180

theorem train_length_proof :
  train_problem 110 80 65 7.199424046076314 :=
by sorry

end train_length_proof_l1850_185075


namespace grid_50_25_toothpicks_l1850_185052

/-- Calculates the number of toothpicks needed for a grid --/
def toothpicks_in_grid (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A grid of 50 by 25 toothpicks requires 2575 toothpicks --/
theorem grid_50_25_toothpicks :
  toothpicks_in_grid 50 25 = 2575 := by
  sorry

end grid_50_25_toothpicks_l1850_185052


namespace permutations_of_middle_digits_l1850_185043

/-- The number of permutations of four digits with two pairs of repeated digits -/
def permutations_with_repetition : ℕ := 6

/-- The set of digits to be permuted -/
def digits : Finset ℕ := {2, 2, 3, 3}

/-- The theorem stating that the number of permutations of the given digits is 6 -/
theorem permutations_of_middle_digits :
  Finset.card (Finset.powersetCard 4 digits) = permutations_with_repetition :=
sorry

end permutations_of_middle_digits_l1850_185043


namespace symmetry_point_yOz_l1850_185007

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the yOz plane
def yOz_plane (p : Point3D) : Prop := p.x = 0

-- Define symmetry with respect to the yOz plane
def symmetric_to_yOz (a b : Point3D) : Prop :=
  b.x = -a.x ∧ b.y = a.y ∧ b.z = a.z

theorem symmetry_point_yOz :
  let a := Point3D.mk (-2) 4 3
  let b := Point3D.mk 2 4 3
  symmetric_to_yOz a b := by
  sorry

end symmetry_point_yOz_l1850_185007


namespace region_is_lower_left_l1850_185022

-- Define the line
def line (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x + y - 6 < 0

-- Define a point on the lower left side of the line
def lower_left_point (x y : ℝ) : Prop := x + y < 6

-- Theorem stating that the region is on the lower left side of the line
theorem region_is_lower_left :
  ∀ (x y : ℝ), region x y ↔ lower_left_point x y :=
sorry

end region_is_lower_left_l1850_185022


namespace least_months_to_triple_l1850_185045

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 1500

/-- The monthly interest rate as a decimal -/
def interest_rate : ℝ := 0.06

/-- The factor by which the borrowed amount increases each month -/
def growth_factor : ℝ := 1 + interest_rate

/-- The amount owed after t months -/
def amount_owed (t : ℕ) : ℝ := initial_amount * growth_factor ^ t

/-- Predicate that checks if the amount owed exceeds three times the initial amount -/
def exceeds_triple (t : ℕ) : Prop := amount_owed t > 3 * initial_amount

theorem least_months_to_triple :
  (∀ m : ℕ, m < 20 → ¬(exceeds_triple m)) ∧ exceeds_triple 20 :=
sorry

end least_months_to_triple_l1850_185045


namespace cube_sum_inequality_l1850_185031

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (z^3 + y^3) / (x^2 + x*y + y^2) + (x^3 + z^3) / (y^2 + y*z + z^2) + (y^3 + x^3) / (z^2 + z*x + x^2) ≥ 2 := by
  sorry

end cube_sum_inequality_l1850_185031


namespace rectangular_field_area_l1850_185081

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio. -/
theorem rectangular_field_area 
  (perimeter : ℝ) 
  (width_to_length_ratio : ℝ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : width_to_length_ratio = 1/3) : 
  let width := perimeter / (2 * (1 + 1/width_to_length_ratio))
  let length := width / width_to_length_ratio
  width * length = 243 := by
sorry

end rectangular_field_area_l1850_185081


namespace triangle_area_with_angle_bisector_l1850_185056

/-- The area of a triangle given two sides and the angle bisector between them. -/
theorem triangle_area_with_angle_bisector 
  (a b f_c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf_c : f_c > 0) 
  (h_triangle : 4 * a^2 * b^2 > (a + b)^2 * f_c^2) : 
  ∃ t : ℝ, t = ((a + b) * f_c) / (4 * a * b) * Real.sqrt (4 * a^2 * b^2 - (a + b)^2 * f_c^2) := by
  sorry


end triangle_area_with_angle_bisector_l1850_185056


namespace equation_solution_l1850_185013

theorem equation_solution : 
  ∀ x : ℝ, x > 0 → (x^(Real.log x / Real.log 5) = x^4 / 250 ↔ x = 5 ∨ x = 125) :=
by sorry

end equation_solution_l1850_185013


namespace no_solution_implies_a_leq_two_thirds_l1850_185006

theorem no_solution_implies_a_leq_two_thirds (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| < 4*x - 1 ∧ x < a)) → a ≤ 2/3 :=
by sorry

end no_solution_implies_a_leq_two_thirds_l1850_185006


namespace reciprocal_roots_iff_m_eq_p_l1850_185089

/-- A quadratic equation with coefficients p, q, and m -/
structure QuadraticEquation where
  p : ℝ
  q : ℝ
  m : ℝ

/-- The roots of a quadratic equation are reciprocals -/
def has_reciprocal_roots (eq : QuadraticEquation) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ eq.p * r^2 + eq.q * r + eq.m = 0 ∧ eq.p * (1/r)^2 + eq.q * (1/r) + eq.m = 0

/-- Theorem: The roots of px^2 + qx + m = 0 are reciprocals iff m = p -/
theorem reciprocal_roots_iff_m_eq_p (eq : QuadraticEquation) :
  has_reciprocal_roots eq ↔ eq.m = eq.p :=
sorry

end reciprocal_roots_iff_m_eq_p_l1850_185089


namespace stratified_sample_size_l1850_185054

theorem stratified_sample_size 
  (population_ratio_A : ℚ) 
  (population_ratio_B : ℚ) 
  (population_ratio_C : ℚ) 
  (sample_size_A : ℕ) 
  (total_sample_size : ℕ) :
  population_ratio_A = 3 / 14 →
  population_ratio_B = 4 / 14 →
  population_ratio_C = 7 / 14 →
  sample_size_A = 15 →
  population_ratio_A = sample_size_A / total_sample_size →
  total_sample_size = 70 := by
sorry

end stratified_sample_size_l1850_185054


namespace travel_ways_count_l1850_185017

/-- The number of available train trips -/
def train_trips : ℕ := 4

/-- The number of available ferry trips -/
def ferry_trips : ℕ := 3

/-- The total number of ways to travel from A to B -/
def total_ways : ℕ := train_trips + ferry_trips

theorem travel_ways_count : total_ways = 7 := by
  sorry

end travel_ways_count_l1850_185017


namespace floor_e_equals_two_l1850_185072

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by sorry

end floor_e_equals_two_l1850_185072


namespace tv_sale_value_increase_l1850_185061

theorem tv_sale_value_increase 
  (original_price original_quantity : ℝ) 
  (original_price_positive : 0 < original_price)
  (original_quantity_positive : 0 < original_quantity) :
  let price_reduction_factor := 0.8
  let sales_increase_factor := 1.8
  let new_price := price_reduction_factor * original_price
  let new_quantity := sales_increase_factor * original_quantity
  let original_sale_value := original_price * original_quantity
  let new_sale_value := new_price * new_quantity
  (new_sale_value - original_sale_value) / original_sale_value = 0.44 := by
sorry

end tv_sale_value_increase_l1850_185061


namespace red_cars_count_l1850_185032

/-- Given a parking lot where the ratio of red cars to black cars is 3:8
    and there are 90 black cars, prove that there are 33 red cars. -/
theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) :
  black_cars = 90 →
  ratio_red = 3 →
  ratio_black = 8 →
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 33 := by
  sorry

end red_cars_count_l1850_185032


namespace fixed_points_for_specific_values_range_of_a_for_two_distinct_fixed_points_l1850_185000

def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

theorem fixed_points_for_specific_values :
  is_fixed_point (f 1 (-2)) (-1) ∧ is_fixed_point (f 1 (-2)) 3 :=
sorry

theorem range_of_a_for_two_distinct_fixed_points :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) →
  (0 < a ∧ a < 1) :=
sorry

end fixed_points_for_specific_values_range_of_a_for_two_distinct_fixed_points_l1850_185000


namespace arthur_walk_distance_l1850_185088

def blocks_west : ℕ := 9
def blocks_south : ℕ := 15
def mile_per_block : ℚ := 1/4

theorem arthur_walk_distance :
  (blocks_west + blocks_south : ℚ) * mile_per_block = 6 := by
  sorry

end arthur_walk_distance_l1850_185088


namespace section_area_regular_triangular_pyramid_l1850_185030

/-- The area of a section in a regular triangular pyramid -/
theorem section_area_regular_triangular_pyramid
  (a h : ℝ)
  (ha : a > 0)
  (hh : h > (a * Real.sqrt 6) / 6) :
  let area := (3 * a^2 * h) / (4 * Real.sqrt (a^2 + 3 * h^2))
  ∃ (S : ℝ), S = area ∧ S > 0 :=
by sorry

end section_area_regular_triangular_pyramid_l1850_185030


namespace abs_neg_one_fourth_l1850_185051

theorem abs_neg_one_fourth : |(-1 : ℚ) / 4| = 1 / 4 := by
  sorry

end abs_neg_one_fourth_l1850_185051


namespace inequality_proof_l1850_185021

theorem inequality_proof (a b : ℝ) (h : a > b) : 2 - a < 2 - b := by
  sorry

end inequality_proof_l1850_185021


namespace net_pay_calculation_l1850_185027

/-- Calculate net pay given gross pay and taxes paid -/
def net_pay (gross_pay taxes_paid : ℕ) : ℕ :=
  gross_pay - taxes_paid

/-- Theorem: Given the conditions, prove that the net pay is 315 dollars -/
theorem net_pay_calculation (gross_pay taxes_paid : ℕ) 
  (h1 : gross_pay = 450)
  (h2 : taxes_paid = 135) :
  net_pay gross_pay taxes_paid = 315 := by
  sorry

end net_pay_calculation_l1850_185027


namespace stratified_sampling_juniors_l1850_185018

theorem stratified_sampling_juniors 
  (total_students : ℕ) 
  (juniors : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1200)
  (h2 : juniors = 500)
  (h3 : sample_size = 120) :
  (juniors : ℚ) / total_students * sample_size = 50 := by
sorry

end stratified_sampling_juniors_l1850_185018


namespace subtract_like_terms_l1850_185041

theorem subtract_like_terms (a b : ℝ) : 5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := by
  sorry

end subtract_like_terms_l1850_185041


namespace remainder_531531_mod_6_l1850_185037

theorem remainder_531531_mod_6 : 531531 % 6 = 3 := by
  sorry

end remainder_531531_mod_6_l1850_185037


namespace polynomial_coefficient_sum_l1850_185066

theorem polynomial_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  ((∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) ∧
   a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) →
  m = 3 ∨ m = -1 := by
sorry

end polynomial_coefficient_sum_l1850_185066


namespace swimming_improvement_l1850_185071

-- Define the initial performance
def initial_laps : ℕ := 15
def initial_time : ℕ := 30

-- Define the improved performance
def improved_laps : ℕ := 20
def improved_time : ℕ := 36

-- Define the improvement in lap time
def lap_time_improvement : ℚ := 
  (initial_time : ℚ) / initial_laps - (improved_time : ℚ) / improved_laps

-- Theorem statement
theorem swimming_improvement : lap_time_improvement = 0.2 := by
  sorry

end swimming_improvement_l1850_185071


namespace candy_distribution_l1850_185099

theorem candy_distribution (x : ℕ) : 
  x > 500 ∧ 
  x % 21 = 5 ∧ 
  x % 22 = 3 →
  x ≥ 509 :=
by sorry

end candy_distribution_l1850_185099


namespace function_approximation_by_additive_l1850_185085

/-- Given a function f: ℝ → ℝ satisfying |f(x+y) - f(x) - f(y)| ≤ 1 for all x, y ∈ ℝ,
    there exists a function g: ℝ → ℝ such that |f(x) - g(x)| ≤ 1 and
    g(x+y) = g(x) + g(y) for all x, y ∈ ℝ. -/
theorem function_approximation_by_additive (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ 
               (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end function_approximation_by_additive_l1850_185085


namespace swap_meet_backpack_price_l1850_185014

/-- Proves that the price of each backpack sold at the swap meet was $18 --/
theorem swap_meet_backpack_price :
  ∀ (swap_meet_price : ℕ),
    (48 : ℕ) = 17 + 10 + (48 - 17 - 10) →
    (576 : ℕ) = 48 * 12 →
    (442 : ℕ) = (17 * swap_meet_price + 10 * 25 + (48 - 17 - 10) * 22) - 576 →
    swap_meet_price = 18 := by
  sorry


end swap_meet_backpack_price_l1850_185014


namespace not_iff_eq_mul_eq_irrational_iff_irrational_plus_five_not_gt_implies_sq_gt_lt_three_implies_lt_five_l1850_185004

-- Statement 1
theorem not_iff_eq_mul_eq (a b c : ℝ) : ¬(a = b ↔ a * c = b * c) :=
sorry

-- Statement 2
theorem irrational_iff_irrational_plus_five (a : ℝ) : Irrational (a + 5) ↔ Irrational a :=
sorry

-- Statement 3
theorem not_gt_implies_sq_gt (a b : ℝ) : ¬(a > b → a^2 > b^2) :=
sorry

-- Statement 4
theorem lt_three_implies_lt_five (a : ℝ) : a < 3 → a < 5 :=
sorry

end not_iff_eq_mul_eq_irrational_iff_irrational_plus_five_not_gt_implies_sq_gt_lt_three_implies_lt_five_l1850_185004


namespace lucas_sandwich_problem_l1850_185019

/-- Luca's sandwich shop problem --/
theorem lucas_sandwich_problem (sandwich_price : ℝ) (discount_rate : ℝ) 
  (avocado_price : ℝ) (salad_price : ℝ) (total_bill : ℝ) 
  (h1 : sandwich_price = 8)
  (h2 : discount_rate = 1/4)
  (h3 : avocado_price = 1)
  (h4 : salad_price = 3)
  (h5 : total_bill = 12) :
  total_bill - (sandwich_price * (1 - discount_rate) + avocado_price + salad_price) = 2 := by
  sorry

#check lucas_sandwich_problem

end lucas_sandwich_problem_l1850_185019


namespace square_perimeter_problem_l1850_185065

theorem square_perimeter_problem (perimeter_C : ℝ) (area_C area_D : ℝ) :
  perimeter_C = 32 →
  area_D = area_C / 8 →
  ∃ (side_D : ℝ), side_D * side_D = area_D ∧ 4 * side_D = 8 * Real.sqrt 2 :=
by sorry

end square_perimeter_problem_l1850_185065


namespace dogwood_trees_planted_tomorrow_l1850_185046

theorem dogwood_trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_trees : ℕ) : 
  initial_trees = 7 → planted_today = 5 → final_trees = 16 → 
  final_trees - (initial_trees + planted_today) = 4 :=
by
  sorry

end dogwood_trees_planted_tomorrow_l1850_185046


namespace symmetric_function_g_l1850_185096

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1 - 2

-- Define the symmetry condition
def is_symmetric_about (g : ℝ → ℝ) (p : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, g x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

-- Theorem statement
theorem symmetric_function_g : 
  ∃ g : ℝ → ℝ, is_symmetric_about g (1, 2) f ∧ (∀ x, g x = 3 * x - 1) :=
sorry

end symmetric_function_g_l1850_185096


namespace max_fraction_sum_l1850_185073

theorem max_fraction_sum (a b c d : ℕ+) (h1 : a + c = 20) (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  (∀ a' b' c' d' : ℕ+, a' + c' = 20 → (a' : ℚ) / b' + (c' : ℚ) / d' < 1 → (a : ℚ) / b + (c : ℚ) / d ≤ (a' : ℚ) / b' + (c' : ℚ) / d') →
  (a : ℚ) / b + (c : ℚ) / d = 20 / 21 :=
sorry

end max_fraction_sum_l1850_185073


namespace smaller_screen_diagonal_l1850_185078

theorem smaller_screen_diagonal (d : ℝ) : 
  d > 0 → d^2 / 2 = 200 - 38 → d = 18 := by
  sorry

end smaller_screen_diagonal_l1850_185078


namespace sugar_solution_percentage_l1850_185002

theorem sugar_solution_percentage (original_percentage : ℝ) (final_percentage : ℝ) : 
  original_percentage = 10 →
  final_percentage = 18 →
  ∃ (second_percentage : ℝ),
    second_percentage = 42 ∧
    (3/4 * original_percentage + 1/4 * second_percentage) / 100 = final_percentage / 100 :=
by sorry

end sugar_solution_percentage_l1850_185002


namespace power_of_fraction_l1850_185063

theorem power_of_fraction : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end power_of_fraction_l1850_185063


namespace smallest_positive_root_comparison_l1850_185080

theorem smallest_positive_root_comparison : ∃ (x₁ x₂ : ℝ), 
  (x₁ > 0 ∧ x₂ > 0) ∧ 
  (x₁^2011 + 2011*x₁ - 1 = 0) ∧
  (x₂^2011 - 2011*x₂ + 1 = 0) ∧
  (∀ y₁ > 0, y₁^2011 + 2011*y₁ - 1 = 0 → y₁ ≥ x₁) ∧
  (∀ y₂ > 0, y₂^2011 - 2011*y₂ + 1 = 0 → y₂ ≥ x₂) ∧
  (x₁ < x₂) := by
sorry

end smallest_positive_root_comparison_l1850_185080


namespace javier_exercise_time_is_350_l1850_185040

/-- Calculates Javier's total exercise time given the conditions of the problem. -/
def javierExerciseTime (javier_daily_time : ℕ) (sanda_daily_time : ℕ) (sanda_days : ℕ) (total_time : ℕ) : ℕ :=
  let javier_days := (total_time - sanda_daily_time * sanda_days) / javier_daily_time
  javier_days * javier_daily_time

/-- Proves that Javier's total exercise time is 350 minutes given the problem conditions. -/
theorem javier_exercise_time_is_350 :
  javierExerciseTime 50 90 3 620 = 350 := by
  sorry

end javier_exercise_time_is_350_l1850_185040


namespace fifteenth_student_age_l1850_185050

theorem fifteenth_student_age 
  (total_students : ℕ)
  (avg_age_all : ℝ)
  (num_group1 : ℕ)
  (avg_age_group1 : ℝ)
  (num_group2 : ℕ)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + (num_group2 : ℝ) * avg_age_group2) = 11 :=
by sorry

end fifteenth_student_age_l1850_185050


namespace variation_problem_l1850_185033

/-- Given:
  - R varies directly as the square of S and inversely as T^2
  - When R = 3, T = 2, S = 1
  Prove that when R = 75 and T = 5, S = 12.5
-/
theorem variation_problem (R S T : ℝ) (c : ℝ) : 
  (∀ R S T, R = c * S^2 / T^2) →  -- Relationship between R, S, and T
  (3 = c * 1^2 / 2^2) →           -- Given condition: R = 3, S = 1, T = 2
  (75 = c * S^2 / 5^2) →          -- Target condition: R = 75, T = 5
  S = 12.5 := by                  -- Prove S = 12.5
sorry


end variation_problem_l1850_185033


namespace quadratic_condition_l1850_185024

/-- A quadratic equation in one variable is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

/-- The equation ax^2 - x + 2 = 0 -/
def equation (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - x + 2 = 0

theorem quadratic_condition (a : ℝ) :
  (∃ x, equation a x) → is_quadratic_equation a (-1) 2 :=
by
  sorry

end quadratic_condition_l1850_185024


namespace perfect_square_factors_of_2880_l1850_185058

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_2880 :
  let factorization := prime_factorization 2880
  (factorization = [(2, 6), (3, 2), (5, 1)]) →
  count_perfect_square_factors 2880 = 8 := by sorry

end perfect_square_factors_of_2880_l1850_185058


namespace valid_word_count_l1850_185023

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 3

/-- The length of the words -/
def word_length : ℕ := 20

/-- Function to calculate the number of valid words -/
def count_valid_words (n : ℕ) : ℕ :=
  alphabet_size * 2^(n - 1)

/-- Theorem stating the number of valid 20-letter words -/
theorem valid_word_count :
  count_valid_words word_length = 786432 :=
sorry

end valid_word_count_l1850_185023


namespace melissa_games_played_l1850_185083

/-- Given a player's points per game and total score, calculate the number of games played -/
def games_played (points_per_game : ℕ) (total_points : ℕ) : ℕ :=
  total_points / points_per_game

/-- Theorem: A player scoring 120 points per game with a total of 1200 points played 10 games -/
theorem melissa_games_played :
  games_played 120 1200 = 10 := by
  sorry

end melissa_games_played_l1850_185083


namespace stream_speed_l1850_185008

/-- Proves that the speed of a stream is 5 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 22 →
  distance = 135 →
  time = 5 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 5 := by
sorry

end stream_speed_l1850_185008


namespace expression_simplification_l1850_185044

theorem expression_simplification (x y : ℝ) : 
  4 * x + 8 * x^2 + 6 * y - (3 - 5 * x - 8 * x^2 - 2 * y) = 16 * x^2 + 9 * x + 8 * y - 3 :=
by sorry

end expression_simplification_l1850_185044


namespace kendra_minivans_count_l1850_185039

/-- The number of minivans Kendra saw in the afternoon -/
def afternoon_minivans : ℕ := 4

/-- The number of minivans Kendra saw in the evening -/
def evening_minivans : ℕ := 1

/-- The total number of minivans Kendra saw during her trip -/
def total_minivans : ℕ := afternoon_minivans + evening_minivans

theorem kendra_minivans_count : total_minivans = 5 := by
  sorry

end kendra_minivans_count_l1850_185039


namespace A_intersect_B_eq_B_l1850_185001

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem A_intersect_B_eq_B : A ∩ B = B := by sorry

end A_intersect_B_eq_B_l1850_185001


namespace gabrielles_peaches_l1850_185029

theorem gabrielles_peaches (martine benjy gabrielle : ℕ) 
  (h1 : martine = 2 * benjy + 6)
  (h2 : benjy = gabrielle / 3)
  (h3 : martine = 16) :
  gabrielle = 15 := by
  sorry

end gabrielles_peaches_l1850_185029


namespace smallest_positive_n_squared_l1850_185048

-- Define the circles c1 and c2
def c1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0
def c2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 175 = 0

-- Define a function to check if a point is on a line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the conditions for external and internal tangency
def externally_tangent (x y r : ℝ) : Prop := (x - 4)^2 + (y - 10)^2 = (r + 7)^2
def internally_tangent (x y r : ℝ) : Prop := (x + 4)^2 + (y - 10)^2 = (11 - r)^2

-- State the theorem
theorem smallest_positive_n_squared (n : ℝ) : 
  (∀ b : ℝ, b > 0 → b < n → 
    ¬∃ x y r : ℝ, on_line x y b ∧ externally_tangent x y r ∧ internally_tangent x y r) →
  (∃ x y r : ℝ, on_line x y n ∧ externally_tangent x y r ∧ internally_tangent x y r) →
  n^2 = 49/64 := by
sorry

end smallest_positive_n_squared_l1850_185048


namespace A_intersect_B_l1850_185059

def A : Set (ℤ × ℤ) := {(1, 2), (2, 1)}
def B : Set (ℤ × ℤ) := {p : ℤ × ℤ | p.1 - p.2 = 1}

theorem A_intersect_B : A ∩ B = {(2, 1)} := by sorry

end A_intersect_B_l1850_185059


namespace min_students_in_math_club_l1850_185009

/-- Represents a math club with boys and girls -/
structure MathClub where
  boys : ℕ
  girls : ℕ

/-- The condition that more than 60% of students are boys -/
def moreThan60PercentBoys (club : MathClub) : Prop :=
  (club.boys : ℚ) / (club.boys + club.girls : ℚ) > 60 / 100

/-- The theorem stating the minimum number of students in the club -/
theorem min_students_in_math_club :
  ∀ (club : MathClub),
  moreThan60PercentBoys club →
  club.girls = 5 →
  club.boys + club.girls ≥ 13 :=
by sorry

end min_students_in_math_club_l1850_185009


namespace best_representative_l1850_185049

structure Student where
  name : String
  average_time : Float
  variance : Float

def is_better (s1 s2 : Student) : Prop :=
  (s1.average_time < s2.average_time) ∨
  (s1.average_time = s2.average_time ∧ s1.variance < s2.variance)

def is_best (s : Student) (students : List Student) : Prop :=
  ∀ other ∈ students, s ≠ other → is_better s other

theorem best_representative (students : List Student) :
  let a := { name := "A", average_time := 7.9, variance := 1.4 }
  let b := { name := "B", average_time := 8.2, variance := 2.2 }
  let c := { name := "C", average_time := 7.9, variance := 2.4 }
  let d := { name := "D", average_time := 8.2, variance := 1.4 }
  students = [a, b, c, d] →
  is_best a students :=
by sorry

end best_representative_l1850_185049


namespace third_side_length_l1850_185038

/-- Triangle inequality theorem for a triangle with sides a, b, and c -/
axiom triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : 
  a + b > c ∧ b + c > a ∧ c + a > b

theorem third_side_length (x : ℝ) (hx : x > 0) : 
  (4 + x > 9 ∧ 9 + x > 4 ∧ 4 + 9 > x) → (x > 5 ∧ x < 13) := by sorry

end third_side_length_l1850_185038


namespace unique_perfect_cube_divisibility_l1850_185026

theorem unique_perfect_cube_divisibility : ∃! X : ℕ+, 
  (∃ Y : ℕ+, X = Y^3) ∧ 
  X = (555 * 465)^2 * (555 - 465)^3 + (555 - 465)^4 := by
  sorry

end unique_perfect_cube_divisibility_l1850_185026


namespace prob_same_color_is_correct_l1850_185076

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 3
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def prob_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem prob_same_color_is_correct : prob_same_color = 66 / 1330 := by
  sorry

end prob_same_color_is_correct_l1850_185076


namespace geometric_sequence_sum_l1850_185091

theorem geometric_sequence_sum (a₁ r : ℝ) (n : ℕ) : 
  a₁ = 4 → r = 2 → n = 4 → 
  (a₁ * (1 - r^n)) / (1 - r) = 60 := by
  sorry

end geometric_sequence_sum_l1850_185091


namespace missing_number_proof_l1850_185086

theorem missing_number_proof (x y : ℝ) : 
  (12 + x + 42 + y + 104) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  y = 78 := by
sorry

end missing_number_proof_l1850_185086


namespace students_with_two_skills_l1850_185097

theorem students_with_two_skills (total : ℕ) (no_poetry : ℕ) (no_paint : ℕ) (no_instrument : ℕ) 
  (h1 : total = 150)
  (h2 : no_poetry = 80)
  (h3 : no_paint = 90)
  (h4 : no_instrument = 60) :
  let poetry := total - no_poetry
  let paint := total - no_paint
  let instrument := total - no_instrument
  let two_skills := poetry + paint + instrument - total
  two_skills = 70 := by
  sorry

end students_with_two_skills_l1850_185097


namespace polynomial_simplification_l1850_185003

theorem polynomial_simplification (x : ℝ) :
  (3 * x^4 + 2 * x^3 - 9 * x^2 + 4 * x - 5) + (-5 * x^4 - 3 * x^3 + x^2 - 4 * x + 7) =
  -2 * x^4 - x^3 - 8 * x^2 + 2 := by
  sorry

end polynomial_simplification_l1850_185003


namespace union_of_A_and_B_l1850_185094

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end union_of_A_and_B_l1850_185094


namespace dot_product_specific_vectors_l1850_185093

/-- Given two vectors a and b in a 2D plane with specific magnitudes and angle between them,
    prove that the dot product of a and (a + b) is 12. -/
theorem dot_product_specific_vectors (a b : ℝ × ℝ) :
  ‖a‖ = 4 →
  ‖b‖ = Real.sqrt 2 →
  a • b = -4 →
  a • (a + b) = 12 := by
  sorry

end dot_product_specific_vectors_l1850_185093


namespace equal_distribution_of_eggs_l1850_185016

-- Define the number of eggs
def total_eggs : ℕ := 2 * 12

-- Define the number of people
def num_people : ℕ := 4

-- Define the function to calculate eggs per person
def eggs_per_person (total : ℕ) (people : ℕ) : ℕ := total / people

-- Theorem to prove
theorem equal_distribution_of_eggs :
  eggs_per_person total_eggs num_people = 6 := by
  sorry

end equal_distribution_of_eggs_l1850_185016


namespace sandwich_combinations_l1850_185011

theorem sandwich_combinations (meat : ℕ) (cheese : ℕ) (condiment : ℕ) :
  meat = 12 →
  cheese = 8 →
  condiment = 5 →
  (meat) * (cheese.choose 2) * (condiment) = 1680 :=
by
  sorry

end sandwich_combinations_l1850_185011


namespace three_digit_number_ending_in_five_divisible_by_five_l1850_185092

theorem three_digit_number_ending_in_five_divisible_by_five (N : ℕ) :
  100 ≤ N ∧ N < 1000 ∧ N % 10 = 5 → N % 5 = 0 := by
  sorry

end three_digit_number_ending_in_five_divisible_by_five_l1850_185092


namespace functional_equation_solution_l1850_185060

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

/-- The main theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesFunctionalEquation f) :
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end functional_equation_solution_l1850_185060


namespace largest_number_divisible_by_88_has_4_digits_l1850_185084

def largest_number_divisible_by_88 : ℕ := 9944

theorem largest_number_divisible_by_88_has_4_digits :
  (largest_number_divisible_by_88 ≥ 1000) ∧ (largest_number_divisible_by_88 < 10000) := by
  sorry

end largest_number_divisible_by_88_has_4_digits_l1850_185084


namespace extreme_points_when_a_is_one_extreme_points_condition_l1850_185047

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x + 1 else a*x

-- Theorem 1: When a = 1, f(x) has exactly two extreme points
theorem extreme_points_when_a_is_one :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧
  (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f 1 y ≤ f 1 x) ∨
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f 1 y ≥ f 1 x)) ↔ (x = x1 ∨ x = x2)) :=
sorry

-- Theorem 2: f(x) has exactly two extreme points iff 0 < a < 2
theorem extreme_points_condition :
  ∀ (a : ℝ), (∃! (x1 x2 : ℝ), x1 ≠ x2 ∧
  (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f a y ≤ f a x) ∨
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f a y ≥ f a x)) ↔ (x = x1 ∨ x = x2)))
  ↔ (0 < a ∧ a < 2) :=
sorry

end extreme_points_when_a_is_one_extreme_points_condition_l1850_185047
