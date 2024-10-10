import Mathlib

namespace cash_preference_factors_l1199_119978

/-- Represents an economic factor influencing payment preference --/
structure EconomicFactor where
  description : String
  favors_cash : Bool

/-- Represents a large retail chain --/
structure RetailChain where
  name : String
  payment_preference : String

/-- Theorem: There exist at least three distinct economic factors that could lead large retail chains to prefer cash payments --/
theorem cash_preference_factors :
  ∃ (f1 f2 f3 : EconomicFactor),
    f1.favors_cash ∧ f2.favors_cash ∧ f3.favors_cash ∧
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (∃ (rc : RetailChain), rc.payment_preference = "cash") :=
by sorry

/-- Definition: Efficiency of operations as an economic factor --/
def efficiency_factor : EconomicFactor :=
  { description := "Efficiency of Operations", favors_cash := true }

/-- Definition: Cost of handling transactions as an economic factor --/
def cost_factor : EconomicFactor :=
  { description := "Cost of Handling Transactions", favors_cash := true }

/-- Definition: Risk of fraud as an economic factor --/
def risk_factor : EconomicFactor :=
  { description := "Risk of Fraud", favors_cash := true }

end cash_preference_factors_l1199_119978


namespace function_extrema_implies_a_range_l1199_119993

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) →
  (a < -1 ∨ a > 2) :=
by sorry

end function_extrema_implies_a_range_l1199_119993


namespace sum_real_imag_parts_of_z_l1199_119959

theorem sum_real_imag_parts_of_z : ∃ z : ℂ, z * (2 + Complex.I) = 2 * Complex.I - 1 ∧ 
  z.re + z.im = 1 := by sorry

end sum_real_imag_parts_of_z_l1199_119959


namespace y_value_at_x_4_l1199_119957

/-- Given a function y = k * x^(1/4) where y = 3√2 when x = 81, 
    prove that y = 2 when x = 4 -/
theorem y_value_at_x_4 (k : ℝ) :
  (∀ x : ℝ, x > 0 → k * x^(1/4) = 3 * Real.sqrt 2 ↔ x = 81) →
  k * 4^(1/4) = 2 :=
by sorry

end y_value_at_x_4_l1199_119957


namespace combinations_count_l1199_119988

/-- Represents the cost of a pencil in cents -/
def pencil_cost : ℕ := 5

/-- Represents the cost of an eraser in cents -/
def eraser_cost : ℕ := 10

/-- Represents the cost of a notebook in cents -/
def notebook_cost : ℕ := 20

/-- Represents the total amount Mrs. Hilt has in cents -/
def total_amount : ℕ := 50

/-- Counts the number of valid combinations of items that can be purchased -/
def count_combinations : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ =>
    pencil_cost * t.1 + eraser_cost * t.2.1 + notebook_cost * t.2.2 = total_amount)
    (Finset.product (Finset.range (total_amount / pencil_cost + 1))
      (Finset.product (Finset.range (total_amount / eraser_cost + 1))
        (Finset.range (total_amount / notebook_cost + 1))))).card

theorem combinations_count :
  count_combinations = 12 := by sorry

end combinations_count_l1199_119988


namespace simplify_fraction_l1199_119951

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l1199_119951


namespace two_digit_reverse_sum_l1199_119960

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverse digits of a two-digit integer -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Main theorem -/
theorem two_digit_reverse_sum (x y m : ℕ) : 
  TwoDigitInt x ∧ 
  TwoDigitInt y ∧
  y = reverseDigits x ∧
  x^2 - y^2 = 4 * m^2 ∧
  0 < m
  →
  x + y + m = 105 := by
sorry

end two_digit_reverse_sum_l1199_119960


namespace cricket_innings_problem_l1199_119948

theorem cricket_innings_problem (initial_average : ℝ) (runs_next_inning : ℕ) (average_increase : ℝ) :
  initial_average = 15 ∧ runs_next_inning = 59 ∧ average_increase = 4 →
  ∃ n : ℕ, n = 10 ∧
    initial_average * n + runs_next_inning = (initial_average + average_increase) * (n + 1) :=
by
  sorry

end cricket_innings_problem_l1199_119948


namespace hyperbola_eccentricity_l1199_119946

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem stating that the eccentricity of the hyperbola is either √5 or √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  eccentricity h = Real.sqrt 5 ∨ eccentricity h = (Real.sqrt 5) / 2 := by
  sorry

end hyperbola_eccentricity_l1199_119946


namespace min_value_expression_min_value_achievable_l1199_119944

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 := by
  sorry

end min_value_expression_min_value_achievable_l1199_119944


namespace ternary_221_greater_than_binary_10111_l1199_119945

/-- Converts a ternary number (represented as a list of digits) to decimal --/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- Converts a binary number (represented as a list of digits) to decimal --/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (2^i)) 0

/-- The ternary number 221 --/
def a : List Nat := [1, 2, 2]

/-- The binary number 10111 --/
def b : List Nat := [1, 1, 1, 0, 1]

theorem ternary_221_greater_than_binary_10111 :
  ternary_to_decimal a > binary_to_decimal b := by
  sorry

end ternary_221_greater_than_binary_10111_l1199_119945


namespace jasons_library_visits_l1199_119914

/-- Jason's library visits in 4 weeks -/
def jasons_visits (williams_weekly_visits : ℕ) (jasons_multiplier : ℕ) (weeks : ℕ) : ℕ :=
  williams_weekly_visits * jasons_multiplier * weeks

/-- Theorem: Jason's library visits in 4 weeks -/
theorem jasons_library_visits :
  jasons_visits 2 4 4 = 32 := by
  sorry

end jasons_library_visits_l1199_119914


namespace bmw_sales_l1199_119920

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (toyota_percent : ℚ) (nissan_percent : ℚ)
  (h_total : total = 300)
  (h_ford : ford_percent = 10 / 100)
  (h_toyota : toyota_percent = 20 / 100)
  (h_nissan : nissan_percent = 30 / 100) :
  (total : ℚ) * (1 - (ford_percent + toyota_percent + nissan_percent)) = 120 := by
  sorry

end bmw_sales_l1199_119920


namespace min_value_x_plus_four_over_x_l1199_119975

theorem min_value_x_plus_four_over_x (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end min_value_x_plus_four_over_x_l1199_119975


namespace quadratic_solution_l1199_119928

theorem quadratic_solution (b : ℝ) : (5^2 + b*5 - 35 = 0) → b = 2 := by
  sorry

end quadratic_solution_l1199_119928


namespace percentage_relation_l1199_119903

theorem percentage_relation (X A B : ℝ) (hA : A = 0.05 * X) (hB : B = 0.25 * X) :
  A = 0.2 * B := by sorry

end percentage_relation_l1199_119903


namespace tan_30_15_product_simplification_l1199_119931

theorem tan_30_15_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end tan_30_15_product_simplification_l1199_119931


namespace correlation_difference_l1199_119986

/-- Represents a relationship between two variables -/
structure Relationship where
  var1 : String
  var2 : String
  description : String

/-- Determines if a relationship represents a positive correlation -/
def is_positive_correlation (r : Relationship) : Bool :=
  sorry  -- The actual implementation would depend on how we define positive correlation

/-- Given set of relationships -/
def relationships : List Relationship := [
  { var1 := "teacher quality", var2 := "student performance", description := "A great teacher produces outstanding students" },
  { var1 := "tide level", var2 := "boat height", description := "A rising tide lifts all boats" },
  { var1 := "moon brightness", var2 := "visible stars", description := "The brighter the moon, the fewer the stars" },
  { var1 := "climbing height", var2 := "viewing distance", description := "Climbing high to see far" }
]

theorem correlation_difference :
  ∃ (i : Fin 4), ¬(is_positive_correlation (relationships.get i)) ∧
    (∀ (j : Fin 4), j ≠ i → is_positive_correlation (relationships.get j)) :=
  sorry

end correlation_difference_l1199_119986


namespace remainder_theorem_l1199_119994

/-- The polynomial f(x) = x^4 - 6x^3 + 12x^2 + 20x - 8 -/
def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 12*x^2 + 20*x - 8

/-- The theorem stating that the remainder when f(x) is divided by (x-4) is 136 -/
theorem remainder_theorem : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 4) * q x + 136 := by
  sorry

end remainder_theorem_l1199_119994


namespace intersection_equivalence_l1199_119906

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (intersect : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_intersects_plane : Line → Plane → Prop)
variable (plane_intersects_plane : Plane → Plane → Prop)

-- Define our specific objects
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem intersection_equivalence 
  (h1 : intersect l m)
  (h2 : in_plane l α)
  (h3 : in_plane m α)
  (h4 : ¬ in_plane l β)
  (h5 : ¬ in_plane m β)
  : (line_intersects_plane l β ∨ line_intersects_plane m β) ↔ plane_intersects_plane α β :=
sorry

end intersection_equivalence_l1199_119906


namespace cades_marbles_l1199_119981

/-- The number of marbles Cade has after receiving marbles from Dylan and Ellie -/
def total_marbles (initial : ℕ) (from_dylan : ℕ) (from_ellie : ℕ) : ℕ :=
  initial + from_dylan + from_ellie

/-- Theorem stating that Cade's total marbles after receiving from Dylan and Ellie is 108 -/
theorem cades_marbles :
  total_marbles 87 8 13 = 108 := by
  sorry

end cades_marbles_l1199_119981


namespace at_op_difference_l1199_119913

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem at_op_difference : at_op 5 9 - at_op 9 5 = 16 := by sorry

end at_op_difference_l1199_119913


namespace arrangements_three_male_two_female_l1199_119925

/-- The number of ways to arrange 3 male and 2 female students in a row,
    such that the female students do not stand at either end -/
def arrangements (n_male : ℕ) (n_female : ℕ) : ℕ :=
  if n_male = 3 ∧ n_female = 2 then
    (n_male + n_female - 2).choose n_female * n_male.factorial
  else
    0

theorem arrangements_three_male_two_female :
  arrangements 3 2 = 36 :=
sorry

end arrangements_three_male_two_female_l1199_119925


namespace cube_volume_equals_surface_area_l1199_119956

theorem cube_volume_equals_surface_area (s : ℝ) (h : s > 0) :
  s^3 = 6 * s^2 → s = 6 := by
  sorry

end cube_volume_equals_surface_area_l1199_119956


namespace undetermined_zeros_l1199_119937

theorem undetermined_zeros (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) (h2 : f a * f b < 0) :
  ∃ (n : ℕ), n ≥ 0 ∧ (∃ (x : ℝ), x ∈ Set.Ioo a b ∧ f x = 0) ∧
  ¬ (∀ (m : ℕ), m ≠ n → ¬ (∃ (x : ℝ), x ∈ Set.Ioo a b ∧ f x = 0 ∧
    (∃ (y : ℝ), y ≠ x ∧ y ∈ Set.Ioo a b ∧ f y = 0))) :=
sorry

end undetermined_zeros_l1199_119937


namespace chimney_bricks_count_l1199_119915

/-- The number of bricks in the chimney. -/
def chimney_bricks : ℕ := 288

/-- The time it takes Brenda to build the chimney alone (in hours). -/
def brenda_time : ℕ := 8

/-- The time it takes Brandon to build the chimney alone (in hours). -/
def brandon_time : ℕ := 12

/-- The reduction in combined output when working together (in bricks per hour). -/
def output_reduction : ℕ := 12

/-- The time it takes Brenda and Brandon to build the chimney together (in hours). -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the chimney is 288. -/
theorem chimney_bricks_count : 
  chimney_bricks = 288 ∧
  brenda_time = 8 ∧
  brandon_time = 12 ∧
  output_reduction = 12 ∧
  combined_time = 6 ∧
  (combined_time * ((chimney_bricks / brenda_time + chimney_bricks / brandon_time) - output_reduction) = chimney_bricks) :=
by sorry

end chimney_bricks_count_l1199_119915


namespace root_equation_sum_l1199_119943

theorem root_equation_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 1 = 0 → x^4 + a*x^2 + b*x + c = 0) →
  a + b + 4*c + 100 = 93 := by
sorry

end root_equation_sum_l1199_119943


namespace wipes_per_pack_l1199_119912

theorem wipes_per_pack (wipes_per_day : ℕ) (days : ℕ) (num_packs : ℕ) : 
  wipes_per_day = 2 → days = 360 → num_packs = 6 → 
  (wipes_per_day * days) / num_packs = 120 := by
  sorry

end wipes_per_pack_l1199_119912


namespace random_subset_is_sample_l1199_119900

/-- Represents a population of elements -/
structure Population (α : Type) where
  elements : Finset α
  size : ℕ
  size_eq : elements.card = size

/-- Represents a sample taken from a population -/
structure Sample (α : Type) where
  elements : Finset α
  size : ℕ
  size_eq : elements.card = size

/-- Defines what it means for a sample to be from a population -/
def is_sample_of {α : Type} (s : Sample α) (p : Population α) : Prop :=
  s.elements ⊆ p.elements ∧ s.size < p.size

/-- The theorem statement -/
theorem random_subset_is_sample 
  {α : Type} (p : Population α) (s : Sample α) 
  (h_p_size : p.size = 50000) 
  (h_s_size : s.size = 2000) 
  (h_subset : s.elements ⊆ p.elements) : 
  is_sample_of s p := by
  sorry


end random_subset_is_sample_l1199_119900


namespace complex_expressions_calculation_l1199_119909

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define square root of 3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

theorem complex_expressions_calculation :
  -- Expression 1
  ((1 + 2*i)^2 + 3*(1 - i)) / (2 + i) = 1/5 + 2/5*i ∧
  -- Expression 2
  (1 - i) / (1 + i)^2 + (1 + i) / (1 - i)^2 = -1 ∧
  -- Expression 3
  (1 - sqrt3*i) / (sqrt3 + i)^2 = -1/4 - (sqrt3/4)*i :=
by sorry

end complex_expressions_calculation_l1199_119909


namespace symmetry_of_lines_l1199_119932

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := y - x = 0

/-- The original line -/
def original_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line_of_symmetry -/
theorem symmetry_of_lines :
  ∀ (x y x' y' : ℝ),
    original_line x y →
    line_of_symmetry ((x + x') / 2) ((y + y') / 2) →
    symmetric_line x' y' :=
by sorry

end symmetry_of_lines_l1199_119932


namespace inequality_proof_l1199_119962

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end inequality_proof_l1199_119962


namespace second_number_in_set_l1199_119973

theorem second_number_in_set (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + x + 13) / 3 + 9 → x = 70 := by
  sorry

end second_number_in_set_l1199_119973


namespace paint_room_time_l1199_119983

/-- Andy's painting rate in rooms per hour -/
def andy_rate : ℚ := 1 / 4

/-- Bob's painting rate in rooms per hour -/
def bob_rate : ℚ := 1 / 6

/-- The combined painting rate of Andy and Bob in rooms per hour -/
def combined_rate : ℚ := andy_rate + bob_rate

/-- The time taken to paint the room, including the lunch break -/
def t : ℚ := 22 / 5

theorem paint_room_time :
  (combined_rate * (t - 2) = 1) ∧ (combined_rate = 5 / 12) := by
  sorry

end paint_room_time_l1199_119983


namespace science_club_enrollment_l1199_119950

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : math = 80)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 5 := by
  sorry

end science_club_enrollment_l1199_119950


namespace max_value_of_d_l1199_119918

theorem max_value_of_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ + (5 + Real.sqrt 105) / 2 = 10 ∧
                    a₀ * b₀ + a₀ * c₀ + a₀ * ((5 + Real.sqrt 105) / 2) + 
                    b₀ * c₀ + b₀ * ((5 + Real.sqrt 105) / 2) + 
                    c₀ * ((5 + Real.sqrt 105) / 2) = 20 := by
  sorry

end max_value_of_d_l1199_119918


namespace geometric_sequence_ratio_l1199_119921

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : a 1 + a 4 = 18) (h3 : a 2 * a 3 = 32) : 
  ∃ q : ℝ, (q = 1/2 ∨ q = 2) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end geometric_sequence_ratio_l1199_119921


namespace buratino_apples_theorem_l1199_119984

theorem buratino_apples_theorem (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_distinct : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > a₆) :
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ + x₂ = a₁ ∧ 
    x₁ + x₃ = a₂ ∧ 
    x₂ + x₃ = a₃ ∧ 
    x₃ + x₄ = a₄ ∧ 
    x₁ + x₄ ≥ a₅ ∧ 
    x₂ + x₄ ≥ a₆ ∧
    ∀ y₁ y₂ y₃ y₄ : ℝ, 
      (y₁ + y₂ = a₁ → y₁ + y₃ = a₂ → y₂ + y₃ = a₃ → y₃ + y₄ = a₄ → 
       y₁ + y₄ = a₅ → y₂ + y₄ = a₆) → False :=
by sorry

end buratino_apples_theorem_l1199_119984


namespace arithmetic_sequence_a6_l1199_119908

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 := by
  sorry

end arithmetic_sequence_a6_l1199_119908


namespace election_winner_votes_l1199_119980

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  (winner_percentage = 62 / 100) →
  (winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) →
  (vote_difference = 300) →
  (winner_percentage * total_votes = 775) := by
  sorry

end election_winner_votes_l1199_119980


namespace square_equality_l1199_119989

theorem square_equality : (2023 + (-1011.5))^2 = (-1011.5)^2 := by
  sorry

end square_equality_l1199_119989


namespace vertex_of_quadratic_l1199_119985

def f (x : ℝ) : ℝ := -3 * x^2 + 2

theorem vertex_of_quadratic (x : ℝ) :
  (∀ x, f x ≤ f 0) ∧ f 0 = 2 := by sorry

end vertex_of_quadratic_l1199_119985


namespace second_quadrant_trig_l1199_119979

theorem second_quadrant_trig (α : Real) (h : π / 2 < α ∧ α < π) : 
  Real.tan α + Real.sin α < 0 := by
  sorry

end second_quadrant_trig_l1199_119979


namespace incorrect_induction_proof_l1199_119991

theorem incorrect_induction_proof (n : ℕ+) : 
  ¬(∀ k : ℕ+, (∀ m : ℕ+, m < k → Real.sqrt (m^2 + m) < m + 1) → 
    Real.sqrt ((k+1)^2 + (k+1)) < (k+1) + 1) := by
  sorry

#check incorrect_induction_proof

end incorrect_induction_proof_l1199_119991


namespace geometric_sequence_property_l1199_119924

/-- Given a geometric sequence {aₙ}, if a₁a₂a₃ = -8, then a₂ = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence property
  a 1 * a 2 * a 3 = -8 →                -- given condition
  a 2 = -2 := by
sorry

end geometric_sequence_property_l1199_119924


namespace max_assembly_and_impossibility_of_simultaneous_completion_l1199_119949

/-- Represents the number of wooden boards available -/
structure WoodenBoards :=
  (typeA : ℕ)
  (typeB : ℕ)

/-- Represents the requirements for assembling a desk and a chair -/
structure AssemblyRequirements :=
  (deskTypeA : ℕ)
  (deskTypeB : ℕ)
  (chairTypeA : ℕ)
  (chairTypeB : ℕ)

/-- Represents the assembly time for a desk and a chair -/
structure AssemblyTime :=
  (desk : ℕ)
  (chair : ℕ)

/-- Theorem stating the maximum number of desks and chairs that can be assembled
    and the impossibility of simultaneous completion -/
theorem max_assembly_and_impossibility_of_simultaneous_completion
  (boards : WoodenBoards)
  (requirements : AssemblyRequirements)
  (students : ℕ)
  (time : AssemblyTime)
  (h1 : boards.typeA = 400)
  (h2 : boards.typeB = 500)
  (h3 : requirements.deskTypeA = 2)
  (h4 : requirements.deskTypeB = 1)
  (h5 : requirements.chairTypeA = 1)
  (h6 : requirements.chairTypeB = 2)
  (h7 : students = 30)
  (h8 : time.desk = 10)
  (h9 : time.chair = 7) :
  (∃ (desks chairs : ℕ),
    desks = 100 ∧
    chairs = 200 ∧
    desks * requirements.deskTypeA + chairs * requirements.chairTypeA ≤ boards.typeA ∧
    desks * requirements.deskTypeB + chairs * requirements.chairTypeB ≤ boards.typeB ∧
    ∀ (desks' chairs' : ℕ),
      desks' > desks ∨ chairs' > chairs →
      desks' * requirements.deskTypeA + chairs' * requirements.chairTypeA > boards.typeA ∨
      desks' * requirements.deskTypeB + chairs' * requirements.chairTypeB > boards.typeB) ∧
  (∀ (group : ℕ),
    group ≤ students →
    (desks : ℚ) * time.desk / group ≠ (chairs : ℚ) * time.chair / (students - group)) :=
by sorry

end max_assembly_and_impossibility_of_simultaneous_completion_l1199_119949


namespace sqrt_sum_squares_geq_sqrt2_sum_l1199_119990

theorem sqrt_sum_squares_geq_sqrt2_sum (a b : ℝ) : 
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end sqrt_sum_squares_geq_sqrt2_sum_l1199_119990


namespace initial_price_equation_l1199_119907

/-- The initial price of speakers before discount -/
def initial_price : ℝ := 475

/-- The final price paid after discount -/
def final_price : ℝ := 199

/-- The discount amount saved -/
def discount : ℝ := 276

/-- Theorem stating that the initial price is equal to the sum of the final price and the discount -/
theorem initial_price_equation : initial_price = final_price + discount := by
  sorry

end initial_price_equation_l1199_119907


namespace equation_equality_l1199_119911

theorem equation_equality (a : ℝ) (h : a ≠ 0) :
  ((1 / a) / ((1 / a) * (1 / a)) - 1 / a) / (1 / a) = (a + 1) * (a - 1) := by
  sorry

end equation_equality_l1199_119911


namespace max_intersections_theorem_l1199_119972

/-- A convex polygon -/
structure ConvexPolygon where
  sides : ℕ

/-- The configuration of two convex polygons where one is contained within the other -/
structure PolygonConfiguration where
  outer : ConvexPolygon
  inner : ConvexPolygon
  inner_contained : inner.sides ≤ outer.sides
  no_coincident_sides : Bool

/-- The maximum number of intersection points between the sides of two polygons in the given configuration -/
def max_intersections (config : PolygonConfiguration) : ℕ :=
  config.inner.sides * config.outer.sides

/-- Theorem stating that the maximum number of intersections is the product of the number of sides -/
theorem max_intersections_theorem (config : PolygonConfiguration) :
  max_intersections config = config.inner.sides * config.outer.sides :=
sorry

end max_intersections_theorem_l1199_119972


namespace largest_prime_factor_of_sum_of_divisors_450_l1199_119998

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define M as the sum of divisors of 450
def M : ℕ := sum_of_divisors 450

-- Define a function to get the largest prime factor
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_sum_of_divisors_450 :
  largest_prime_factor M = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_450_l1199_119998


namespace set_operations_l1199_119963

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B : Set ℝ := {x | -1 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define the universal set U
def U : Set ℝ := Set.univ

theorem set_operations :
  (A ∪ B = {x | -2 < x ∧ x < 5}) ∧
  (A ∩ B = {x | 0 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ Bᶜ = U) ∧
  (A ∩ Bᶜ = {x | -2 < x ∧ x < 0} ∪ {x | 3 < x ∧ x < 5}) :=
by sorry

end set_operations_l1199_119963


namespace sin_inequality_solution_set_l1199_119997

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2*n - 1)*Real.pi - θ < x ∧ x < 2*n*Real.pi + θ} = {x : ℝ | Real.sin x < a} := by
  sorry

end sin_inequality_solution_set_l1199_119997


namespace complement_intersection_AB_l1199_119947

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_AB : (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end complement_intersection_AB_l1199_119947


namespace some_base_value_l1199_119992

theorem some_base_value (k : ℕ) (some_base : ℝ) 
  (h1 : (1/2)^16 * (1/some_base)^k = 1/(18^16))
  (h2 : k = 8) : 
  some_base = 81 := by
sorry

end some_base_value_l1199_119992


namespace happy_properties_l1199_119999

/-- A positive integer is happy if it can be expressed as the sum of two squares. -/
def IsHappy (n : ℕ+) : Prop :=
  ∃ a b : ℤ, n.val = a^2 + b^2

theorem happy_properties (t : ℕ+) (ht : IsHappy t) :
  (IsHappy (2 * t)) ∧ (¬IsHappy (3 * t)) := by
  sorry

end happy_properties_l1199_119999


namespace jane_lost_twenty_points_l1199_119969

/-- Represents the card game scenario --/
structure CardGame where
  pointsPerWin : ℕ
  totalRounds : ℕ
  finalPoints : ℕ

/-- Calculates the points lost in the card game --/
def pointsLost (game : CardGame) : ℕ :=
  game.pointsPerWin * game.totalRounds - game.finalPoints

/-- Theorem stating that Jane lost 20 points --/
theorem jane_lost_twenty_points :
  let game : CardGame := {
    pointsPerWin := 10,
    totalRounds := 8,
    finalPoints := 60
  }
  pointsLost game = 20 := by
  sorry


end jane_lost_twenty_points_l1199_119969


namespace coefficient_of_x_l1199_119927

theorem coefficient_of_x (x : ℝ) : 
  let expr := 4*(x - 5) + 3*(2 - 3*x^2 + 6*x) - 10*(3*x - 2)
  ∃ (a b c : ℝ), expr = a*x^2 + (-8)*x + c :=
by sorry

end coefficient_of_x_l1199_119927


namespace unique_solution_l1199_119941

def is_valid_number (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧
  (18600 + 10 * a + b) % 3 = 2 ∧
  (18600 + 10 * a + b) % 5 = 3 ∧
  (18600 + 10 * a + b) % 11 = 0

theorem unique_solution :
  ∃! (a b : ℕ), is_valid_number a b ∧ a = 2 ∧ b = 3 :=
sorry

end unique_solution_l1199_119941


namespace successive_discounts_equivalence_l1199_119940

theorem successive_discounts_equivalence :
  let discount1 : ℝ := 0.15
  let discount2 : ℝ := 0.10
  let discount3 : ℝ := 0.05
  let equivalent_single_discount : ℝ := 1 - (1 - discount1) * (1 - discount2) * (1 - discount3)
  equivalent_single_discount = 0.27325 :=
by sorry

end successive_discounts_equivalence_l1199_119940


namespace complex_equation_solution_l1199_119961

theorem complex_equation_solution (m : ℝ) : 
  let z₁ : ℂ := m^2 - 3*m + m^2*Complex.I
  let z₂ : ℂ := 4 + (5*m + 6)*Complex.I
  z₁ - z₂ = 0 → m = -1 := by
  sorry

end complex_equation_solution_l1199_119961


namespace cube_inequality_negation_l1199_119976

theorem cube_inequality_negation (x y : ℝ) (h : x > y) : 
  ¬(x^3 > y^3) ↔ x^3 ≤ y^3 := by
sorry

end cube_inequality_negation_l1199_119976


namespace range_of_a_l1199_119982

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 ≥ a

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l1199_119982


namespace monica_students_count_l1199_119926

/-- The number of students Monica sees each day -/
def monica_total_students : ℕ :=
  let first_class : ℕ := 20
  let second_third_classes : ℕ := 25 + 25
  let fourth_class : ℕ := first_class / 2
  let fifth_sixth_classes : ℕ := 28 + 28
  first_class + second_third_classes + fourth_class + fifth_sixth_classes

/-- Theorem stating the total number of students Monica sees each day -/
theorem monica_students_count : monica_total_students = 136 := by
  sorry

end monica_students_count_l1199_119926


namespace youngest_child_age_l1199_119971

def total_bill : ℚ := 12.25
def mother_meal : ℚ := 3.75
def cost_per_year : ℚ := 0.5

structure Family :=
  (triplet_age : ℕ)
  (youngest_age : ℕ)

def valid_family (f : Family) : Prop :=
  f.youngest_age < f.triplet_age ∧
  mother_meal + cost_per_year * (3 * f.triplet_age + f.youngest_age) = total_bill

theorem youngest_child_age :
  ∃ (f₁ f₂ : Family), valid_family f₁ ∧ valid_family f₂ ∧
    f₁.youngest_age = 2 ∧ f₂.youngest_age = 5 ∧
    ∀ (f : Family), valid_family f → f.youngest_age = 2 ∨ f.youngest_age = 5 :=
sorry

end youngest_child_age_l1199_119971


namespace book_sale_gain_percentage_l1199_119987

/-- Calculates the desired gain percentage for a book sale --/
theorem book_sale_gain_percentage 
  (loss_price : ℝ) 
  (loss_percentage : ℝ) 
  (desired_price : ℝ) : 
  loss_price = 800 ∧ 
  loss_percentage = 20 ∧ 
  desired_price = 1100 → 
  (desired_price - loss_price / (1 - loss_percentage / 100)) / 
  (loss_price / (1 - loss_percentage / 100)) * 100 = 10 :=
by
  sorry

end book_sale_gain_percentage_l1199_119987


namespace max_value_expression_l1199_119902

theorem max_value_expression (a b : ℝ) (h : a^2 + b^2 = 3 + a*b) :
  (∃ x y : ℝ, x^2 + y^2 = 3 + x*y ∧ (2*x - 3*y)^2 + (x + 2*y)*(x - 2*y) ≤ 22) ∧
  (∃ x y : ℝ, x^2 + y^2 = 3 + x*y ∧ (2*x - 3*y)^2 + (x + 2*y)*(x - 2*y) = 22) :=
by sorry

end max_value_expression_l1199_119902


namespace x_range_l1199_119964

theorem x_range (x : ℝ) : 
  (|x - 1| + |x - 5| = 4) ↔ (1 ≤ x ∧ x ≤ 5) := by
sorry

end x_range_l1199_119964


namespace problem_statement_l1199_119936

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = -2) : 
  (x - 2*y)^y = 1/121 := by sorry

end problem_statement_l1199_119936


namespace kaleb_shirts_removed_l1199_119919

/-- The number of shirts Kaleb got rid of -/
def shirts_removed (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Kaleb got rid of 7 shirts -/
theorem kaleb_shirts_removed :
  let initial_shirts : ℕ := 17
  let remaining_shirts : ℕ := 10
  shirts_removed initial_shirts remaining_shirts = 7 := by
  sorry

end kaleb_shirts_removed_l1199_119919


namespace cat_food_insufficient_l1199_119965

theorem cat_food_insufficient (B S : ℝ) 
  (h1 : B > S) 
  (h2 : B < 2 * S) : 
  4 * B + 4 * S < 3 * (B + 2 * S) := by
sorry

end cat_food_insufficient_l1199_119965


namespace f_satisfies_points_l1199_119929

/-- The relation between x and y --/
def f (x : ℝ) : ℝ := 200 - 15 * x - 15 * x^2

/-- The set of points that the function should satisfy --/
def points : List (ℝ × ℝ) := [(0, 200), (1, 170), (2, 120), (3, 50), (4, 0)]

/-- Theorem stating that the function satisfies all given points --/
theorem f_satisfies_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

#check f_satisfies_points

end f_satisfies_points_l1199_119929


namespace integer_product_100000_l1199_119977

theorem integer_product_100000 : ∃ (a b : ℤ), 
  a * b = 100000 ∧ 
  a % 10 ≠ 0 ∧ 
  b % 10 ≠ 0 ∧ 
  (a = 32 ∨ b = 32) :=
by sorry

end integer_product_100000_l1199_119977


namespace complex_number_simplification_l1199_119901

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  (i^3) / (1 - i) = (1 / 2 : ℂ) - (1 / 2 : ℂ) * i := by sorry

end complex_number_simplification_l1199_119901


namespace triangle_inequality_l1199_119966

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a < b + c) (hbc : b < a + c) (hca : c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end triangle_inequality_l1199_119966


namespace nina_travel_period_l1199_119922

/-- Nina's travel pattern over two months -/
def two_month_distance : ℕ := 400 + 800

/-- Total distance Nina wants to travel -/
def total_distance : ℕ := 14400

/-- Number of two-month periods needed to reach the total distance -/
def num_two_month_periods : ℕ := total_distance / two_month_distance

/-- Duration of Nina's travel period in months -/
def travel_period_months : ℕ := num_two_month_periods * 2

/-- Theorem stating that Nina's travel period is 24 months -/
theorem nina_travel_period :
  travel_period_months = 24 :=
sorry

end nina_travel_period_l1199_119922


namespace cost_price_calculation_l1199_119970

/-- 
Given a product with:
- Marked price of 1100 yuan
- Sold at 80% of the marked price
- Makes a 10% profit

Prove that the cost price is 800 yuan
-/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : marked_price = 1100)
  (h2 : discount_rate = 0.8)
  (h3 : profit_rate = 0.1) :
  marked_price * discount_rate = (1 + profit_rate) * 800 := by
  sorry

end cost_price_calculation_l1199_119970


namespace game_points_difference_l1199_119955

theorem game_points_difference (layla_points nahima_points total_points : ℕ) : 
  layla_points = 70 → total_points = 112 → layla_points + nahima_points = total_points →
  layla_points - nahima_points = 28 := by
sorry

end game_points_difference_l1199_119955


namespace intersection_and_solution_set_l1199_119967

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the solution set of x^2 + ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -1 ∨ x > 2}

theorem intersection_and_solution_set :
  (A ∩ B = A_intersect_B) ∧
  (∀ a b : ℝ, ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) →
              ({x : ℝ | x^2 + a*x - b < 0} = solution_set a b)) :=
by sorry

end intersection_and_solution_set_l1199_119967


namespace trigonometric_equation_solution_l1199_119942

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (∀ K : ℤ, x ≠ π * K / 3) →
  (cos x)^2 = (sin (2 * x))^2 + cos (3 * x) / sin (3 * x) →
  (∃ n : ℤ, x = π / 2 + π * n) ∨ (∃ k : ℤ, x = π / 6 + π * k) ∨ (∃ k : ℤ, x = -π / 6 + π * k) :=
by sorry

end trigonometric_equation_solution_l1199_119942


namespace rectangle_max_area_l1199_119904

/-- A rectangle with integer sides and perimeter 80 has a maximum area of 400. -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * (l + w) = 80 →
  ∀ l' w' : ℕ,
  l' > 0 → w' > 0 →
  2 * (l' + w') = 80 →
  l * w ≤ 400 :=
by sorry

end rectangle_max_area_l1199_119904


namespace cake_area_increase_percentage_cake_area_increase_percentage_approx_l1199_119974

/-- The percent increase in area of a circular cake when its diameter increases from 8 inches to 10 inches -/
theorem cake_area_increase_percentage : ℝ := by
  -- Define the initial and final diameters
  let initial_diameter : ℝ := 8
  let final_diameter : ℝ := 10
  
  -- Define the function to calculate the area of a circular cake given its diameter
  let cake_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2
  
  -- Calculate the initial and final areas
  let initial_area := cake_area initial_diameter
  let final_area := cake_area final_diameter
  
  -- Calculate the percent increase
  let percent_increase := (final_area - initial_area) / initial_area * 100
  
  -- Prove that the percent increase is 56.25%
  sorry

/-- The result of cake_area_increase_percentage is approximately 56.25 -/
theorem cake_area_increase_percentage_approx :
  |cake_area_increase_percentage - 56.25| < 0.01 := by sorry

end cake_area_increase_percentage_cake_area_increase_percentage_approx_l1199_119974


namespace pauline_total_spend_l1199_119935

/-- The total amount Pauline will spend, including sales tax -/
def total_amount (pre_tax_amount : ℝ) (tax_rate : ℝ) : ℝ :=
  pre_tax_amount * (1 + tax_rate)

/-- Proof that Pauline will spend $162 on all items, including sales tax -/
theorem pauline_total_spend :
  total_amount 150 0.08 = 162 := by
  sorry

end pauline_total_spend_l1199_119935


namespace triangle_area_theorem_l1199_119933

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circles
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem triangle_area_theorem (ABC : Triangle) (circle1 circle2 : Circle) 
  (L K M N : ℝ × ℝ) :
  -- Given conditions
  circle1.radius = 1/18 →
  circle2.radius = 2/9 →
  (ABC.A.1 - L.1)^2 + (ABC.A.2 - L.2)^2 = (1/9)^2 →
  (ABC.C.1 - M.1)^2 + (ABC.C.2 - M.2)^2 = (1/6)^2 →
  -- Circle1 touches AB at L and AC at K
  ((ABC.A.1 - L.1)^2 + (ABC.A.2 - L.2)^2 = circle1.radius^2 ∧
   (ABC.B.1 - L.1)^2 + (ABC.B.2 - L.2)^2 = circle1.radius^2) →
  ((ABC.A.1 - K.1)^2 + (ABC.A.2 - K.2)^2 = circle1.radius^2 ∧
   (ABC.C.1 - K.1)^2 + (ABC.C.2 - K.2)^2 = circle1.radius^2) →
  -- Circle2 touches AC at N and BC at M
  ((ABC.A.1 - N.1)^2 + (ABC.A.2 - N.2)^2 = circle2.radius^2 ∧
   (ABC.C.1 - N.1)^2 + (ABC.C.2 - N.2)^2 = circle2.radius^2) →
  ((ABC.B.1 - M.1)^2 + (ABC.B.2 - M.2)^2 = circle2.radius^2 ∧
   (ABC.C.1 - M.1)^2 + (ABC.C.2 - M.2)^2 = circle2.radius^2) →
  -- Circles touch each other
  (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2 →
  -- Conclusion: Area of triangle ABC is 15/11
  abs ((ABC.B.1 - ABC.A.1) * (ABC.C.2 - ABC.A.2) - (ABC.C.1 - ABC.A.1) * (ABC.B.2 - ABC.A.2)) / 2 = 15/11 :=
by sorry


end triangle_area_theorem_l1199_119933


namespace pond_soil_volume_l1199_119923

/-- The volume of soil extracted from a rectangular pond -/
def soil_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of soil extracted from a rectangular pond
    with dimensions 20 m × 15 m × 5 m is 1500 cubic meters -/
theorem pond_soil_volume :
  soil_volume 20 15 5 = 1500 := by
  sorry

end pond_soil_volume_l1199_119923


namespace angle_range_for_point_in_first_quadrant_l1199_119930

def is_in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem angle_range_for_point_in_first_quadrant (α : ℝ) :
  0 ≤ α ∧ α ≤ 2 * Real.pi →
  is_in_first_quadrant (Real.tan α) (Real.sin α - Real.cos α) →
  (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2)) ∨ (α ∈ Set.Ioo Real.pi (5 * Real.pi / 4)) :=
by sorry

end angle_range_for_point_in_first_quadrant_l1199_119930


namespace honey_savings_l1199_119916

/-- Calculates the savings given daily earnings, number of days worked, and total spent -/
def calculate_savings (daily_earnings : ℕ) (days_worked : ℕ) (total_spent : ℕ) : ℕ :=
  daily_earnings * days_worked - total_spent

/-- Proves that given the problem conditions, Honey's savings are $240 -/
theorem honey_savings :
  let daily_earnings : ℕ := 80
  let days_worked : ℕ := 20
  let total_spent : ℕ := 1360
  calculate_savings daily_earnings days_worked total_spent = 240 := by
sorry

#eval calculate_savings 80 20 1360

end honey_savings_l1199_119916


namespace parallelogram_area_l1199_119953

/-- The area of a parallelogram with base 12 and height 6 is 72 -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 12 → 
  height = 6 → 
  area = base * height → 
  area = 72 := by sorry

end parallelogram_area_l1199_119953


namespace rain_probability_l1199_119995

theorem rain_probability (weihai_rain : Real) (zibo_rain : Real) (both_rain : Real) :
  weihai_rain = 0.2 →
  zibo_rain = 0.15 →
  both_rain = 0.06 →
  both_rain / weihai_rain = 0.3 :=
by sorry

end rain_probability_l1199_119995


namespace cos_alpha_plus_pi_sixth_l1199_119934

theorem cos_alpha_plus_pi_sixth (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : (Real.cos (2 * α)) / (1 + Real.tan α ^ 2) = 3 / 8) : 
  Real.cos (α + Real.pi / 6) = 1 / 2 := by
sorry

end cos_alpha_plus_pi_sixth_l1199_119934


namespace parallel_vectors_sum_l1199_119952

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Vector addition -/
def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → vec_add a b = (6, 3) := by
  sorry

end parallel_vectors_sum_l1199_119952


namespace sphere_in_cylinder_ratio_l1199_119905

theorem sphere_in_cylinder_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (π * r^2 * h = 3 * (4/3 * π * r^3)) → (h / (2 * r) = 2) := by
  sorry

end sphere_in_cylinder_ratio_l1199_119905


namespace radical_equation_solution_l1199_119958

theorem radical_equation_solution (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) :
  (∀ N : ℝ, N ≠ 1 → N^((1 : ℝ)/a + 1/(a*b) + 2/(a*b*c)) = N^(17/24)) →
  b = 4 := by
sorry

end radical_equation_solution_l1199_119958


namespace max_value_abcd_l1199_119939

theorem max_value_abcd (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1) :
  2 * a * b * Real.sqrt 2 + 2 * b * c + 2 * c * d ≤ 1 ∧ 
  ∃ a' b' c' d', 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 0 ≤ d' ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 = 1 ∧
    2 * a' * b' * Real.sqrt 2 + 2 * b' * c' + 2 * c' * d' = 1 :=
by sorry

end max_value_abcd_l1199_119939


namespace line_not_in_second_quadrant_iff_a_ge_two_l1199_119954

/-- A line that does not pass through the second quadrant -/
structure LineNotInSecondQuadrant where
  a : ℝ
  not_in_second_quadrant : ∀ (x y : ℝ), (a - 2) * y = (3 * a - 1) * x - 4 → ¬(x < 0 ∧ y > 0)

/-- The range of values for a when the line does not pass through the second quadrant -/
theorem line_not_in_second_quadrant_iff_a_ge_two (l : LineNotInSecondQuadrant) :
  l.a ∈ Set.Ici 2 ↔ ∀ (x y : ℝ), (l.a - 2) * y = (3 * l.a - 1) * x - 4 → ¬(x < 0 ∧ y > 0) := by
  sorry

end line_not_in_second_quadrant_iff_a_ge_two_l1199_119954


namespace negation_of_exists_square_nonpositive_l1199_119910

theorem negation_of_exists_square_nonpositive :
  (¬ ∃ a : ℝ, a^2 ≤ 0) ↔ (∀ a : ℝ, a^2 > 0) := by sorry

end negation_of_exists_square_nonpositive_l1199_119910


namespace unripe_oranges_calculation_l1199_119968

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := 28

/-- The number of days of harvest -/
def harvest_days : ℕ := 26

/-- The total number of sacks of oranges after the harvest period -/
def total_oranges : ℕ := 2080

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := (total_oranges - ripe_oranges_per_day * harvest_days) / harvest_days

theorem unripe_oranges_calculation :
  unripe_oranges_per_day = 52 :=
by sorry

end unripe_oranges_calculation_l1199_119968


namespace hamburger_count_l1199_119938

theorem hamburger_count (served left_over : ℕ) (h1 : served = 3) (h2 : left_over = 6) :
  served + left_over = 9 := by
  sorry

end hamburger_count_l1199_119938


namespace rebate_percentage_l1199_119917

theorem rebate_percentage (num_pairs : ℕ) (price_per_pair : ℚ) (total_rebate : ℚ) :
  num_pairs = 5 →
  price_per_pair = 28 →
  total_rebate = 14 →
  (total_rebate / (num_pairs * price_per_pair)) * 100 = 10 := by
  sorry

end rebate_percentage_l1199_119917


namespace work_completion_time_l1199_119996

/-- The time taken to complete a work when three workers with given efficiencies work together -/
theorem work_completion_time 
  (total_work : ℝ) 
  (efficiency_x efficiency_y efficiency_z : ℝ) 
  (hx : efficiency_x = 1 / 20)
  (hy : efficiency_y = 3 / 80)
  (hz : efficiency_z = 3 / 40)
  (h_total : total_work = 1) :
  (total_work / (efficiency_x + efficiency_y + efficiency_z)) = 80 / 13 := by
  sorry

end work_completion_time_l1199_119996
