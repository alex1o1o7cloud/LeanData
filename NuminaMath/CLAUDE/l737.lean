import Mathlib

namespace dans_initial_green_marbles_count_l737_73758

def dans_initial_green_marbles : ℕ := sorry

def mikes_taken_marbles : ℕ := 23

def dans_remaining_green_marbles : ℕ := 9

theorem dans_initial_green_marbles_count : 
  dans_initial_green_marbles = dans_remaining_green_marbles + mikes_taken_marbles := by
  sorry

end dans_initial_green_marbles_count_l737_73758


namespace common_tangents_exist_l737_73796

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a line is a common tangent to two circles -/
def isCommonTangent (l : Line) (c1 c2 : Circle) : Prop := 
  isTangent l c1 ∧ isTangent l c2

/-- The line connecting the centers of two circles -/
def centerLine (c1 c2 : Circle) : Line := sorry

/-- Checks if a line intersects another line -/
def intersects (l1 l2 : Line) : Prop := sorry

/-- Theorem: For any two circles, there exist common tangents in two cases -/
theorem common_tangents_exist (c1 c2 : Circle) : 
  ∃ (l1 l2 : Line), 
    (isCommonTangent l1 c1 c2 ∧ ¬intersects l1 (centerLine c1 c2)) ∧
    (isCommonTangent l2 c1 c2 ∧ intersects l2 (centerLine c1 c2)) := by
  sorry

end common_tangents_exist_l737_73796


namespace equation_system_solution_l737_73717

theorem equation_system_solution (a b : ℝ) :
  (∃ (a' : ℝ), a' * 1 + 4 * (-1) = 23 ∧ 3 * 1 - b * (-1) = 5) →
  (∃ (b' : ℝ), a * 7 + 4 * (-3) = 23 ∧ 3 * 7 - b' * (-3) = 5) →
  (a^2 - 2*a*b + b^2 = 9) ∧
  (a * 3 + 4 * 2 = 23 ∧ 3 * 3 - b * 2 = 5) := by
sorry

end equation_system_solution_l737_73717


namespace average_age_problem_l737_73709

theorem average_age_problem (a b c : ℕ) : 
  (a + c) / 2 = 29 →
  b = 17 →
  (a + b + c) / 3 = 25 := by
sorry

end average_age_problem_l737_73709


namespace x_minus_y_squared_l737_73718

theorem x_minus_y_squared (x y : ℝ) : 
  y = Real.sqrt (2 * x - 3) + Real.sqrt (3 - 2 * x) - 4 →
  x = 3 / 2 →
  x - y^2 = -29 / 2 := by
sorry

end x_minus_y_squared_l737_73718


namespace binary_sum_equals_116_l737_73706

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The binary representation of 11111₂ -/
def binary2 : List Bool := [true, true, true, true, true]

/-- Theorem stating that the sum of 1010101₂ and 11111₂ in decimal is 116 -/
theorem binary_sum_equals_116 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 116 := by
  sorry


end binary_sum_equals_116_l737_73706


namespace polynomial_divisibility_l737_73746

def P (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, ∃ q, P a b c x = (x - 1)^3 * q) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
sorry

end polynomial_divisibility_l737_73746


namespace percentage_equality_l737_73781

theorem percentage_equality (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (20 / 100) * (x + y) →
  y = (50 / 100) * x →
  P = 60 := by
sorry

end percentage_equality_l737_73781


namespace hyperbola_and_parabola_properties_l737_73733

-- Define the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the parabola
def parabola_equation (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_and_parabola_properties :
  -- Length of real axis
  (∃ a : ℝ, a = 3 ∧ 2 * a = 6) ∧
  -- Length of imaginary axis
  (∃ b : ℝ, b = 4 ∧ 2 * b = 8) ∧
  -- Eccentricity
  (∃ e : ℝ, e = 5 / 3) ∧
  -- Parabola equation
  (∀ x y : ℝ, hyperbola_equation x y →
    (x = 0 ∧ y = 0 → parabola_equation x y) ∧
    (x = -3 ∧ y = 0 → parabola_equation x y)) :=
by sorry

end hyperbola_and_parabola_properties_l737_73733


namespace dragon_population_l737_73741

theorem dragon_population (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 117) 
  (h2 : total_legs = 108) : 
  ∃ (three_headed six_headed : ℕ), 
    three_headed = 15 ∧ 
    six_headed = 12 ∧ 
    3 * three_headed + 6 * six_headed = total_heads ∧ 
    4 * (three_headed + six_headed) = total_legs :=
by sorry

end dragon_population_l737_73741


namespace alien_sequence_valid_l737_73734

/-- Represents a symbol in the alien sequence -/
inductive AlienSymbol
| percent
| exclamation
| ampersand
| plus
| zero

/-- Represents the possible operations -/
inductive Operation
| addition
| subtraction
| multiplication
| division
| exponentiation

/-- Represents a mapping of symbols to digits or operations -/
structure SymbolMapping where
  base : ℕ
  digit_map : AlienSymbol → Fin base
  operation : AlienSymbol → Option Operation
  equality : AlienSymbol

/-- Converts a list of alien symbols to a natural number given a symbol mapping -/
def alien_to_nat (mapping : SymbolMapping) (symbols : List AlienSymbol) : ℕ := sorry

/-- Checks if a list of alien symbols represents a valid equation given a symbol mapping -/
def is_valid_equation (mapping : SymbolMapping) (symbols : List AlienSymbol) : Prop := sorry

/-- The alien sequence -/
def alien_sequence : List AlienSymbol :=
  [AlienSymbol.percent, AlienSymbol.exclamation, AlienSymbol.ampersand,
   AlienSymbol.plus, AlienSymbol.exclamation, AlienSymbol.zero,
   AlienSymbol.plus, AlienSymbol.plus, AlienSymbol.exclamation,
   AlienSymbol.exclamation, AlienSymbol.exclamation]

theorem alien_sequence_valid :
  ∃ (mapping : SymbolMapping), is_valid_equation mapping alien_sequence := by
  sorry

#check alien_sequence_valid

end alien_sequence_valid_l737_73734


namespace at_least_one_greater_than_one_l737_73755

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) : max x y > 1 := by
  sorry

end at_least_one_greater_than_one_l737_73755


namespace last_triangle_perimeter_l737_73754

/-- Represents a triangle in the sequence -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { a := t.a / 2 - 1,
    b := t.b / 2,
    c := t.c / 2 + 1 }

/-- Checks if a triangle is valid (satisfies triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The initial triangle T₁ -/
def T₁ : Triangle :=
  { a := 1009, b := 1010, c := 1011 }

/-- Generates the sequence of triangles -/
def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- Finds the index of the last valid triangle in the sequence -/
def lastValidTriangleIndex : ℕ := sorry

/-- The last valid triangle in the sequence -/
def lastValidTriangle : Triangle :=
  triangleSequence lastValidTriangleIndex

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℚ :=
  t.a + t.b + t.c

theorem last_triangle_perimeter :
  perimeter lastValidTriangle = 71 / 8 := by sorry

end last_triangle_perimeter_l737_73754


namespace history_not_statistics_l737_73713

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 90 →
  history = 36 →
  statistics = 30 →
  history_or_statistics = 59 →
  history - (history + statistics - history_or_statistics) = 29 := by
  sorry

end history_not_statistics_l737_73713


namespace sufficient_not_necessary_condition_l737_73728

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < Real.pi / 2 → (x - 1) * Real.tan x > 0) ∧
  (∃ x : ℝ, (x - 1) * Real.tan x > 0 ∧ ¬(1 < x ∧ x < Real.pi / 2)) := by
  sorry

end sufficient_not_necessary_condition_l737_73728


namespace complex_product_zero_l737_73766

theorem complex_product_zero (z : ℂ) (h : z^2 + 1 = 0) :
  (z^4 + Complex.I) * (z^4 - Complex.I) = 0 := by
sorry

end complex_product_zero_l737_73766


namespace cat_food_sale_l737_73788

theorem cat_food_sale (total_customers : Nat) (first_group : Nat) (middle_group : Nat) (last_group : Nat)
  (first_group_cases : Nat) (last_group_cases : Nat) (total_cases : Nat)
  (h1 : total_customers = first_group + middle_group + last_group)
  (h2 : total_customers = 20)
  (h3 : first_group = 8)
  (h4 : middle_group = 4)
  (h5 : last_group = 8)
  (h6 : first_group_cases = 3)
  (h7 : last_group_cases = 1)
  (h8 : total_cases = 40)
  (h9 : total_cases = first_group * first_group_cases + middle_group * x + last_group * last_group_cases)
  : x = 2 := by
  sorry

#check cat_food_sale

end cat_food_sale_l737_73788


namespace find_r_l737_73762

theorem find_r (m : ℝ) (r : ℝ) 
  (h1 : 5 = m * 3^r) 
  (h2 : 45 = m * 9^(2*r)) : 
  r = 2/3 := by
  sorry

end find_r_l737_73762


namespace composition_equality_l737_73792

theorem composition_equality (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 5 * x + 6) (h2 : ∀ x, φ x = 7 * x + 4) :
  (∀ x, δ (φ x) = 1) ↔ (∀ x, x = -5/7) :=
by sorry

end composition_equality_l737_73792


namespace parallel_implies_t_half_magnitude_when_t_one_l737_73738

-- Define the vectors a and b as functions of t
def a (t : ℝ) : Fin 2 → ℝ := ![2 - t, 3]
def b (t : ℝ) : Fin 2 → ℝ := ![t, 1]

-- Theorem 1: If a and b are parallel, then t = 1/2
theorem parallel_implies_t_half :
  ∀ t : ℝ, (∃ k : ℝ, a t = k • b t) → t = 1/2 := by sorry

-- Theorem 2: When t = 1, |a - 4b| = √10
theorem magnitude_when_t_one :
  ‖(a 1) - 4 • (b 1)‖ = Real.sqrt 10 := by sorry

end parallel_implies_t_half_magnitude_when_t_one_l737_73738


namespace candies_per_block_l737_73782

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) : 
  candies_per_house = 7 → houses_per_block = 5 → candies_per_house * houses_per_block = 35 :=
by
  sorry

end candies_per_block_l737_73782


namespace multiplicative_inverse_154_mod_257_l737_73739

theorem multiplicative_inverse_154_mod_257 : ∃ x : ℕ, x < 257 ∧ (154 * x) % 257 = 1 :=
  by
    use 20
    sorry

end multiplicative_inverse_154_mod_257_l737_73739


namespace absolute_value_equals_negative_l737_73730

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end absolute_value_equals_negative_l737_73730


namespace multiplicative_inverse_modulo_l737_73769

def A : ℕ := 123456
def B : ℕ := 142857
def M : ℕ := 1000009
def N : ℕ := 750298

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 :=
sorry

end multiplicative_inverse_modulo_l737_73769


namespace fourth_power_sum_equals_108_to_fourth_l737_73753

theorem fourth_power_sum_equals_108_to_fourth : ∃ m : ℕ+, 
  97^4 + 84^4 + 27^4 + 3^4 = m^4 ∧ m = 108 := by
  sorry

end fourth_power_sum_equals_108_to_fourth_l737_73753


namespace bryden_payment_is_correct_l737_73737

/-- The face value of a state quarter in dollars -/
def quarter_value : ℝ := 0.25

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 6

/-- The percentage of face value the collector offers, expressed as a decimal -/
def collector_offer_percentage : ℝ := 16

/-- The discount percentage applied to the total payment, expressed as a decimal -/
def discount_percentage : ℝ := 0.1

/-- The amount Bryden receives for his state quarters -/
def bryden_payment : ℝ :=
  (bryden_quarters : ℝ) * quarter_value * collector_offer_percentage * (1 - discount_percentage)

theorem bryden_payment_is_correct :
  bryden_payment = 21.6 := by sorry

end bryden_payment_is_correct_l737_73737


namespace trigonometric_identity_l737_73735

theorem trigonometric_identity (α : Real) : 
  (Real.sin (45 * π / 180 + α))^2 - (Real.sin (30 * π / 180 - α))^2 - 
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180 + 2 * α) = 
  Real.sin (2 * α) := by
  sorry

end trigonometric_identity_l737_73735


namespace number_1349_is_valid_l737_73757

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 100 % 10 = 3 * (n / 1000)) ∧
  (n % 10 = 3 * (n / 100 % 10))

theorem number_1349_is_valid : is_valid_number 1349 := by
  sorry

end number_1349_is_valid_l737_73757


namespace roberto_outfits_l737_73708

/-- Represents the number of different outfits Roberto can create --/
def number_of_outfits (trousers shirts jackets constrained_trousers constrained_jackets : ℕ) : ℕ :=
  ((trousers - constrained_trousers) * jackets + constrained_trousers * constrained_jackets) * shirts

/-- Theorem stating the number of outfits Roberto can create given his wardrobe constraints --/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 7
  let jackets : ℕ := 3
  let constrained_trousers : ℕ := 2
  let constrained_jackets : ℕ := 2
  number_of_outfits trousers shirts jackets constrained_trousers constrained_jackets = 91 :=
by
  sorry


end roberto_outfits_l737_73708


namespace stone_slab_area_l737_73722

/-- Given 50 square stone slabs with a length of 120 cm each, 
    the total floor area covered is 72 square meters. -/
theorem stone_slab_area (n : ℕ) (length_cm : ℝ) (total_area_m2 : ℝ) : 
  n = 50 → 
  length_cm = 120 → 
  total_area_m2 = (n * (length_cm / 100)^2) → 
  total_area_m2 = 72 := by
sorry

end stone_slab_area_l737_73722


namespace fourth_student_in_sample_l737_73779

def systematic_sample (total_students : ℕ) (sample_size : ℕ) (sample : Finset ℕ) : Prop :=
  sample.card = sample_size ∧
  ∃ k : ℕ, ∀ i ∈ sample, ∃ j : ℕ, i = 1 + j * (total_students / sample_size)

theorem fourth_student_in_sample 
  (total_students : ℕ) (sample_size : ℕ) (sample : Finset ℕ) 
  (h1 : total_students = 52)
  (h2 : sample_size = 4)
  (h3 : 3 ∈ sample)
  (h4 : 29 ∈ sample)
  (h5 : 42 ∈ sample)
  (h6 : systematic_sample total_students sample_size sample) :
  16 ∈ sample :=
sorry

end fourth_student_in_sample_l737_73779


namespace expression_value_l737_73789

theorem expression_value (x y : ℝ) (h : x^2 - 4*x - 1 = 0) :
  (2*x - 3)^2 - (x + y)*(x - y) - y^2 = 12 := by
  sorry

end expression_value_l737_73789


namespace count_true_props_l737_73700

def original_prop : Prop := ∀ x : ℝ, x^2 > 1 → x > 1

def converse_prop : Prop := ∀ x : ℝ, x > 1 → x^2 > 1

def inverse_prop : Prop := ∀ x : ℝ, x^2 ≤ 1 → x ≤ 1

def contrapositive_prop : Prop := ∀ x : ℝ, x ≤ 1 → x^2 ≤ 1

theorem count_true_props :
  (converse_prop ∧ inverse_prop ∧ ¬contrapositive_prop) ∨
  (converse_prop ∧ ¬inverse_prop ∧ contrapositive_prop) ∨
  (¬converse_prop ∧ inverse_prop ∧ contrapositive_prop) :=
sorry

end count_true_props_l737_73700


namespace pie_crust_flour_calculation_l737_73771

theorem pie_crust_flour_calculation (total_flour : ℚ) (original_crusts new_crusts : ℕ) :
  total_flour > 0 →
  original_crusts > 0 →
  new_crusts > 0 →
  (total_flour / original_crusts) * new_crusts = total_flour →
  total_flour / new_crusts = 1 / 5 := by
  sorry

#check pie_crust_flour_calculation (5 : ℚ) 40 25

end pie_crust_flour_calculation_l737_73771


namespace square_less_than_triple_l737_73743

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end square_less_than_triple_l737_73743


namespace arithmetic_sequence_difference_l737_73773

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the common difference of the specific arithmetic sequence -/
theorem arithmetic_sequence_difference (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) (h2015 : a 2015 = a 2013 + 6) : 
    ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

end arithmetic_sequence_difference_l737_73773


namespace gcd_factorial_problem_l737_73716

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : 
  Nat.gcd (factorial 7) ((factorial 10) / (factorial 4)) = 2520 := by
  sorry

end gcd_factorial_problem_l737_73716


namespace shortest_distance_to_circle_l737_73715

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 24*x + y^2 + 10*y + 160 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 10

theorem shortest_distance_to_circle :
  ∀ p : ℝ × ℝ, circle_equation p.1 p.2 →
  ∃ q : ℝ × ℝ, circle_equation q.1 q.2 ∧
  ∀ r : ℝ × ℝ, circle_equation r.1 r.2 →
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ≤ Real.sqrt ((r.1 - 0)^2 + (r.2 - 0)^2) ∧
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = shortest_distance := by
  sorry


end shortest_distance_to_circle_l737_73715


namespace problem_proof_l737_73776

theorem problem_proof : (1 / (2 - Real.sqrt 3)) - 1 - 2 * (Real.sqrt 3 / 2) = 1 :=
by sorry

end problem_proof_l737_73776


namespace rods_in_mile_l737_73752

/-- Represents the number of furlongs in a mile -/
def furlongs_per_mile : ℕ := 10

/-- Represents the number of rods in a furlong -/
def rods_per_furlong : ℕ := 50

/-- Theorem stating that one mile is equal to 500 rods -/
theorem rods_in_mile : furlongs_per_mile * rods_per_furlong = 500 := by
  sorry

end rods_in_mile_l737_73752


namespace replacement_preserves_mean_and_variance_l737_73742

def initial_set : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
def new_set : List ℤ := [-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5]

def mean (s : List ℤ) : ℚ := (s.sum : ℚ) / s.length

def variance (s : List ℤ) : ℚ :=
  let m := mean s
  (s.map (λ x => ((x : ℚ) - m) ^ 2)).sum / s.length

theorem replacement_preserves_mean_and_variance :
  mean initial_set = mean new_set ∧ variance initial_set = variance new_set :=
sorry

end replacement_preserves_mean_and_variance_l737_73742


namespace fifth_term_of_8998_sequence_l737_73759

-- Define the sequence generation function
def generateSequence (n : ℕ) : List ℕ :=
  -- Implementation of the sequence generation rules
  sorry

-- Define a function to get the nth term of a sequence
def getNthTerm (sequence : List ℕ) (n : ℕ) : ℕ :=
  -- Implementation to get the nth term
  sorry

-- Theorem statement
theorem fifth_term_of_8998_sequence :
  getNthTerm (generateSequence 8998) 5 = 4625 :=
sorry

end fifth_term_of_8998_sequence_l737_73759


namespace parallelogram_area_l737_73702

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 12) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 60 * Real.sqrt 3 :=
sorry

end parallelogram_area_l737_73702


namespace opposite_of_2023_l737_73790

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l737_73790


namespace bicycle_distance_l737_73747

/-- Proves that a bicycle traveling 1/2 as fast as a motorcycle moving at 40 miles per hour
    will cover a distance of 10 miles in 30 minutes. -/
theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time : ℝ) :
  motorcycle_speed = 40 →
  bicycle_speed_ratio = (1 : ℝ) / 2 →
  time = 30 / 60 →
  (bicycle_speed_ratio * motorcycle_speed) * time = 10 := by
  sorry

end bicycle_distance_l737_73747


namespace cherry_pie_degrees_l737_73799

theorem cherry_pie_degrees (total : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total = 40)
  (h2 : chocolate = 15)
  (h3 : apple = 10)
  (h4 : blueberry = 7)
  (h5 : (total - (chocolate + apple + blueberry)) % 2 = 0) :
  let remaining := total - (chocolate + apple + blueberry)
  let cherry := remaining / 2
  (cherry : ℚ) / total * 360 = 36 := by
  sorry

end cherry_pie_degrees_l737_73799


namespace stereo_trade_in_value_l737_73768

theorem stereo_trade_in_value (old_cost new_cost discount_percent out_of_pocket : ℚ) 
  (h1 : old_cost = 250)
  (h2 : new_cost = 600)
  (h3 : discount_percent = 25)
  (h4 : out_of_pocket = 250) :
  let discounted_price := new_cost * (1 - discount_percent / 100)
  let trade_in_value := discounted_price - out_of_pocket
  trade_in_value / old_cost * 100 = 80 := by
sorry

end stereo_trade_in_value_l737_73768


namespace mangoes_per_neighbor_l737_73711

-- Define the given conditions
def total_mangoes : ℕ := 560
def mangoes_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Define the relationship between x and total mangoes
def mangoes_sold (total : ℕ) : ℕ := total / 2

-- Theorem statement
theorem mangoes_per_neighbor : 
  (total_mangoes - mangoes_sold total_mangoes - mangoes_to_family) / num_neighbors = 19 := by
  sorry

end mangoes_per_neighbor_l737_73711


namespace unique_b_c_solution_l737_73749

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem unique_b_c_solution :
  ∃! (b c : ℝ), 
    (∃ a : ℝ, A a ≠ B b c) ∧ 
    (∃ a : ℝ, A a ∪ B b c = {-3, 4}) ∧
    (∃ a : ℝ, A a ∩ B b c = {-3}) ∧
    b = 6 ∧ c = 9 := by
  sorry


end unique_b_c_solution_l737_73749


namespace tangent_directrix_parabola_circle_l737_73760

/-- Given a circle and a parabola with a tangent directrix, prove the value of m -/
theorem tangent_directrix_parabola_circle (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), x^2 + y^2 = 1/4) →
  (∃ (x y : ℝ), y = m * x^2) →
  (∃ (d : ℝ), d = 1/(4*m) ∧ d = 1/2) →
  m = 1/2 := by
sorry

end tangent_directrix_parabola_circle_l737_73760


namespace mandarin_ducks_count_l737_73744

/-- The number of pairs of mandarin ducks -/
def num_pairs : ℕ := 3

/-- The number of ducks in each pair -/
def ducks_per_pair : ℕ := 2

/-- The total number of mandarin ducks -/
def total_ducks : ℕ := num_pairs * ducks_per_pair

theorem mandarin_ducks_count : total_ducks = 6 := by
  sorry

end mandarin_ducks_count_l737_73744


namespace thumbtack_solution_l737_73720

/-- Represents the problem of calculating remaining thumbtacks --/
structure ThumbTackProblem where
  total_cans : Nat
  total_tacks : Nat
  boards_tested : Nat
  tacks_per_board : Nat

/-- Calculates the number of remaining thumbtacks in each can --/
def remaining_tacks (problem : ThumbTackProblem) : Nat :=
  (problem.total_tacks / problem.total_cans) - (problem.boards_tested * problem.tacks_per_board)

/-- Theorem stating the solution to the specific problem --/
theorem thumbtack_solution :
  let problem : ThumbTackProblem := {
    total_cans := 3,
    total_tacks := 450,
    boards_tested := 120,
    tacks_per_board := 1
  }
  remaining_tacks problem = 30 := by sorry


end thumbtack_solution_l737_73720


namespace weight_difference_l737_73774

/-- Given the weights of Mildred and Carol, prove the difference in their weights. -/
theorem weight_difference (mildred_weight carol_weight : ℕ) 
  (h1 : mildred_weight = 59) 
  (h2 : carol_weight = 9) : 
  mildred_weight - carol_weight = 50 := by
  sorry

end weight_difference_l737_73774


namespace f_even_implies_a_zero_f_min_value_when_a_zero_f_never_odd_l737_73783

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- Part I: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
sorry

-- Part II: When a = 0, the minimum value of f is 1
theorem f_min_value_when_a_zero :
  ∀ x, f 0 x ≥ 1 :=
sorry

-- Part III: f can never be an odd function for any real a
theorem f_never_odd (a : ℝ) :
  ¬(∀ x, f a x = -(f a (-x))) :=
sorry

end f_even_implies_a_zero_f_min_value_when_a_zero_f_never_odd_l737_73783


namespace total_eyes_in_pond_l737_73731

/-- The number of eyes an animal has -/
def eyes_per_animal : ℕ := 2

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 6

/-- The total number of animals in the pond -/
def total_animals : ℕ := num_frogs + num_crocodiles

/-- Theorem: The total number of animal eyes in the pond is 52 -/
theorem total_eyes_in_pond : num_frogs * eyes_per_animal + num_crocodiles * eyes_per_animal = 52 := by
  sorry

end total_eyes_in_pond_l737_73731


namespace solution_set_theorem_l737_73748

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f_def (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2*x

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_def : f_def f) : 
  {x : ℝ | f (x + 1) < 3} = Set.Ioo (-4 : ℝ) 2 := by
sorry

end solution_set_theorem_l737_73748


namespace birds_and_storks_l737_73770

theorem birds_and_storks (initial_birds : ℕ) (initial_storks : ℕ) (joining_storks : ℕ) :
  initial_birds = 6 →
  initial_storks = 3 →
  joining_storks = 2 →
  initial_birds - (initial_storks + joining_storks) = 1 :=
by
  sorry

end birds_and_storks_l737_73770


namespace polynomial_remainder_theorem_l737_73732

theorem polynomial_remainder_theorem (x : ℝ) :
  let p (x : ℝ) := x^4 - 4*x^2 + 7
  let r := p 3
  r = 52 := by sorry

end polynomial_remainder_theorem_l737_73732


namespace unique_solution_cube_root_equation_l737_73775

def f (x : ℝ) := (20 * x + (20 * x + 13) ^ (1/3)) ^ (1/3)

theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 13 ∧ x = 546/5 := by sorry

end unique_solution_cube_root_equation_l737_73775


namespace sum_inequality_l737_73786

theorem sum_inequality (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end sum_inequality_l737_73786


namespace consecutive_odd_product_l737_73780

theorem consecutive_odd_product (m : ℕ) (N : ℤ) : 
  Odd N → 
  N = (m - 1) * m * (m + 1) - ((m - 1) + m + (m + 1)) → 
  (∃ k : ℕ, m = 2 * k + 1) ∧ 
  N = (m - 2) * m * (m + 2) ∧ 
  Odd (m - 2) ∧ Odd m ∧ Odd (m + 2) :=
sorry

end consecutive_odd_product_l737_73780


namespace company_employees_l737_73729

theorem company_employees (december_employees : ℕ) (increase_percentage : ℚ) :
  december_employees = 470 →
  increase_percentage = 15 / 100 →
  ∃ (january_employees : ℕ),
    (january_employees : ℚ) * (1 + increase_percentage) = december_employees ∧
    january_employees = 409 :=
by sorry

end company_employees_l737_73729


namespace cricket_match_playtime_l737_73763

-- Define the total duration of the match in minutes
def total_duration : ℕ := 12 * 60 + 35

-- Define the lunch break duration in minutes
def lunch_break : ℕ := 15

-- Theorem to prove the actual playtime
theorem cricket_match_playtime :
  total_duration - lunch_break = 740 := by
  sorry

end cricket_match_playtime_l737_73763


namespace inequality_solution_set_l737_73703

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end inequality_solution_set_l737_73703


namespace part_one_part_two_l737_73778

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 2} := by sorry

-- Part II
theorem part_two : 
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) → -7 < a ∧ a < -1 := by sorry

end part_one_part_two_l737_73778


namespace solve_system_l737_73740

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 3 + 1 / x) :
  y = 3 / 2 + Real.sqrt 15 / 2 := by
  sorry

end solve_system_l737_73740


namespace r_squared_equals_one_for_linear_plot_l737_73795

/-- A scatter plot where all points lie on a straight line -/
structure LinearScatterPlot where
  /-- The slope of the line on which all points lie -/
  slope : ℝ
  /-- All points in the scatter plot lie on a straight line -/
  all_points_on_line : Bool

/-- The coefficient of determination (R²) for a scatter plot -/
def r_squared (plot : LinearScatterPlot) : ℝ :=
  sorry

/-- Theorem: If all points in a scatter plot lie on a straight line with a slope of 2,
    then R² equals 1 -/
theorem r_squared_equals_one_for_linear_plot (plot : LinearScatterPlot)
    (h1 : plot.slope = 2)
    (h2 : plot.all_points_on_line = true) :
    r_squared plot = 1 := by
  sorry

end r_squared_equals_one_for_linear_plot_l737_73795


namespace tech_club_enrollment_l737_73751

theorem tech_club_enrollment (total : ℕ) (cs : ℕ) (electronics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 90)
  (h3 : electronics = 60)
  (h4 : both = 20) :
  total - (cs + electronics - both) = 20 := by
  sorry

end tech_club_enrollment_l737_73751


namespace pages_copied_example_l737_73727

/-- Given a cost per page in cents, a flat service charge in cents, and a total budget in cents,
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (service_charge : ℕ) (total_budget : ℕ) : ℕ :=
  (total_budget - service_charge) / cost_per_page

/-- Prove that with a cost of 3 cents per page, a flat service charge of 500 cents,
    and a total budget of 5000 cents, the maximum number of pages that can be copied is 1500. -/
theorem pages_copied_example : max_pages_copied 3 500 5000 = 1500 := by
  sorry

end pages_copied_example_l737_73727


namespace line_point_k_value_l737_73791

/-- A line contains the points (3,5), (1,k), and (7,9). Prove that k = 3. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), m * 3 + b = 5 ∧ m * 1 + b = k ∧ m * 7 + b = 9) → k = 3 := by
  sorry

end line_point_k_value_l737_73791


namespace race_track_width_l737_73785

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 880 →
  outer_radius = 165.0563499208679 →
  ∃ width : ℝ, (abs (width - 25.049) < 0.001 ∧
    width = outer_radius - inner_circumference / (2 * Real.pi)) :=
by sorry

end race_track_width_l737_73785


namespace all_trinomials_no_roots_l737_73745

/-- Represents a quadratic trinomial ax² + bx + c -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Calculates the discriminant of a quadratic trinomial -/
def discriminant (q : QuadraticTrinomial) : ℤ :=
  q.b ^ 2 - 4 * q.a * q.c

/-- Checks if a quadratic trinomial has no real roots -/
def has_no_real_roots (q : QuadraticTrinomial) : Prop :=
  discriminant q < 0

/-- Creates all permutations of three coefficients -/
def all_permutations (a b c : ℤ) : List QuadraticTrinomial :=
  [
    ⟨a, b, c⟩, ⟨a, c, b⟩,
    ⟨b, a, c⟩, ⟨b, c, a⟩,
    ⟨c, a, b⟩, ⟨c, b, a⟩
  ]

theorem all_trinomials_no_roots
  (a b c : ℤ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  ∀ q ∈ all_permutations a b c, has_no_real_roots q :=
sorry

end all_trinomials_no_roots_l737_73745


namespace gum_purchase_cost_l737_73764

/-- Calculates the total cost in dollars for buying gum with a discount -/
def total_cost_with_discount (price_per_piece : ℚ) (num_pieces : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_cost_cents := price_per_piece * num_pieces
  let discount_amount := discount_rate * total_cost_cents
  let final_cost_cents := total_cost_cents - discount_amount
  final_cost_cents / 100

/-- Theorem: The total cost of buying 1500 pieces of gum at 2 cents each with a 10% discount is $27 -/
theorem gum_purchase_cost :
  total_cost_with_discount 2 1500 (10/100) = 27 := by
  sorry


end gum_purchase_cost_l737_73764


namespace zhang_apple_sales_l737_73784

/-- Represents the number of apples Zhang needs to sell to earn a specific profit -/
def apples_to_sell (buy_price : ℚ) (sell_price : ℚ) (target_profit : ℚ) : ℚ :=
  target_profit / (sell_price - buy_price)

/-- Theorem stating the number of apples Zhang needs to sell to earn 15 yuan -/
theorem zhang_apple_sales : 
  let buy_price : ℚ := 1 / 4  -- 4 apples for 1 yuan
  let sell_price : ℚ := 2 / 5 -- 5 apples for 2 yuan
  let target_profit : ℚ := 15
  apples_to_sell buy_price sell_price target_profit = 100 :=
by
  sorry

#eval apples_to_sell (1/4) (2/5) 15

end zhang_apple_sales_l737_73784


namespace goals_theorem_l737_73724

def goals_problem (bruce_goals michael_goals jack_goals sarah_goals : ℕ) : Prop :=
  bruce_goals = 4 ∧
  michael_goals = 2 * bruce_goals ∧
  jack_goals = bruce_goals - 1 ∧
  sarah_goals = jack_goals / 2 ∧
  michael_goals + jack_goals + sarah_goals = 12

theorem goals_theorem :
  ∃ (bruce_goals michael_goals jack_goals sarah_goals : ℕ),
    goals_problem bruce_goals michael_goals jack_goals sarah_goals :=
by
  sorry

end goals_theorem_l737_73724


namespace expression_simplification_l737_73725

theorem expression_simplification (x : ℝ) : 
  3*x - 4*(2 + x^2) + 5*(3 - x) - 6*(1 - 2*x + x^2) = 10*x - 10*x^2 + 1 := by
  sorry

end expression_simplification_l737_73725


namespace unique_solution_for_digit_sum_equation_l737_73798

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that 402 is the only solution to n(S(n) - 1) = 2010 -/
theorem unique_solution_for_digit_sum_equation :
  ∀ n : ℕ, n > 0 → (n * (S n - 1) = 2010) ↔ n = 402 := by sorry

end unique_solution_for_digit_sum_equation_l737_73798


namespace seven_faced_prism_has_five_lateral_faces_l737_73756

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  total_faces : ℕ
  base_faces : ℕ := 2

/-- Define a function that calculates the number of lateral faces of a prism. -/
def lateral_faces (p : Prism) : ℕ :=
  p.total_faces - p.base_faces

/-- Theorem stating that a prism with 7 faces has 5 lateral faces. -/
theorem seven_faced_prism_has_five_lateral_faces (p : Prism) (h : p.total_faces = 7) :
  lateral_faces p = 5 := by
  sorry


end seven_faced_prism_has_five_lateral_faces_l737_73756


namespace pollen_diameter_scientific_notation_l737_73777

/-- Expresses a given number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem pollen_diameter_scientific_notation :
  scientific_notation 0.0000021 = (2.1, -6) :=
sorry

end pollen_diameter_scientific_notation_l737_73777


namespace fraction_of_states_1790_to_1799_l737_73794

theorem fraction_of_states_1790_to_1799 (total_states : ℕ) (states_1790_to_1799 : ℕ) : 
  total_states = 30 → states_1790_to_1799 = 9 → (states_1790_to_1799 : ℚ) / total_states = 3 / 10 := by
  sorry

end fraction_of_states_1790_to_1799_l737_73794


namespace inverse_of_three_mod_forty_l737_73701

theorem inverse_of_three_mod_forty :
  ∃ x : ℕ, x < 40 ∧ (3 * x) % 40 = 1 :=
by
  use 27
  sorry

end inverse_of_three_mod_forty_l737_73701


namespace base6_divisibility_by_19_l737_73765

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6 + d

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem base6_divisibility_by_19 (y : ℕ) (h : y < 6) :
  isDivisibleBy19 (base6ToDecimal 2 5 y 3) ↔ y = 2 := by sorry

end base6_divisibility_by_19_l737_73765


namespace quadratic_equation_roots_l737_73736

/-- Given a quadratic equation of the form (x^2 - bx + b^2) / (ax^2 - c) = (m-1) / (m+1),
    if the roots are numerically equal but of opposite signs, and c = b^2,
    then m = (a-1) / (a+1) -/
theorem quadratic_equation_roots (a b m : ℝ) :
  (∃ x y : ℝ, x = -y ∧ x ≠ 0 ∧
    (x^2 - b*x + b^2) / (a*x^2 - b^2) = (m-1) / (m+1) ∧
    (y^2 - b*y + b^2) / (a*y^2 - b^2) = (m-1) / (m+1)) →
  m = (a-1) / (a+1) := by
sorry

end quadratic_equation_roots_l737_73736


namespace first_neighbor_height_l737_73793

/-- The height of Lucille's house in feet -/
def lucille_height : ℝ := 80

/-- The height of the second neighbor's house in feet -/
def neighbor2_height : ℝ := 99

/-- The height difference between Lucille's house and the average height in feet -/
def height_difference : ℝ := 3

/-- The height of the first neighbor's house in feet -/
def neighbor1_height : ℝ := 70

theorem first_neighbor_height :
  (lucille_height + neighbor1_height + neighbor2_height) / 3 - height_difference = lucille_height :=
by sorry

end first_neighbor_height_l737_73793


namespace smallest_x_power_inequality_l737_73723

theorem smallest_x_power_inequality : 
  ∃ x : ℕ, (∀ y : ℕ, 27^y > 3^24 → x ≤ y) ∧ 27^x > 3^24 :=
by
  -- The proof goes here
  sorry

end smallest_x_power_inequality_l737_73723


namespace sum_of_binary_digits_315_l737_73767

theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end sum_of_binary_digits_315_l737_73767


namespace chairs_to_remove_l737_73750

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 15

/-- The initial number of chairs set up -/
def initial_chairs : ℕ := 225

/-- The number of expected attendees -/
def expected_attendees : ℕ := 180

/-- Theorem: The number of chairs to be removed is 45 -/
theorem chairs_to_remove :
  ∃ (removed : ℕ),
    removed = initial_chairs - expected_attendees ∧
    removed % chairs_per_row = 0 ∧
    (initial_chairs - removed) ≥ expected_attendees ∧
    (initial_chairs - removed) % chairs_per_row = 0 ∧
    removed = 45 := by
  sorry

end chairs_to_remove_l737_73750


namespace middle_number_proof_l737_73719

theorem middle_number_proof (x y z : ℕ) : 
  x < y ∧ y < z ∧ 
  x + y = 22 ∧ 
  x + z = 29 ∧ 
  y + z = 31 ∧ 
  x = 10 → 
  y = 12 := by
sorry

end middle_number_proof_l737_73719


namespace train_stoppage_time_l737_73761

/-- Calculates the stoppage time per hour for a train given its speeds with and without stoppages. -/
theorem train_stoppage_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48) 
  (h2 : speed_with_stops = 40) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

end train_stoppage_time_l737_73761


namespace halfway_between_one_eighth_and_one_third_l737_73704

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end halfway_between_one_eighth_and_one_third_l737_73704


namespace absolute_value_simplification_l737_73707

theorem absolute_value_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := by
  sorry

end absolute_value_simplification_l737_73707


namespace square_root_of_nine_l737_73721

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end square_root_of_nine_l737_73721


namespace jumping_jacks_ratio_l737_73797

/-- The ratio of Brooke's jumping jacks to Sidney's jumping jacks is 3:1 -/
theorem jumping_jacks_ratio : 
  let sidney_jj := [20, 36, 40, 50]
  let brooke_jj := 438
  (brooke_jj : ℚ) / (sidney_jj.sum : ℚ) = 3 / 1 := by sorry

end jumping_jacks_ratio_l737_73797


namespace parabola_theorem_l737_73710

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Definition of the parabola E: x^2 = 4y -/
def parabolaE : Parabola := ⟨4, by norm_num⟩

/-- Focus of the parabola -/
def focusF : Point := ⟨0, 1⟩

/-- Origin point -/
def originO : Point := ⟨0, 0⟩

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Function to check if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Theorem statement -/
theorem parabola_theorem (l : Line) (A B : Point) 
  (h1 : A.x^2 = 4 * A.y) -- A is on the parabola
  (h2 : B.x^2 = 4 * B.y) -- B is on the parabola
  (h3 : focusF.y = l.m * focusF.x + l.b) -- l passes through F
  (h4 : A.y = l.m * A.x + l.b) -- A is on l
  (h5 : B.y = l.m * B.x + l.b) -- B is on l
  :
  (∃ (minArea : ℝ), minArea = 2 ∧ 
    ∀ (A' B' : Point), A'.x^2 = 4 * A'.y → B'.x^2 = 4 * B'.y → 
    A'.y = l.m * A'.x + l.b → B'.y = l.m * B'.x + l.b →
    triangleArea originO A' B' ≥ minArea) ∧ 
  (∃ (C : Point) (lAO lBC : Line), 
    C.y = -1 ∧ -- C is on the directrix
    A.y = lAO.m * A.x + lAO.b ∧ -- AO line
    originO.y = lAO.m * originO.x + lAO.b ∧
    C.y = lAO.m * C.x + lAO.b ∧
    B.x = C.x ∧ -- BC is vertical
    isParallel lBC ⟨0, 1⟩) -- BC is parallel to y-axis
  := by sorry

end parabola_theorem_l737_73710


namespace smallest_mersenne_prime_above_30_l737_73714

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem smallest_mersenne_prime_above_30 :
  ∃ p : ℕ, mersenne_prime p ∧ p > 30 ∧ 
  ∀ q : ℕ, mersenne_prime q → q > 30 → p ≤ q :=
sorry

end smallest_mersenne_prime_above_30_l737_73714


namespace fraction_value_implies_m_l737_73705

theorem fraction_value_implies_m (m : ℚ) : (m - 5) / m = 2 → m = -5 := by
  sorry

end fraction_value_implies_m_l737_73705


namespace basketball_game_difference_l737_73772

/-- Given a ratio of boys to girls and the number of girls, 
    calculate the difference between the number of boys and girls -/
def boys_girls_difference (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : ℕ :=
  let num_boys := (num_girls / girls_ratio) * boys_ratio
  num_boys - num_girls

/-- Theorem stating that with a ratio of 8:5 boys to girls and 30 girls, 
    there are 18 more boys than girls -/
theorem basketball_game_difference : boys_girls_difference 8 5 30 = 18 := by
  sorry

end basketball_game_difference_l737_73772


namespace log_inequality_characterization_l737_73787

theorem log_inequality_characterization (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha_neq_1 : a ≠ 1) :
  (Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)) ↔
  (b = 1 ∧ a ≠ 1) ∨ (a > b ∧ b > 1) ∨ (b > 1 ∧ 1 > a) := by
  sorry

end log_inequality_characterization_l737_73787


namespace coffee_break_probabilities_l737_73712

/-- Represents the state of knowledge among scientists -/
structure ScientistGroup where
  total : Nat
  initial_knowers : Nat
  
/-- Represents the outcome after the coffee break -/
structure CoffeeBreakOutcome where
  final_knowers : Nat

/-- Probability of a specific outcome after the coffee break -/
def probability_of_outcome (group : ScientistGroup) (outcome : CoffeeBreakOutcome) : ℚ :=
  sorry

/-- Expected number of scientists who know the news after the coffee break -/
def expected_final_knowers (group : ScientistGroup) : ℚ :=
  sorry

theorem coffee_break_probabilities (group : ScientistGroup) 
  (h1 : group.total = 18) 
  (h2 : group.initial_knowers = 10) : 
  probability_of_outcome group ⟨13⟩ = 0 ∧ 
  probability_of_outcome group ⟨14⟩ = 1120 / 2431 ∧
  expected_final_knowers group = 14 + 12 / 17 :=
  sorry

end coffee_break_probabilities_l737_73712


namespace simplify_and_evaluate_l737_73726

theorem simplify_and_evaluate (a b : ℝ) : 
  a = 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180) →
  b = 3 →
  1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3 := by
  sorry

end simplify_and_evaluate_l737_73726
