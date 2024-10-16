import Mathlib

namespace NUMINAMATH_CALUDE_value_of_y_l1061_106188

theorem value_of_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1061_106188


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1061_106157

theorem factorization_of_quadratic (a : ℝ) : a^2 - 2*a - 15 = (a + 3) * (a - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1061_106157


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l1061_106191

theorem exactly_one_correct_probability
  (prob_A prob_B : ℝ)
  (h_A : prob_A = 0.8)
  (h_B : prob_B = 0.7)
  (h_independent : True)  -- Representing independence
  : prob_A * (1 - prob_B) + prob_B * (1 - prob_A) = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l1061_106191


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1061_106165

/-- The complex number z defined as 1 + 2i + i^3 -/
def z : ℂ := 1 + 2 * Complex.I + Complex.I ^ 3

/-- Theorem stating that the magnitude of z is √2 -/
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1061_106165


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1061_106107

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 5005
  let y : ℝ := 15 + Real.sqrt 5005
  x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1061_106107


namespace NUMINAMATH_CALUDE_gp_common_ratio_l1061_106182

/-- 
Theorem: In a geometric progression where the ratio of the sum of the first 6 terms 
to the sum of the first 3 terms is 217, the common ratio is 6.
-/
theorem gp_common_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l1061_106182


namespace NUMINAMATH_CALUDE_u_less_than_v_l1061_106190

theorem u_less_than_v (u v : ℝ) 
  (hu : (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10*u^9 = 8)
  (hv : (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10*v^11 = 8) :
  u < v := by
sorry

end NUMINAMATH_CALUDE_u_less_than_v_l1061_106190


namespace NUMINAMATH_CALUDE_pencil_exchange_coloring_l1061_106117

-- Define a permutation as a bijective function from ℕ to ℕ
def Permutation (n : ℕ) := {f : ℕ → ℕ // Function.Bijective f ∧ ∀ i, i ≥ n → f i = i}

-- Define a coloring as a function from ℕ to a three-element type
def Coloring (n : ℕ) := ℕ → Fin 3

-- The main theorem
theorem pencil_exchange_coloring (n : ℕ) (p : Permutation n) :
  ∃ c : Coloring n, ∀ i < n, c i ≠ c (p.val i) :=
sorry

end NUMINAMATH_CALUDE_pencil_exchange_coloring_l1061_106117


namespace NUMINAMATH_CALUDE_valid_outfit_choices_eq_239_l1061_106142

/-- Represents the number of valid outfit choices given the specified conditions -/
def valid_outfit_choices : ℕ := by
  -- Define the number of shirts, pants, and hats
  let num_shirts : ℕ := 6
  let num_pants : ℕ := 7
  let num_hats : ℕ := 6
  
  -- Define the number of colors
  let num_colors : ℕ := 6
  
  -- Calculate total number of outfits without restrictions
  let total_outfits : ℕ := num_shirts * num_pants * num_hats
  
  -- Calculate number of outfits with all items the same color
  let all_same_color : ℕ := num_colors
  
  -- Calculate number of outfits with shirt and pants the same color
  let shirt_pants_same : ℕ := num_colors + 1
  
  -- Calculate the number of valid outfits
  exact total_outfits - all_same_color - shirt_pants_same

/-- Theorem stating that the number of valid outfit choices is 239 -/
theorem valid_outfit_choices_eq_239 : valid_outfit_choices = 239 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_eq_239_l1061_106142


namespace NUMINAMATH_CALUDE_probability_play_one_instrument_l1061_106150

/-- Given a population with the following properties:
  * The total population is 10000
  * One-third of the population plays at least one instrument
  * 450 people play two or more instruments
  This theorem states that the probability of a randomly selected person
  playing exactly one instrument is 0.2883 -/
theorem probability_play_one_instrument (total_population : ℕ)
  (plays_at_least_one : ℕ) (plays_two_or_more : ℕ) :
  total_population = 10000 →
  plays_at_least_one = total_population / 3 →
  plays_two_or_more = 450 →
  (plays_at_least_one - plays_two_or_more : ℚ) / total_population = 2883 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_probability_play_one_instrument_l1061_106150


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4851_l1061_106167

theorem largest_prime_factor_of_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 4851 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4851_l1061_106167


namespace NUMINAMATH_CALUDE_largest_number_with_property_l1061_106192

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def property (n : Nat) : Prop :=
  n % sum_of_digits n = 0

theorem largest_number_with_property :
  ∃ (n : Nat), n < 900 ∧ property n ∧ ∀ (m : Nat), m < 900 → property m → m ≤ n :=
by
  use 888
  sorry

#eval sum_of_digits 888  -- Should output 24
#eval 888 % 24           -- Should output 0

end NUMINAMATH_CALUDE_largest_number_with_property_l1061_106192


namespace NUMINAMATH_CALUDE_system_solution_l1061_106138

theorem system_solution :
  ∃! (x y : ℚ), (2 * x + 3 * y = (7 - 2 * x) + (7 - 3 * y)) ∧
                 (3 * x - 2 * y = (x - 2) + (y - 2)) ∧
                 x = 3 / 4 ∧ y = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1061_106138


namespace NUMINAMATH_CALUDE_division_simplification_l1061_106155

theorem division_simplification : 180 / (12 + 15 * 3) = 180 / 57 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1061_106155


namespace NUMINAMATH_CALUDE_max_fourth_term_arithmetic_sequence_l1061_106166

theorem max_fourth_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (∀ k : Fin 5, 0 < a + k * d) →
  (5 * a + 10 * d = 75) →
  (∀ a' d' : ℕ, (∀ k : Fin 5, 0 < a' + k * d') → (5 * a' + 10 * d' = 75) → a + 3 * d ≥ a' + 3 * d') →
  a + 3 * d = 22 := by
sorry

end NUMINAMATH_CALUDE_max_fourth_term_arithmetic_sequence_l1061_106166


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1061_106113

/-- Two lines ax + 2y = 3 and x + (a-1)y = 1 are parallel -/
def are_parallel (a : ℝ) : Prop :=
  a = 3 ∨ a = -1

/-- a = 2 is a sufficient condition for parallelism -/
def is_sufficient : Prop :=
  ∀ a : ℝ, a = 2 → are_parallel a

/-- a = 2 is a necessary condition for parallelism -/
def is_necessary : Prop :=
  ∀ a : ℝ, are_parallel a → a = 2

theorem not_sufficient_nor_necessary : ¬is_sufficient ∧ ¬is_necessary :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1061_106113


namespace NUMINAMATH_CALUDE_fraction_difference_l1061_106173

theorem fraction_difference : 
  let a := 2^2 + 4^2 + 6^2
  let b := 1^2 + 3^2 + 5^2
  (a / b) - (b / a) = 1911 / 1960 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l1061_106173


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1061_106177

theorem complex_modulus_problem (w z : ℂ) : 
  w * z = 18 - 24 * I ∧ Complex.abs w = 3 * Real.sqrt 13 → 
  Complex.abs z = 10 / Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1061_106177


namespace NUMINAMATH_CALUDE_kiras_breakfast_time_l1061_106172

/-- The time it takes Kira to make breakfast given the number of sausages, eggs, and cooking times -/
def breakfast_time (num_sausages : ℕ) (num_eggs : ℕ) (sausage_time : ℕ) (egg_time : ℕ) : ℕ :=
  num_sausages * sausage_time + num_eggs * egg_time

/-- Theorem stating that Kira's breakfast time is 39 minutes -/
theorem kiras_breakfast_time :
  breakfast_time 3 6 5 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kiras_breakfast_time_l1061_106172


namespace NUMINAMATH_CALUDE_room_length_proof_l1061_106141

/-- Given a room with dimensions L * 15 * 12 feet, prove that L = 25 feet
    based on the whitewashing cost and room features. -/
theorem room_length_proof (L : ℝ) : 
  L * 15 * 12 > 0 →  -- room has positive volume
  (3 : ℝ) * (2 * (L * 12 + 15 * 12) - (6 * 3 + 3 * 4 * 3)) = 2718 →
  L = 25 := by sorry

end NUMINAMATH_CALUDE_room_length_proof_l1061_106141


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1061_106147

theorem complex_number_modulus : 
  let z : ℂ := 2 / (1 + Complex.I) + (1 - Complex.I)^2
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1061_106147


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1061_106149

theorem polynomial_division_remainder : ∃ (Q : Polynomial ℤ) (R : Polynomial ℤ),
  (X : Polynomial ℤ)^50 = (X^2 - 5*X + 6) * Q + R ∧
  (Polynomial.degree R < 2) ∧
  R = (3^50 - 2^50) * X + (2^50 - 2 * 3^50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1061_106149


namespace NUMINAMATH_CALUDE_books_per_shelf_l1061_106118

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 504) (h2 : num_shelves = 9) :
  total_books / num_shelves = 56 := by
sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1061_106118


namespace NUMINAMATH_CALUDE_units_digit_of_199_factorial_l1061_106189

theorem units_digit_of_199_factorial (n : ℕ) : n = 199 → (n.factorial % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_199_factorial_l1061_106189


namespace NUMINAMATH_CALUDE_fabric_usage_period_l1061_106127

/-- The number of shirts Jenson makes per day -/
def shirts_per_day : ℕ := 3

/-- The number of pants Kingsley makes per day -/
def pants_per_day : ℕ := 5

/-- The amount of fabric used for one shirt (in yards) -/
def fabric_per_shirt : ℕ := 2

/-- The amount of fabric used for one pair of pants (in yards) -/
def fabric_per_pants : ℕ := 5

/-- The total amount of fabric needed (in yards) -/
def total_fabric_needed : ℕ := 93

/-- Theorem: The number of days needed to use the total fabric is 3 -/
theorem fabric_usage_period : 
  (shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pants) * 3 = total_fabric_needed :=
by sorry

end NUMINAMATH_CALUDE_fabric_usage_period_l1061_106127


namespace NUMINAMATH_CALUDE_last_four_digits_of_2_to_1965_l1061_106164

theorem last_four_digits_of_2_to_1965 : 2^1965 % 10000 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_2_to_1965_l1061_106164


namespace NUMINAMATH_CALUDE_even_expressions_l1061_106160

theorem even_expressions (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (k₁ k₂ k₃ : ℕ),
    (m - n)^2 = 2 * k₁ ∧
    (m - n - 4)^2 = 2 * k₂ ∧
    2 * m * n + 4 = 2 * k₃ := by
  sorry

end NUMINAMATH_CALUDE_even_expressions_l1061_106160


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1061_106114

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) ↔ (∀ x : ℝ, |x - 2| + |x - 4| > 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1061_106114


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1061_106146

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 2*z = 0 →
  x*z / (y^2) = 10 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1061_106146


namespace NUMINAMATH_CALUDE_area_ratio_of_angle_bisector_l1061_106158

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the lengths of the sides
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the angle bisector
def is_angle_bisector (X P Y Z : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_ratio_of_angle_bisector (XYZ : Triangle) (P : ℝ × ℝ) :
  side_length XYZ.X XYZ.Y = 20 →
  side_length XYZ.X XYZ.Z = 30 →
  side_length XYZ.Y XYZ.Z = 26 →
  is_angle_bisector XYZ.X P XYZ.Y XYZ.Z →
  (triangle_area XYZ.X XYZ.Y P) / (triangle_area XYZ.X XYZ.Z P) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_angle_bisector_l1061_106158


namespace NUMINAMATH_CALUDE_optimal_price_for_profit_l1061_106183

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 400

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 10) * sales_volume x

-- State the theorem
theorem optimal_price_for_profit :
  ∃ x : ℝ, 
    x > 0 ∧ 
    profit x = 2160 ∧ 
    ∀ y : ℝ, y > 0 ∧ profit y = 2160 → sales_volume x ≤ sales_volume y := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_for_profit_l1061_106183


namespace NUMINAMATH_CALUDE_new_person_weight_l1061_106143

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1061_106143


namespace NUMINAMATH_CALUDE_distinct_integer_parts_l1061_106132

theorem distinct_integer_parts (N : ℕ) (h : N > 1) :
  {α : ℝ | (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i * α⌋ ≠ ⌊j * α⌋) ∧
           (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i / α⌋ ≠ ⌊j / α⌋)} =
  {α : ℝ | (N - 1) / N ≤ α ∧ α ≤ N / (N - 1)} :=
sorry

end NUMINAMATH_CALUDE_distinct_integer_parts_l1061_106132


namespace NUMINAMATH_CALUDE_fly_distance_l1061_106124

/-- The distance flown by a fly between two approaching pedestrians -/
theorem fly_distance (d : ℝ) (v_ped : ℝ) (v_fly : ℝ) (h1 : d > 0) (h2 : v_ped > 0) (h3 : v_fly > 0) :
  let t := d / (2 * v_ped)
  v_fly * t = v_fly * d / (2 * v_ped) := by sorry

#check fly_distance

end NUMINAMATH_CALUDE_fly_distance_l1061_106124


namespace NUMINAMATH_CALUDE_inessa_is_cleverest_l1061_106168

-- Define the foxes
inductive Fox : Type
  | Alisa : Fox
  | Larisa : Fox
  | Inessa : Fox

-- Define a relation for "is cleverer than"
def is_cleverer_than : Fox → Fox → Prop := sorry

-- Define a property for being the cleverest
def is_cleverest : Fox → Prop := sorry

-- Define a function to check if a fox is telling the truth
def tells_truth : Fox → Prop := sorry

-- State the theorem
theorem inessa_is_cleverest :
  -- The cleverest fox lies, others tell the truth
  (∀ f : Fox, is_cleverest f ↔ ¬(tells_truth f)) →
  -- Larisa's statement
  (tells_truth Fox.Larisa ↔ ¬(is_cleverest Fox.Alisa)) →
  -- Alisa's statement
  (tells_truth Fox.Alisa ↔ is_cleverer_than Fox.Alisa Fox.Larisa) →
  -- Inessa's statement
  (tells_truth Fox.Inessa ↔ is_cleverer_than Fox.Alisa Fox.Inessa) →
  -- There is exactly one cleverest fox
  (∃! f : Fox, is_cleverest f) →
  -- Conclusion: Inessa is the cleverest
  is_cleverest Fox.Inessa :=
by
  sorry

end NUMINAMATH_CALUDE_inessa_is_cleverest_l1061_106168


namespace NUMINAMATH_CALUDE_fifty_men_left_l1061_106140

/-- Represents the scenario of a hostel with changing occupancy and food provisions. -/
structure Hostel where
  initialMen : ℕ
  initialDays : ℕ
  finalDays : ℕ

/-- Calculates the number of men who left the hostel based on the change in provision duration. -/
def menWhoLeft (h : Hostel) : ℕ :=
  h.initialMen - (h.initialMen * h.initialDays) / h.finalDays

/-- Theorem stating that in the given hostel scenario, 50 men left. -/
theorem fifty_men_left (h : Hostel)
  (h_initial_men : h.initialMen = 250)
  (h_initial_days : h.initialDays = 36)
  (h_final_days : h.finalDays = 45) :
  menWhoLeft h = 50 := by
  sorry

end NUMINAMATH_CALUDE_fifty_men_left_l1061_106140


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1061_106119

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    prove that if a₂ * a₃ = 2a₁ and the arithmetic mean of (1/2)a₄ and a₇ is 5/8,
    then the sum of the first 4 terms (S₄) is 30. -/
theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : a₁ * q * (a₁ * q^2) = 2 * a₁)
    (h2 : (1/2 * a₁ * q^3 + a₁ * q^6) / 2 = 5/8) :
  a₁ * (1 - q^4) / (1 - q) = 30 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1061_106119


namespace NUMINAMATH_CALUDE_prob_odd_total_is_221_441_l1061_106196

/-- Represents a standard die with one dot removed randomly -/
structure ModifiedDie :=
  (remaining_dots : Fin 21)

/-- The probability of a modified die showing an odd number of dots on top -/
def prob_odd_top (d : ModifiedDie) : ℚ := 11 / 21

/-- The probability of a modified die showing an even number of dots on top -/
def prob_even_top (d : ModifiedDie) : ℚ := 10 / 21

/-- The probability of two modified dice showing an odd total number of dots on top when rolled simultaneously -/
def prob_odd_total (d1 d2 : ModifiedDie) : ℚ :=
  (prob_odd_top d1 * prob_odd_top d2) + (prob_even_top d1 * prob_even_top d2)

theorem prob_odd_total_is_221_441 (d1 d2 : ModifiedDie) :
  prob_odd_total d1 d2 = 221 / 441 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_total_is_221_441_l1061_106196


namespace NUMINAMATH_CALUDE_forward_journey_time_l1061_106136

/-- Represents the journey of a car -/
structure Journey where
  distance : ℝ
  forwardTime : ℝ
  returnTime : ℝ
  speedIncrease : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem forward_journey_time (j : Journey)
  (h1 : j.distance = 210)
  (h2 : j.returnTime = 5)
  (h3 : j.speedIncrease = 12)
  (h4 : j.distance = j.distance / j.forwardTime * j.returnTime + j.speedIncrease * j.returnTime) :
  j.forwardTime = 7 := by
  sorry

end NUMINAMATH_CALUDE_forward_journey_time_l1061_106136


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1061_106184

/-
  Define the hyperbola equation
-/
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-
  Define the asymptote equation
-/
def has_asymptote (x y : ℝ) : Prop :=
  y = 2 * x

/-
  Define the parabola equation
-/
def is_parabola (x y : ℝ) : Prop :=
  y^2 = 20 * x

/-
  State the theorem
-/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, is_hyperbola a b x y ∧ has_asymptote x y) →
  (∃ x y : ℝ, is_parabola x y ∧ 
    ((x - 5)^2 + y^2 = a^2 + b^2 ∨ (x + 5)^2 + y^2 = a^2 + b^2)) →
  a^2 = 5 ∧ b^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1061_106184


namespace NUMINAMATH_CALUDE_mike_work_hours_l1061_106187

theorem mike_work_hours : 
  let wash_time : ℕ := 10  -- minutes to wash a car
  let oil_change_time : ℕ := 15  -- minutes to change oil
  let tire_change_time : ℕ := 30  -- minutes to change a set of tires
  let cars_washed : ℕ := 9  -- number of cars Mike washed
  let oil_changes : ℕ := 6  -- number of oil changes Mike performed
  let tire_changes : ℕ := 2  -- number of tire sets Mike changed
  
  let total_minutes : ℕ := 
    wash_time * cars_washed + 
    oil_change_time * oil_changes + 
    tire_change_time * tire_changes
  
  let total_hours : ℕ := total_minutes / 60

  total_hours = 4 := by sorry

end NUMINAMATH_CALUDE_mike_work_hours_l1061_106187


namespace NUMINAMATH_CALUDE_x_value_l1061_106123

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 3, 9}

theorem x_value (x : ℕ) (h1 : x ∈ A) (h2 : x ∉ B) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1061_106123


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l1061_106154

theorem relationship_between_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l1061_106154


namespace NUMINAMATH_CALUDE_jellybean_ratio_l1061_106109

/-- Given the number of jellybeans for Tino, Lee, and Arnold, prove the ratio of Arnold's to Lee's jellybeans --/
theorem jellybean_ratio 
  (tino lee arnold : ℕ) 
  (h1 : tino = lee + 24) 
  (h2 : tino = 34) 
  (h3 : arnold = 5) : 
  arnold / lee = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l1061_106109


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_is_thirteen_l1061_106194

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSame (machine : GumballMachine) : ℕ := 13

/-- Theorem stating that for the given gumball machine configuration, 
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_is_thirteen (machine : GumballMachine) 
  (h1 : machine.red = 12)
  (h2 : machine.white = 10)
  (h3 : machine.blue = 9)
  (h4 : machine.green = 8) : 
  minGumballsForFourSame machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_is_thirteen_l1061_106194


namespace NUMINAMATH_CALUDE_danas_class_size_l1061_106116

/-- Proves that the total number of students in Dana's senior high school class is 200. -/
theorem danas_class_size :
  ∀ (total_students : ℕ),
  (total_students : ℝ) * 0.6 * 0.5 * 0.5 = 30 →
  total_students = 200 := by
  sorry

end NUMINAMATH_CALUDE_danas_class_size_l1061_106116


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l1061_106197

theorem max_rabbits_with_traits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both_traits : ℕ) :
  (long_ears = 13 ∧ jump_far = 17 ∧ both_traits ≥ 3) →
  (∀ n : ℕ, n > N → ∃ arrangement : ℕ × ℕ × ℕ, 
    arrangement.1 + arrangement.2.1 + arrangement.2.2 = n ∧
    arrangement.1 + arrangement.2.1 = long_ears ∧
    arrangement.1 + arrangement.2.2 = jump_far ∧
    arrangement.1 < both_traits) →
  N = 27 :=
sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l1061_106197


namespace NUMINAMATH_CALUDE_first_solution_carbonated_water_percentage_l1061_106181

/-- Represents a solution with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_100 : lemonade + carbonated_water = 100

/-- Proves that the first solution is 80% carbonated water given the conditions -/
theorem first_solution_carbonated_water_percentage
  (solution1 : Solution)
  (solution2 : Solution)
  (h1 : solution1.lemonade = 20)
  (h2 : solution2.lemonade = 45)
  (h3 : solution2.carbonated_water = 55)
  (h_mixture : 0.5 * solution1.carbonated_water + 0.5 * solution2.carbonated_water = 67.5) :
  solution1.carbonated_water = 80 := by
  sorry

#check first_solution_carbonated_water_percentage

end NUMINAMATH_CALUDE_first_solution_carbonated_water_percentage_l1061_106181


namespace NUMINAMATH_CALUDE_georges_donation_ratio_l1061_106100

theorem georges_donation_ratio : 
  ∀ (monthly_income donation groceries remaining : ℕ),
    monthly_income = 240 →
    groceries = 20 →
    remaining = 100 →
    monthly_income - donation - groceries = remaining →
    (donation : ℚ) / monthly_income = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_georges_donation_ratio_l1061_106100


namespace NUMINAMATH_CALUDE_odd_cycle_existence_l1061_106198

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type) :=
  (edges : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- The minimum degree of a graph is the minimum of the degrees of all vertices. -/
def min_degree {V : Type} (G : Graph V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each adjacent pair is connected by an edge. -/
def is_path {V : Type} (G : Graph V) (p : List V) : Prop := sorry

/-- A cycle in a graph is a path that starts and ends at the same vertex. -/
def is_cycle {V : Type} (G : Graph V) (c : List V) : Prop := sorry

/-- The length of a cycle is the number of edges in the cycle. -/
def cycle_length {V : Type} (c : List V) : ℕ := sorry

/-- A theorem stating that any graph with minimum degree at least 3 contains an odd cycle. -/
theorem odd_cycle_existence {V : Type} (G : Graph V) :
  min_degree G ≥ 3 → ∃ c : List V, is_cycle G c ∧ Odd (cycle_length c) := by
  sorry

end NUMINAMATH_CALUDE_odd_cycle_existence_l1061_106198


namespace NUMINAMATH_CALUDE_a_plays_d_on_third_day_l1061_106170

-- Define the players
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

-- Define a match as a pair of players
def Match := Player × Player

-- Define the schedule as a function from day to pair of matches
def Schedule := Nat → Match × Match

-- Define the condition that each player plays against each other exactly once
def playsAgainstEachOther (s : Schedule) : Prop :=
  ∀ p1 p2 : Player, p1 ≠ p2 → ∃ d : Nat, (s d).1 = (p1, p2) ∨ (s d).1 = (p2, p1) ∨ (s d).2 = (p1, p2) ∨ (s d).2 = (p2, p1)

-- Define the condition that each player plays only one match per day
def oneMatchPerDay (s : Schedule) : Prop :=
  ∀ d : Nat, ∀ p : Player, 
    ((s d).1.1 = p ∨ (s d).1.2 = p) → ((s d).2.1 ≠ p ∧ (s d).2.2 ≠ p)

-- Define the given conditions for the first two days
def givenConditions (s : Schedule) : Prop :=
  (s 1).1 = (Player.A, Player.C) ∨ (s 1).1 = (Player.C, Player.A) ∨ 
  (s 1).2 = (Player.A, Player.C) ∨ (s 1).2 = (Player.C, Player.A) ∧
  (s 2).1 = (Player.C, Player.D) ∨ (s 2).1 = (Player.D, Player.C) ∨ 
  (s 2).2 = (Player.C, Player.D) ∨ (s 2).2 = (Player.D, Player.C)

-- Theorem statement
theorem a_plays_d_on_third_day (s : Schedule) 
  (h1 : playsAgainstEachOther s) 
  (h2 : oneMatchPerDay s) 
  (h3 : givenConditions s) : 
  (s 3).1 = (Player.A, Player.D) ∨ (s 3).1 = (Player.D, Player.A) ∨ 
  (s 3).2 = (Player.A, Player.D) ∨ (s 3).2 = (Player.D, Player.A) :=
sorry

end NUMINAMATH_CALUDE_a_plays_d_on_third_day_l1061_106170


namespace NUMINAMATH_CALUDE_race_average_time_per_km_l1061_106131

theorem race_average_time_per_km (race_distance : ℝ) (first_half_time second_half_time : ℝ) :
  race_distance = 10 →
  first_half_time = 20 →
  second_half_time = 30 →
  (first_half_time + second_half_time) / race_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_race_average_time_per_km_l1061_106131


namespace NUMINAMATH_CALUDE_always_negative_l1061_106104

-- Define the chessboard as a function from positions to integers
def Chessboard := Fin 8 → Fin 8 → Int

-- Initial configuration of the chessboard
def initial_board : Chessboard :=
  fun row col => if row = 1 ∧ col = 1 then -1 else 1

-- Define a single operation (flipping signs in a row or column)
def flip_row_or_col (board : Chessboard) (is_row : Bool) (index : Fin 8) : Chessboard :=
  fun row col => 
    if (is_row ∧ row = index) ∨ (¬is_row ∧ col = index) then
      -board row col
    else
      board row col

-- Define a sequence of operations
def apply_operations (board : Chessboard) (ops : List (Bool × Fin 8)) : Chessboard :=
  ops.foldl (fun b (is_row, index) => flip_row_or_col b is_row index) board

-- Theorem statement
theorem always_negative (ops : List (Bool × Fin 8)) :
  ∃ row col, (apply_operations initial_board ops) row col < 0 := by
  sorry

end NUMINAMATH_CALUDE_always_negative_l1061_106104


namespace NUMINAMATH_CALUDE_coin_collection_value_l1061_106139

theorem coin_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) 
  (h1 : total_coins = 20)
  (h2 : sample_coins = 4)
  (h3 : sample_value = 16) :
  (total_coins : ℚ) * (sample_value : ℚ) / (sample_coins : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_value_l1061_106139


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_l1061_106135

theorem tan_theta_two_implies_expression (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ + Real.cos θ) * Real.cos (2 * θ) / Real.sin θ = -9/10 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_l1061_106135


namespace NUMINAMATH_CALUDE_lcm_504_630_980_l1061_106111

theorem lcm_504_630_980 : Nat.lcm (Nat.lcm 504 630) 980 = 17640 := by
  sorry

end NUMINAMATH_CALUDE_lcm_504_630_980_l1061_106111


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1061_106145

theorem inequality_solution_set :
  let f : ℝ → ℝ := λ x ↦ 2 * x
  let integral_value : ℝ := ∫ x in (0:ℝ)..1, f x
  {x : ℝ | |x - 2| > integral_value} = Set.Ioi 3 ∪ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1061_106145


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1061_106193

theorem complex_fraction_equality : ∀ (i : ℂ), i^2 = -1 → (5*i)/(1+2*i) = 2+i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1061_106193


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1061_106137

theorem quadratic_inequality_solution (p : ℝ) : 
  (∀ x, x^2 + p*x - 6 < 0 ↔ -3 < x ∧ x < 2) → p = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1061_106137


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_ten_l1061_106133

theorem least_subtraction_for_divisibility_by_ten (n : ℕ) (h : n = 427398) :
  ∃ (k : ℕ), k = 8 ∧ (n - k) % 10 = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_ten_l1061_106133


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1061_106110

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 2 ∧ c = b + 2
  even_a : Even a

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The triangle inequality for an EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    ∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t ≤ perimeter t' ∧
    perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1061_106110


namespace NUMINAMATH_CALUDE_opposite_sides_iff_a_in_range_l1061_106125

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation 3x - 2y + a = 0 -/
def line_equation (p : Point) (a : ℝ) : ℝ := 3 * p.x - 2 * p.y + a

/-- Two points are on opposite sides of the line if their line equation values have opposite signs -/
def opposite_sides (p1 p2 : Point) (a : ℝ) : Prop :=
  line_equation p1 a * line_equation p2 a < 0

/-- The main theorem -/
theorem opposite_sides_iff_a_in_range :
  ∀ (a : ℝ),
  opposite_sides (Point.mk 3 1) (Point.mk (-4) 6) a ↔ a > -7 ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_iff_a_in_range_l1061_106125


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_BaSO4_l1061_106101

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The molecular weight of BaSO4 in g/mol -/
def molecular_weight_BaSO4 : ℝ :=
  atomic_weight_Ba + atomic_weight_S + 4 * atomic_weight_O

/-- The number of moles of BaSO4 -/
def moles_BaSO4 : ℝ := 3

theorem molecular_weight_3_moles_BaSO4 :
  moles_BaSO4 * molecular_weight_BaSO4 = 700.164 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_BaSO4_l1061_106101


namespace NUMINAMATH_CALUDE_cistern_problem_l1061_106199

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

theorem cistern_problem :
  let length : ℝ := 6
  let width : ℝ := 4
  let depth : ℝ := 1.25
  cistern_wet_surface_area length width depth = 49 := by
  sorry

end NUMINAMATH_CALUDE_cistern_problem_l1061_106199


namespace NUMINAMATH_CALUDE_num_paths_to_bottom_right_l1061_106126

/-- Represents a vertex in the triangle grid --/
structure Vertex :=
  (x : Nat) (y : Nat)

/-- The number of paths to a vertex in the triangle grid --/
def numPaths : Vertex → Nat
| ⟨0, 0⟩ => 1  -- Top vertex
| ⟨0, y⟩ => 1  -- Left edge
| ⟨x, y⟩ => sorry  -- Other vertices

/-- The bottom right vertex of the triangle --/
def bottomRightVertex : Vertex :=
  ⟨3, 3⟩

/-- Theorem stating the number of paths to the bottom right vertex --/
theorem num_paths_to_bottom_right :
  numPaths bottomRightVertex = 22 := by sorry

end NUMINAMATH_CALUDE_num_paths_to_bottom_right_l1061_106126


namespace NUMINAMATH_CALUDE_gravel_path_width_is_quarter_length_l1061_106105

/-- Represents a rectangular garden with a rose garden and gravel path. -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  roseGardenArea : ℝ
  gravelPathWidth : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  roseGarden_half : roseGardenArea = (length * width) / 2
  gravelPath_constant : gravelPathWidth > 0

/-- Theorem stating that the gravel path width is one-fourth of the garden length. -/
theorem gravel_path_width_is_quarter_length (garden : RectangularGarden) :
  garden.gravelPathWidth = garden.length / 4 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_width_is_quarter_length_l1061_106105


namespace NUMINAMATH_CALUDE_joyce_typing_speed_l1061_106171

def team_size : ℕ := 5
def team_average : ℕ := 80
def rudy_speed : ℕ := 64
def gladys_speed : ℕ := 91
def lisa_speed : ℕ := 80
def mike_speed : ℕ := 89

theorem joyce_typing_speed :
  ∃ (joyce_speed : ℕ),
    joyce_speed = team_size * team_average - (rudy_speed + gladys_speed + lisa_speed + mike_speed) ∧
    joyce_speed = 76 := by
  sorry

end NUMINAMATH_CALUDE_joyce_typing_speed_l1061_106171


namespace NUMINAMATH_CALUDE_second_concert_attendance_l1061_106159

theorem second_concert_attendance 
  (first_concert : Nat) 
  (attendance_increase : Nat) 
  (h1 : first_concert = 65899)
  (h2 : attendance_increase = 119) :
  first_concert + attendance_increase = 66018 := by
  sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l1061_106159


namespace NUMINAMATH_CALUDE_curve_symmetry_l1061_106120

/-- The curve represented by the equation xy(x+y)=1 is symmetric about the line y=x -/
theorem curve_symmetry (x y : ℝ) : x * y * (x + y) = 1 ↔ y * x * (y + x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_l1061_106120


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1061_106180

theorem election_winner_percentage : 
  let votes : List ℕ := [1036, 4636, 11628]
  let total_votes := votes.sum
  let winning_votes := votes.maximum?
  let winning_percentage := (winning_votes.getD 0 : ℚ) / total_votes * 100
  winning_percentage = 67.2 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1061_106180


namespace NUMINAMATH_CALUDE_problem_solution_l1061_106129

/-- The function g(x) = -|x+m| -/
def g (m : ℝ) (x : ℝ) : ℝ := -|x + m|

/-- The function f(x) = 2|x-1| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*|x - 1| - a

theorem problem_solution :
  (∃! (n : ℤ), g m n > -1) ∧ (∀ (x : ℤ), g m x > -1 → x = -3) →
  m = 3 ∧
  (∀ x, f a x > g 3 x) →
  a < 4 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1061_106129


namespace NUMINAMATH_CALUDE_f_positive_implies_a_range_a_range_implies_f_positive_f_positive_iff_a_range_l1061_106176

/-- The function f(x) = ax^2 - 2x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

/-- Theorem: If f(x) > 0 for all x in (1, 4), then a > 1/2 -/
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f a x > 0) → a > 1/2 := by
  sorry

/-- Theorem: If a > 1/2, then f(x) > 0 for all x in (1, 4) -/
theorem a_range_implies_f_positive (a : ℝ) :
  a > 1/2 → (∀ x, 1 < x ∧ x < 4 → f a x > 0) := by
  sorry

/-- The main theorem combining both directions -/
theorem f_positive_iff_a_range (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f a x > 0) ↔ a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_range_a_range_implies_f_positive_f_positive_iff_a_range_l1061_106176


namespace NUMINAMATH_CALUDE_total_selling_price_l1061_106179

def bicycle_cost : ℚ := 1600
def scooter_cost : ℚ := 8000
def motorcycle_cost : ℚ := 15000

def bicycle_loss_percent : ℚ := 10
def scooter_loss_percent : ℚ := 5
def motorcycle_loss_percent : ℚ := 8

def selling_price (cost : ℚ) (loss_percent : ℚ) : ℚ :=
  cost - (cost * loss_percent / 100)

theorem total_selling_price :
  selling_price bicycle_cost bicycle_loss_percent +
  selling_price scooter_cost scooter_loss_percent +
  selling_price motorcycle_cost motorcycle_loss_percent = 22840 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_l1061_106179


namespace NUMINAMATH_CALUDE_second_athlete_high_jump_l1061_106175

def athlete1_long_jump : ℝ := 26
def athlete1_triple_jump : ℝ := 30
def athlete1_high_jump : ℝ := 7

def athlete2_long_jump : ℝ := 24
def athlete2_triple_jump : ℝ := 34

def winner_average_jump : ℝ := 22

def number_of_jumps : ℕ := 3

theorem second_athlete_high_jump :
  let athlete1_total := athlete1_long_jump + athlete1_triple_jump + athlete1_high_jump
  let athlete1_average := athlete1_total / number_of_jumps
  let athlete2_total_before_high := athlete2_long_jump + athlete2_triple_jump
  let winner_total := winner_average_jump * number_of_jumps
  athlete1_average < winner_average_jump →
  winner_total - athlete2_total_before_high = 8 := by
sorry

end NUMINAMATH_CALUDE_second_athlete_high_jump_l1061_106175


namespace NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l1061_106153

theorem one_fourth_divided_by_one_eighth : (1 / 4 : ℚ) / (1 / 8 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l1061_106153


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1061_106108

theorem rectangular_field_dimensions (m : ℝ) : 
  (2*m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1061_106108


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1061_106103

theorem simplify_and_rationalize (x : ℝ) (h : x = Real.sqrt 5) :
  1 / (2 + 2 / (x + 3)) = (7 + x) / 22 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1061_106103


namespace NUMINAMATH_CALUDE_equilateral_perimeter_is_60_l1061_106130

/-- An equilateral triangle with a side shared with an isosceles triangle -/
structure TrianglePair where
  equilateral_side : ℝ
  isosceles_base : ℝ
  isosceles_perimeter : ℝ
  equilateral_side_positive : 0 < equilateral_side
  isosceles_base_positive : 0 < isosceles_base
  isosceles_perimeter_positive : 0 < isosceles_perimeter

/-- The perimeter of the equilateral triangle in the TrianglePair -/
def equilateral_perimeter (tp : TrianglePair) : ℝ := 3 * tp.equilateral_side

/-- Theorem: The perimeter of the equilateral triangle is 60 -/
theorem equilateral_perimeter_is_60 (tp : TrianglePair)
  (h1 : tp.isosceles_base = 15)
  (h2 : tp.isosceles_perimeter = 55) :
  equilateral_perimeter tp = 60 := by
  sorry

#check equilateral_perimeter_is_60

end NUMINAMATH_CALUDE_equilateral_perimeter_is_60_l1061_106130


namespace NUMINAMATH_CALUDE_male_students_count_l1061_106162

/-- Calculates the total number of male students in first and second year -/
def total_male_students (total_first_year : ℕ) (female_first_year : ℕ) (male_second_year : ℕ) : ℕ :=
  (total_first_year - female_first_year) + male_second_year

/-- Proves that the total number of male students in first and second year is 620 -/
theorem male_students_count : 
  total_male_students 695 329 254 = 620 := by
  sorry

end NUMINAMATH_CALUDE_male_students_count_l1061_106162


namespace NUMINAMATH_CALUDE_sin_pi_12_function_value_l1061_106156

theorem sin_pi_12_function_value
  (f : ℝ → ℝ)
  (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_sin_pi_12_function_value_l1061_106156


namespace NUMINAMATH_CALUDE_perimeter_of_parallelogram_l1061_106151

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB AC BC : ℝ)
  (angleBAC : ℝ)

-- Define the parallelogram ADEF
structure Parallelogram :=
  (A D E F : ℝ × ℝ)

-- Define the problem statement
theorem perimeter_of_parallelogram (t : Triangle) (p : Parallelogram) : 
  t.AB = 20 →
  t.AC = 24 →
  t.BC = 18 →
  t.angleBAC = 60 * π / 180 →
  (p.D.1 - t.A.1) / (t.B.1 - t.A.1) = (p.D.2 - t.A.2) / (t.B.2 - t.A.2) →
  (p.E.1 - t.B.1) / (t.C.1 - t.B.1) = (p.E.2 - t.B.2) / (t.C.2 - t.B.2) →
  (p.F.1 - t.A.1) / (t.C.1 - t.A.1) = (p.F.2 - t.A.2) / (t.C.2 - t.A.2) →
  (p.E.1 - p.D.1) / (t.C.1 - t.A.1) = (p.E.2 - p.D.2) / (t.C.2 - t.A.2) →
  (p.F.1 - p.E.1) / (t.B.1 - t.A.1) = (p.F.2 - p.E.2) / (t.B.2 - t.A.2) →
  Real.sqrt ((p.A.1 - p.D.1)^2 + (p.A.2 - p.D.2)^2) +
  Real.sqrt ((p.D.1 - p.E.1)^2 + (p.D.2 - p.E.2)^2) +
  Real.sqrt ((p.E.1 - p.F.1)^2 + (p.E.2 - p.F.2)^2) +
  Real.sqrt ((p.F.1 - p.A.1)^2 + (p.F.2 - p.A.2)^2) = 44 :=
by sorry


end NUMINAMATH_CALUDE_perimeter_of_parallelogram_l1061_106151


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l1061_106106

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem eighth_term_of_specific_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = -1)
  (h_diff : ∃ d : ℤ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 8 = -22 :=
sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l1061_106106


namespace NUMINAMATH_CALUDE_triangle_type_l1061_106128

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B = Real.pi / 6 ∧  -- 30 degrees in radians
  t.c = 15 ∧
  t.b = 5 * Real.sqrt 3

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define right triangle
def is_right (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_type (t : Triangle) :
  triangle_conditions t → (is_isosceles t ∨ is_right t) :=
sorry

end NUMINAMATH_CALUDE_triangle_type_l1061_106128


namespace NUMINAMATH_CALUDE_rectangular_room_shorter_side_l1061_106112

/-- Given a rectangular room with perimeter 50 feet and area 126 square feet,
    prove that the length of the shorter side is 9 feet. -/
theorem rectangular_room_shorter_side
  (perimeter : ℝ)
  (area : ℝ)
  (h_perimeter : perimeter = 50)
  (h_area : area = 126) :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length ≥ width ∧
    2 * (length + width) = perimeter ∧
    length * width = area ∧
    width = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_room_shorter_side_l1061_106112


namespace NUMINAMATH_CALUDE_petya_has_five_five_ruble_coins_l1061_106148

/-- Represents the coin denominations --/
inductive Denomination
  | One
  | Two
  | Five
  | Ten

/-- Represents Petya's coin collection --/
structure CoinCollection where
  total : Nat
  not_two : Nat
  not_ten : Nat
  not_one : Nat

/-- Calculates the number of five-ruble coins in the collection --/
def count_five_ruble_coins (c : CoinCollection) : Nat :=
  c.total - ((c.total - c.not_two) + (c.total - c.not_ten) + (c.total - c.not_one))

/-- Theorem stating that Petya has 5 five-ruble coins --/
theorem petya_has_five_five_ruble_coins :
  let petya_coins : CoinCollection := {
    total := 25,
    not_two := 19,
    not_ten := 20,
    not_one := 16
  }
  count_five_ruble_coins petya_coins = 5 := by
  sorry

#eval count_five_ruble_coins {
  total := 25,
  not_two := 19,
  not_ten := 20,
  not_one := 16
}

end NUMINAMATH_CALUDE_petya_has_five_five_ruble_coins_l1061_106148


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1061_106186

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ 6) ∧ (∃ x ∈ Set.Icc 1 3, f a x = 6) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1061_106186


namespace NUMINAMATH_CALUDE_range_of_m_l1061_106169

/-- The condition p -/
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0

/-- The condition q -/
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

/-- The statement that "not p" is sufficient but not necessary for "not q" -/
def not_p_suff_not_nec_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

/-- The main theorem -/
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ not_p_suff_not_nec_not_q m) ↔ (m > 0 ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1061_106169


namespace NUMINAMATH_CALUDE_f_is_linear_equation_l1061_106195

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants. -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The specific equation we want to prove is linear. -/
def f (x y : ℝ) : ℝ := 4 * x - 5 * y - 5

theorem f_is_linear_equation : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_linear_equation_l1061_106195


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_l1061_106178

theorem arithmetic_progression_implies_equal (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let g := Real.sqrt (a * b)
  let p := (a + b) / 2
  let q := Real.sqrt ((a^2 + b^2) / 2)
  (g + q = 2 * p) → a = b :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_l1061_106178


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_is_integer_l1061_106163

theorem greatest_integer_fraction_is_integer : 
  ∀ y : ℤ, y > 12 → ¬(∃ k : ℤ, (y^2 - 3*y + 4) / (y - 4) = k) ∧ 
  ∃ k : ℤ, (12^2 - 3*12 + 4) / (12 - 4) = k := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_is_integer_l1061_106163


namespace NUMINAMATH_CALUDE_triangle_max_area_l1061_106144

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) 
  (h5 : 0 < C) (h6 : C < π) (h7 : A + B + C = π) 
  (h8 : Real.tan A * Real.tan B = 1) (h9 : Real.sqrt 3 = 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :
  (∃ (S : ℝ → ℝ), (∀ x, S x ≤ S (π/4)) ∧ S A = (3/4) * Real.sin (2*A)) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1061_106144


namespace NUMINAMATH_CALUDE_proportion_fourth_number_l1061_106122

theorem proportion_fourth_number (x y : ℚ) : 
  (0.75 : ℚ) / x = 10 / y ∧ x = 0.6 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_number_l1061_106122


namespace NUMINAMATH_CALUDE_median_and_mode_of_scores_l1061_106121

def scores : List Nat := [74, 84, 84, 84, 87, 92, 92]

def median (l : List Nat) : Nat := sorry

def mode (l : List Nat) : Nat := sorry

theorem median_and_mode_of_scores :
  median scores = 84 ∧ mode scores = 84 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_of_scores_l1061_106121


namespace NUMINAMATH_CALUDE_kennel_arrangement_count_l1061_106152

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_chickens : ℕ := 6
def num_dogs : ℕ := 4
def num_cats : ℕ := 5

def total_arrangements : ℕ := 2 * factorial num_chickens * factorial num_dogs * factorial num_cats

theorem kennel_arrangement_count :
  total_arrangements = 4147200 :=
by sorry

end NUMINAMATH_CALUDE_kennel_arrangement_count_l1061_106152


namespace NUMINAMATH_CALUDE_system_solution_l1061_106161

theorem system_solution (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (eq1 : x * y = 4 * z)
  (eq2 : x / y = 81)
  (eq3 : x * z = 36) :
  x = 36 ∧ y = 4/9 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1061_106161


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l1061_106134

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n.mod 8 = 0 → digit_sum n = 24 → n ≤ 888 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l1061_106134


namespace NUMINAMATH_CALUDE_richards_score_l1061_106102

/-- Richard and Bruno's miniature golf scores -/
def miniature_golf (richard_score bruno_score : ℕ) : Prop :=
  bruno_score = richard_score - 14 ∧ bruno_score = 48

theorem richards_score : ∃ (richard_score : ℕ), miniature_golf richard_score 48 ∧ richard_score = 62 := by
  sorry

end NUMINAMATH_CALUDE_richards_score_l1061_106102


namespace NUMINAMATH_CALUDE_haley_final_lives_l1061_106174

/-- Calculate the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem stating that for the given scenario, the final number of lives is 46 -/
theorem haley_final_lives :
  final_lives 14 4 36 = 46 := by
  sorry

end NUMINAMATH_CALUDE_haley_final_lives_l1061_106174


namespace NUMINAMATH_CALUDE_house_sale_loss_percentage_l1061_106185

def initial_value : ℝ := 100000
def profit_percentage : ℝ := 0.10
def final_selling_price : ℝ := 99000

theorem house_sale_loss_percentage :
  let first_sale_price := initial_value * (1 + profit_percentage)
  let loss_amount := first_sale_price - final_selling_price
  let loss_percentage := loss_amount / first_sale_price * 100
  loss_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_house_sale_loss_percentage_l1061_106185


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1061_106115

theorem no_primes_in_range (n m : ℕ) (hn : n > 1) (hm : 1 ≤ m ∧ m ≤ n) :
  ∀ k, n! + m < k ∧ k < n! + n + m → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1061_106115
