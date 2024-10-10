import Mathlib

namespace sachin_rahul_age_difference_l2696_269675

theorem sachin_rahul_age_difference :
  ∀ (sachin_age rahul_age : ℝ),
    sachin_age = 38.5 →
    sachin_age / rahul_age = 11 / 9 →
    sachin_age - rahul_age = 7 :=
by
  sorry

end sachin_rahul_age_difference_l2696_269675


namespace sum_in_base_5_l2696_269626

/-- Converts a natural number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Converts a list representing a number in a given base to base 10 -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Adds two numbers in a given base -/
def addInBase (a b : List ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem sum_in_base_5 :
  let n1 := 29
  let n2 := 45
  let base4 := toBase n1 4
  let base5 := toBase n2 5
  let sum := addInBase base4 base5 5
  sum = [2, 4, 4] := by
  sorry

end sum_in_base_5_l2696_269626


namespace two_zeros_iff_m_in_range_l2696_269628

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (2 * log x - x) + 1 / x^2 - 1 / x

theorem two_zeros_iff_m_in_range (m : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧
    ∀ z : ℝ, 0 < z → f m z = 0 → (z = x ∨ z = y)) ↔
  m ∈ Set.Ioo (1 / (8 * (log 2 - 1))) 0 :=
sorry

end two_zeros_iff_m_in_range_l2696_269628


namespace complex_division_l2696_269661

theorem complex_division (i : ℂ) (h : i * i = -1) : 
  (2 - i) / (1 + i) = 1/2 - 3/2 * i := by sorry

end complex_division_l2696_269661


namespace total_cost_calculation_l2696_269614

def tshirt_price : ℝ := 8
def sweater_price : ℝ := 18
def jacket_price : ℝ := 80
def jeans_price : ℝ := 35
def shoe_price : ℝ := 60

def jacket_discount : ℝ := 0.1
def shoe_discount : ℝ := 0.15

def clothing_tax_rate : ℝ := 0.05
def shoe_tax_rate : ℝ := 0.08

def tshirt_quantity : ℕ := 6
def sweater_quantity : ℕ := 4
def jacket_quantity : ℕ := 5
def jeans_quantity : ℕ := 3
def shoe_quantity : ℕ := 2

theorem total_cost_calculation :
  let tshirt_cost := tshirt_price * tshirt_quantity
  let sweater_cost := sweater_price * sweater_quantity
  let jacket_cost := jacket_price * jacket_quantity * (1 - jacket_discount)
  let jeans_cost := jeans_price * jeans_quantity
  let shoe_cost := shoe_price * shoe_quantity * (1 - shoe_discount)
  
  let clothing_subtotal := tshirt_cost + sweater_cost + jacket_cost + jeans_cost
  let shoe_subtotal := shoe_cost
  
  let clothing_tax := clothing_subtotal * clothing_tax_rate
  let shoe_tax := shoe_subtotal * shoe_tax_rate
  
  let total_cost := clothing_subtotal + shoe_subtotal + clothing_tax + shoe_tax
  
  total_cost = 724.41 := by sorry

end total_cost_calculation_l2696_269614


namespace book_donation_equation_l2696_269606

/-- Proves that the equation for book donations over three years is correct -/
theorem book_donation_equation (x : ℝ) : 
  (400 : ℝ) + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 → 
  (∃ (y : ℝ), y > 0 ∧ 400 * (1 + y) + 400 * (1 + y)^2 = 1125) :=
by
  sorry


end book_donation_equation_l2696_269606


namespace inequality_proof_l2696_269698

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) := by
  sorry

end inequality_proof_l2696_269698


namespace set_operations_l2696_269673

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6,7,8}

-- Define set A
def A : Finset Nat := {5,6,7,8}

-- Define set B
def B : Finset Nat := {2,4,6,8}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {6,8}) ∧
  (U \ A = {1,2,3,4}) ∧
  (U \ B = {1,3,5,7}) := by
  sorry

end set_operations_l2696_269673


namespace binomial_expansion_cube_l2696_269611

theorem binomial_expansion_cube (x y : ℝ) : 
  (x + y)^3 = x^3 + 3*x^2*y + 3*x*y^2 + y^3 := by sorry

end binomial_expansion_cube_l2696_269611


namespace jacket_cost_ratio_l2696_269658

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 4/5
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price = 3/5 := by
sorry

end jacket_cost_ratio_l2696_269658


namespace det_equals_xy_l2696_269679

/-- The determinant of the matrix
    [1, x, y]
    [1, x+y, y]
    [1, x, x+y]
    is equal to xy -/
theorem det_equals_xy (x y : ℝ) : 
  Matrix.det !![1, x, y; 1, x+y, y; 1, x, x+y] = x * y := by
  sorry

end det_equals_xy_l2696_269679


namespace mod_seven_equivalence_l2696_269627

theorem mod_seven_equivalence : 47^1357 - 23^1357 ≡ 3 [ZMOD 7] := by sorry

end mod_seven_equivalence_l2696_269627


namespace gcd_of_36_45_495_l2696_269642

theorem gcd_of_36_45_495 : Nat.gcd 36 (Nat.gcd 45 495) = 9 := by
  sorry

end gcd_of_36_45_495_l2696_269642


namespace infinite_geometric_series_common_ratio_l2696_269685

theorem infinite_geometric_series_common_ratio 
  (a : ℝ) 
  (S : ℝ) 
  (h1 : a = 512) 
  (h2 : S = 8000) : 
  ∃ (r : ℝ), r = 0.936 ∧ S = a / (1 - r) := by
sorry

end infinite_geometric_series_common_ratio_l2696_269685


namespace cost_of_apples_and_oranges_l2696_269618

/-- The cost of apples and oranges given the initial amount and remaining amount -/
def cost_of_fruits (initial_amount remaining_amount : ℚ) : ℚ :=
  initial_amount - remaining_amount

/-- Theorem: The cost of apples and oranges is $15.00 -/
theorem cost_of_apples_and_oranges :
  cost_of_fruits 100 85 = 15 := by
  sorry

end cost_of_apples_and_oranges_l2696_269618


namespace b_age_l2696_269637

def problem (a b c d : ℕ) : Prop :=
  (a = b + 2) ∧ 
  (b = 2 * c) ∧ 
  (d = b - 3) ∧ 
  (a + b + c + d = 60)

theorem b_age (a b c d : ℕ) (h : problem a b c d) : b = 17 := by
  sorry

end b_age_l2696_269637


namespace hyperbola_intersection_midpoint_l2696_269669

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line
def line (x y : ℝ) : Prop := 4*x - y - 7 = 0

-- Theorem statement
theorem hyperbola_intersection_midpoint :
  ∃ (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    line P.1 P.2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
sorry

end hyperbola_intersection_midpoint_l2696_269669


namespace max_remaining_pairs_l2696_269657

def original_total_pairs : ℕ := 20
def original_high_heeled_pairs : ℕ := 4
def original_flat_pairs : ℕ := 16
def lost_high_heeled_shoes : ℕ := 5
def lost_flat_shoes : ℕ := 11

def shoes_per_pair : ℕ := 2

theorem max_remaining_pairs : 
  let original_high_heeled_shoes := original_high_heeled_pairs * shoes_per_pair
  let original_flat_shoes := original_flat_pairs * shoes_per_pair
  let remaining_high_heeled_shoes := original_high_heeled_shoes - lost_high_heeled_shoes
  let remaining_flat_shoes := original_flat_shoes - lost_flat_shoes
  let remaining_high_heeled_pairs := remaining_high_heeled_shoes / shoes_per_pair
  let remaining_flat_pairs := remaining_flat_shoes / shoes_per_pair
  remaining_high_heeled_pairs + remaining_flat_pairs = 11 :=
by sorry

end max_remaining_pairs_l2696_269657


namespace typists_productivity_l2696_269677

/-- Given that 20 typists can type 42 letters in 20 minutes, 
    prove that 30 typists can type 189 letters in 1 hour at the same rate. -/
theorem typists_productivity (typists_20 : ℕ) (letters_20 : ℕ) (minutes_20 : ℕ) 
  (typists_30 : ℕ) (minutes_60 : ℕ) :
  typists_20 = 20 →
  letters_20 = 42 →
  minutes_20 = 20 →
  typists_30 = 30 →
  minutes_60 = 60 →
  (typists_30 : ℚ) * (letters_20 : ℚ) / (typists_20 : ℚ) * (minutes_60 : ℚ) / (minutes_20 : ℚ) = 189 :=
by sorry

end typists_productivity_l2696_269677


namespace inconsistent_statistics_l2696_269672

theorem inconsistent_statistics (x_bar m S_squared : ℝ) 
  (h1 : x_bar = 0)
  (h2 : m = 4)
  (h3 : S_squared = 15.917)
  : ¬ (|x_bar - m| ≤ Real.sqrt S_squared) := by
  sorry

end inconsistent_statistics_l2696_269672


namespace compound_molecular_weight_l2696_269635

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 5

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := num_N * atomic_weight_N + num_O * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 108.02 := by sorry

end compound_molecular_weight_l2696_269635


namespace cubic_equation_roots_l2696_269639

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = Real.sqrt 3 ∧ x₂ = (Real.sqrt 3) / 3 ∧ x₃ = -(2 * Real.sqrt 3)) ∧
  (x₁ * x₂ = 1) ∧
  (3 * x₁^3 + 2 * Real.sqrt 3 * x₁^2 - 21 * x₁ + 6 * Real.sqrt 3 = 0) ∧
  (3 * x₂^3 + 2 * Real.sqrt 3 * x₂^2 - 21 * x₂ + 6 * Real.sqrt 3 = 0) ∧
  (3 * x₃^3 + 2 * Real.sqrt 3 * x₃^2 - 21 * x₃ + 6 * Real.sqrt 3 = 0) :=
by sorry

end cubic_equation_roots_l2696_269639


namespace movie_collection_average_usage_l2696_269699

/-- Given a movie collection that occupies 27,000 megabytes of disk space and lasts for 15 days
    of continuous viewing, the average megabyte usage per hour is 75 megabytes. -/
theorem movie_collection_average_usage
  (total_megabytes : ℕ)
  (total_days : ℕ)
  (h_megabytes : total_megabytes = 27000)
  (h_days : total_days = 15) :
  (total_megabytes : ℚ) / (total_days * 24 : ℚ) = 75 := by
  sorry

end movie_collection_average_usage_l2696_269699


namespace basketball_team_starters_l2696_269608

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players -/
def total_players : ℕ := 18

/-- The number of quadruplets -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 7

/-- The number of non-quadruplet players -/
def non_quadruplets : ℕ := total_players - quadruplets

theorem basketball_team_starters :
  choose total_players starters - choose non_quadruplets (starters - quadruplets) = 31460 :=
by sorry

end basketball_team_starters_l2696_269608


namespace remaining_two_average_l2696_269634

theorem remaining_two_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₁ + n₂) / 2 = 3.4 →
  (n₃ + n₄) / 2 = 3.85 →
  (n₅ + n₆) / 2 = 4.6 := by
sorry

end remaining_two_average_l2696_269634


namespace bread_and_ham_percentage_l2696_269666

def bread_cost : ℚ := 50
def ham_cost : ℚ := 150
def cake_cost : ℚ := 200

theorem bread_and_ham_percentage : 
  (bread_cost + ham_cost) / (bread_cost + ham_cost + cake_cost) = 1/2 := by
  sorry

end bread_and_ham_percentage_l2696_269666


namespace four_digit_integers_with_one_or_seven_l2696_269686

/-- The number of four-digit positive integers -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit positive integers without 1 or 7 -/
def four_digit_integers_without_one_or_seven : ℕ := 3584

/-- Theorem: The number of four-digit positive integers with at least one digit as 1 or 7 -/
theorem four_digit_integers_with_one_or_seven :
  total_four_digit_integers - four_digit_integers_without_one_or_seven = 5416 := by
  sorry

end four_digit_integers_with_one_or_seven_l2696_269686


namespace fence_cost_l2696_269683

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 60) :
  4 * Real.sqrt area * price_per_foot = 4080 := by
  sorry

#check fence_cost

end fence_cost_l2696_269683


namespace cable_on_hand_theorem_l2696_269623

/-- Given a total length of cable and a section length, calculates the number of sections. -/
def calculateSections (totalLength sectionLength : ℕ) : ℕ := totalLength / sectionLength

/-- Calculates the number of sections given away. -/
def sectionsGivenAway (totalSections : ℕ) : ℕ := totalSections / 4

/-- Calculates the number of sections remaining after giving some away. -/
def remainingSections (totalSections givenAway : ℕ) : ℕ := totalSections - givenAway

/-- Calculates the number of sections put in storage. -/
def sectionsInStorage (remainingSections : ℕ) : ℕ := remainingSections / 2

/-- Calculates the number of sections kept on hand. -/
def sectionsOnHand (remainingSections inStorage : ℕ) : ℕ := remainingSections - inStorage

/-- Calculates the total length of cable kept on hand. -/
def cableOnHand (sectionsOnHand sectionLength : ℕ) : ℕ := sectionsOnHand * sectionLength

theorem cable_on_hand_theorem (totalLength sectionLength : ℕ) 
    (h1 : totalLength = 1000)
    (h2 : sectionLength = 25) : 
  cableOnHand 
    (sectionsOnHand 
      (remainingSections 
        (calculateSections totalLength sectionLength) 
        (sectionsGivenAway (calculateSections totalLength sectionLength)))
      (sectionsInStorage 
        (remainingSections 
          (calculateSections totalLength sectionLength) 
          (sectionsGivenAway (calculateSections totalLength sectionLength)))))
    sectionLength = 375 := by
  sorry

end cable_on_hand_theorem_l2696_269623


namespace xy_minus_ten_squared_ge_64_l2696_269630

theorem xy_minus_ten_squared_ge_64 (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 := by
  sorry

end xy_minus_ten_squared_ge_64_l2696_269630


namespace problem_statement_l2696_269615

theorem problem_statement (x y : ℝ) (h : |x - 5| + (x - y - 1)^2 = 0) : 
  (x - y)^2023 = 1 := by
sorry

end problem_statement_l2696_269615


namespace log_3_81_sqrt_81_equals_6_l2696_269605

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_3_81_sqrt_81_equals_6 :
  log 3 (81 * Real.sqrt 81) = 6 := by
  sorry

end log_3_81_sqrt_81_equals_6_l2696_269605


namespace fib_linear_combination_fib_quadratic_combination_l2696_269629

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Part (a)
theorem fib_linear_combination (a b : ℝ) :
  (∀ n : ℕ, ∃ k : ℕ, a * fib n + b * fib (n + 1) = fib k) ↔
  ∃ k : ℕ, a = fib (k - 1) ∧ b = fib k :=
sorry

-- Part (b)
theorem fib_quadratic_combination (u v : ℝ) :
  (u > 0 ∧ v > 0 ∧ ∀ n : ℕ, ∃ k : ℕ, u * (fib n)^2 + v * (fib (n + 1))^2 = fib k) ↔
  u = 1 ∧ v = 1 :=
sorry

end fib_linear_combination_fib_quadratic_combination_l2696_269629


namespace blue_marble_probability_l2696_269682

theorem blue_marble_probability (total : ℕ) (yellow : ℕ) :
  total = 120 →
  yellow = 30 →
  let green := yellow / 3
  let red := 2 * yellow
  let blue := total - (yellow + green + red)
  (blue : ℚ) / total = 1 / 6 := by sorry

end blue_marble_probability_l2696_269682


namespace quadratic_equation_conditions_l2696_269660

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 2 = 0
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x → x < 1 → x^2 - a < 0

-- Define the set of real numbers that satisfy the conditions for p
def S₁ : Set ℝ := {a : ℝ | a ≤ -2 * Real.sqrt 2 ∨ a ≥ 2 * Real.sqrt 2}

-- Define the set of real numbers that satisfy the conditions for exactly one of p or q
def S₂ : Set ℝ := {a : ℝ | a ≤ -2 * Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2 * Real.sqrt 2)}

-- State the theorem
theorem quadratic_equation_conditions (a : ℝ) :
  (p a ↔ a ∈ S₁) ∧
  ((p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ a ∈ S₂) :=
sorry

end quadratic_equation_conditions_l2696_269660


namespace lucas_initial_beds_l2696_269632

/-- The number of pet beds Lucas can add to his room -/
def additional_beds : ℕ := 8

/-- The number of beds required per pet -/
def beds_per_pet : ℕ := 2

/-- The total number of pets Lucas's room can accommodate -/
def total_pets : ℕ := 10

/-- The initial number of pet beds in Lucas's room -/
def initial_beds : ℕ := total_pets * beds_per_pet - additional_beds

theorem lucas_initial_beds :
  initial_beds = 12 :=
by sorry

end lucas_initial_beds_l2696_269632


namespace sequence_general_term_l2696_269612

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = 2^n + 3) →
  (a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) :=
by sorry

end sequence_general_term_l2696_269612


namespace seventh_term_of_sequence_l2696_269667

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ := a * r^(n - 1)

theorem seventh_term_of_sequence (a₁ a₅ : ℕ) (h₁ : a₁ = 3) (h₅ : a₅ = 243) :
  ∃ r : ℕ, 
    (geometric_sequence a₁ r 5 = a₅) ∧ 
    (geometric_sequence a₁ r 7 = 2187) := by
  sorry

end seventh_term_of_sequence_l2696_269667


namespace function_properties_l2696_269652

def f (a b c x : ℝ) := a * x^4 + b * x^2 + c

theorem function_properties (a b c : ℝ) :
  f a b c 0 = 1 ∧
  (∀ x, x = 1 → f a b c x + 2 = x) ∧
  f a b c 1 = -1 →
  a = 5/2 ∧ c = 1 ∧
  ∀ x, (- (3 * Real.sqrt 10) / 10 < x ∧ x < 0) ∨ (3 * Real.sqrt 10 / 10 < x) →
    ∀ y, x < y → f a b c x < f a b c y :=
by sorry

end function_properties_l2696_269652


namespace triangle_area_product_l2696_269676

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 2 * a * x + 3 * b * y = 24) →
  (1/2 * (24 / (2 * a)) * (24 / (3 * b)) = 12) →
  a * b = 4 := by
sorry

end triangle_area_product_l2696_269676


namespace yoongis_rank_l2696_269689

theorem yoongis_rank (namjoons_rank yoongis_rank : ℕ) : 
  namjoons_rank = 2 →
  yoongis_rank = namjoons_rank + 10 →
  yoongis_rank = 12 :=
by sorry

end yoongis_rank_l2696_269689


namespace parabola_sum_l2696_269650

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate stating that a point (x, y) is on the parabola -/
def on_parabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Predicate stating that (h, k) is the vertex of the parabola -/
def has_vertex (p : Parabola) (h k : ℝ) : Prop :=
  ∀ x, p.a * (x - h)^2 + k = p.a * x^2 + p.b * x + p.c

/-- The axis of symmetry is vertical when x = h, where (h, k) is the vertex -/
def has_vertical_axis_of_symmetry (p : Parabola) (h : ℝ) : Prop :=
  ∀ x y, on_parabola p x y ↔ on_parabola p (2*h - x) y

theorem parabola_sum (p : Parabola) :
  has_vertex p 4 4 →
  has_vertical_axis_of_symmetry p 4 →
  on_parabola p 3 0 →
  p.a + p.b + p.c = -32 := by
sorry

end parabola_sum_l2696_269650


namespace total_players_is_60_l2696_269653

/-- Represents the total number of players in each sport and their intersections --/
structure SportPlayers where
  cricket : ℕ
  hockey : ℕ
  football : ℕ
  softball : ℕ
  cricket_hockey : ℕ
  cricket_football : ℕ
  cricket_softball : ℕ
  hockey_football : ℕ
  hockey_softball : ℕ
  football_softball : ℕ
  cricket_hockey_football : ℕ

/-- Calculate the total number of unique players given the sport participation data --/
def totalUniquePlayers (sp : SportPlayers) : ℕ :=
  sp.cricket + sp.hockey + sp.football + sp.softball
  - sp.cricket_hockey - sp.cricket_football - sp.cricket_softball
  - sp.hockey_football - sp.hockey_softball - sp.football_softball
  + sp.cricket_hockey_football

/-- The main theorem stating that given the specific sport participation data,
    the total number of unique players is 60 --/
theorem total_players_is_60 (sp : SportPlayers)
  (h1 : sp.cricket = 25)
  (h2 : sp.hockey = 20)
  (h3 : sp.football = 30)
  (h4 : sp.softball = 18)
  (h5 : sp.cricket_hockey = 5)
  (h6 : sp.cricket_football = 8)
  (h7 : sp.cricket_softball = 3)
  (h8 : sp.hockey_football = 4)
  (h9 : sp.hockey_softball = 6)
  (h10 : sp.football_softball = 9)
  (h11 : sp.cricket_hockey_football = 2) :
  totalUniquePlayers sp = 60 := by
  sorry


end total_players_is_60_l2696_269653


namespace cos_alpha_value_l2696_269609

theorem cos_alpha_value (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.sin (π / 6) ∧ y = -2 * Real.cos (π / 6) ∧ 
   x = 2 * Real.sin α ∧ y = -2 * Real.cos α) → 
  Real.cos α = 1 / 2 := by
sorry

end cos_alpha_value_l2696_269609


namespace tan_double_angle_l2696_269668

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.tan (2 * α) = -3/4 := by
sorry

end tan_double_angle_l2696_269668


namespace binomial_19_10_l2696_269688

theorem binomial_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end binomial_19_10_l2696_269688


namespace problem_solution_l2696_269638

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x - 3 / (x^2) - 1

theorem problem_solution :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, Real.exp 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ m, (∀ x, 0 < x → 2 * f x ≥ g m x) ↔ m ≤ 4) ∧
  (∀ x, 0 < x → Real.log x < (2 * x / Real.exp 1) - (x^2 / Real.exp x)) := by
sorry

end problem_solution_l2696_269638


namespace base_10_255_equals_base_4_3333_l2696_269693

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Theorem: 255 in base 10 is equal to 3333 in base 4 -/
theorem base_10_255_equals_base_4_3333 :
  255 = base4ToNat [3, 3, 3, 3] := by
  sorry

end base_10_255_equals_base_4_3333_l2696_269693


namespace solution_set_when_a_is_one_a_value_when_max_is_six_l2696_269664

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x < 1} = {x : ℝ | x < 1/2} := by sorry

-- Part II
theorem a_value_when_max_is_six :
  (∃ (x : ℝ), f a x = 6) ∧ (∀ (x : ℝ), f a x ≤ 6) → a = 5 ∨ a = -7 := by sorry

end solution_set_when_a_is_one_a_value_when_max_is_six_l2696_269664


namespace angle_terminal_side_ratio_l2696_269631

theorem angle_terminal_side_ratio (a : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos a = 1 ∧ r * Real.sin a = -2) →
  (2 * Real.sin a) / Real.cos a = 4 := by
  sorry

end angle_terminal_side_ratio_l2696_269631


namespace rhombus_area_l2696_269621

/-- The area of a rhombus with vertices at (0, 3.5), (12, 0), (0, -3.5), and (-12, 0) is 84 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (12, 0), (0, -3.5), (-12, 0)]
  let diagonal1 : ℝ := |3.5 - (-3.5)|
  let diagonal2 : ℝ := |12 - (-12)|
  let area : ℝ := (diagonal1 * diagonal2) / 2
  area = 84 := by sorry

end rhombus_area_l2696_269621


namespace stratified_sampling_theorem_l2696_269604

/-- Represents the number of papers drawn from a school --/
structure SchoolSample where
  total : ℕ
  drawn : ℕ

/-- Represents the sampling data for all schools --/
structure SamplingData where
  schoolA : SchoolSample
  schoolB : SchoolSample
  schoolC : SchoolSample

/-- Calculates the total number of papers drawn using stratified sampling --/
def totalDrawn (data : SamplingData) : ℕ :=
  let ratio := data.schoolC.drawn / data.schoolC.total
  (data.schoolA.total + data.schoolB.total + data.schoolC.total) * ratio

theorem stratified_sampling_theorem (data : SamplingData) 
  (h1 : data.schoolA.total = 1260)
  (h2 : data.schoolB.total = 720)
  (h3 : data.schoolC.total = 900)
  (h4 : data.schoolC.drawn = 50) :
  totalDrawn data = 160 := by
  sorry

#eval totalDrawn { 
  schoolA := { total := 1260, drawn := 0 },
  schoolB := { total := 720, drawn := 0 },
  schoolC := { total := 900, drawn := 50 }
}

end stratified_sampling_theorem_l2696_269604


namespace scientific_notation_of_3185800_l2696_269681

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of decimal places -/
def roundToDecimalPlaces (sn : ScientificNotation) (places : ℕ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_3185800 :
  let original := 3185800
  let scientificForm := toScientificNotation original
  let rounded := roundToDecimalPlaces scientificForm 1
  rounded.coefficient = 3.2 ∧ rounded.exponent = 6 := by
  sorry

end scientific_notation_of_3185800_l2696_269681


namespace part_I_part_II_l2696_269644

-- Define set A
def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem for part (I)
theorem part_I : A = {x | -1 < x ∧ x ≤ 5} := by sorry

-- Theorem for part (II)
theorem part_II : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

end part_I_part_II_l2696_269644


namespace orange_picking_ratio_l2696_269674

/-- Proves that the ratio of oranges picked on Tuesday to Monday is 3:1 --/
theorem orange_picking_ratio :
  let monday_oranges : ℕ := 100
  let wednesday_oranges : ℕ := 70
  let total_oranges : ℕ := 470
  let tuesday_oranges : ℕ := total_oranges - monday_oranges - wednesday_oranges
  tuesday_oranges / monday_oranges = 3 := by
  sorry

end orange_picking_ratio_l2696_269674


namespace luncheon_table_capacity_l2696_269671

/-- Given a luncheon where 24 people were invited, 10 didn't show up, and 2 tables were needed,
    prove that each table could hold 7 people. -/
theorem luncheon_table_capacity :
  ∀ (invited : ℕ) (no_show : ℕ) (tables : ℕ) (capacity : ℕ),
    invited = 24 →
    no_show = 10 →
    tables = 2 →
    capacity = (invited - no_show) / tables →
    capacity = 7 := by
  sorry

end luncheon_table_capacity_l2696_269671


namespace hyperbola_asymptotes_l2696_269694

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 / 4 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = 2 * Real.sqrt 3 / 3 * x ∨ y = -2 * Real.sqrt 3 / 3 * x

/-- Theorem stating that the given asymptote equations are correct for the given hyperbola -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l2696_269694


namespace min_sum_m_n_in_arithmetic_sequence_l2696_269690

theorem min_sum_m_n_in_arithmetic_sequence (a : ℕ → ℕ) (d m n : ℕ) :
  (∀ k, a k > 0) →
  (∀ k, a (k + 1) = a k + d) →
  a 1 = 1919 →
  a m = 1949 →
  a n = 2019 →
  m > 0 →
  n > 0 →
  ∃ (m' n' : ℕ), m' > 0 ∧ n' > 0 ∧ a m' = 1949 ∧ a n' = 2019 ∧ m' + n' = 15 ∧
    ∀ (p q : ℕ), p > 0 → q > 0 → a p = 1949 → a q = 2019 → m' + n' ≤ p + q :=
by sorry

end min_sum_m_n_in_arithmetic_sequence_l2696_269690


namespace symmetry_axis_implies_equal_coefficients_l2696_269602

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where ab ≠ 0,
    if f has a symmetry axis at x = π/8, then a = b -/
theorem symmetry_axis_implies_equal_coefficients
  (a b : ℝ) (hab : a * b ≠ 0)
  (h_symmetry : ∀ x : ℝ, a * Real.sin (2 * (π/8 + x)) + b * Real.cos (2 * (π/8 + x)) =
                         a * Real.sin (2 * (π/8 - x)) + b * Real.cos (2 * (π/8 - x))) :
  a = b :=
sorry

end symmetry_axis_implies_equal_coefficients_l2696_269602


namespace similar_triangles_side_length_l2696_269663

/-- Two triangles are similar -/
structure SimilarTriangles (T1 T2 : Type) :=
  (ratio : ℝ)
  (similar : T1 → T2 → Prop)

/-- Triangle XYZ -/
structure TriangleXYZ :=
  (X Y Z : ℝ × ℝ)
  (YZ : ℝ)
  (XZ : ℝ)

/-- Triangle MNP -/
structure TriangleMNP :=
  (M N P : ℝ × ℝ)
  (MN : ℝ)
  (NP : ℝ)

theorem similar_triangles_side_length 
  (XYZ : TriangleXYZ) 
  (MNP : TriangleMNP) 
  (sim : SimilarTriangles TriangleXYZ TriangleMNP) 
  (h_sim : sim.similar XYZ MNP) 
  (h_YZ : XYZ.YZ = 10) 
  (h_XZ : XYZ.XZ = 7) 
  (h_MN : MNP.MN = 4.2) : 
  MNP.NP = 6 := by
  sorry

end similar_triangles_side_length_l2696_269663


namespace impossible_to_empty_heap_l2696_269617

/-- Represents the state of the three heaps of stones -/
structure HeapState :=
  (heap1 : Nat) (heap2 : Nat) (heap3 : Nat)

/-- Defines the allowed operations on the heaps -/
inductive Operation
  | Add (target : Nat) (source1 : Nat) (source2 : Nat)
  | Remove (target : Nat) (source1 : Nat) (source2 : Nat)

/-- Applies an operation to a heap state -/
def applyOperation (state : HeapState) (op : Operation) : HeapState :=
  match op with
  | Operation.Add 0 1 2 => HeapState.mk (state.heap1 + state.heap2 + state.heap3) state.heap2 state.heap3
  | Operation.Add 1 0 2 => HeapState.mk state.heap1 (state.heap2 + state.heap1 + state.heap3) state.heap3
  | Operation.Add 2 0 1 => HeapState.mk state.heap1 state.heap2 (state.heap3 + state.heap1 + state.heap2)
  | Operation.Remove 0 1 2 => HeapState.mk (state.heap1 - state.heap2 - state.heap3) state.heap2 state.heap3
  | Operation.Remove 1 0 2 => HeapState.mk state.heap1 (state.heap2 - state.heap1 - state.heap3) state.heap3
  | Operation.Remove 2 0 1 => HeapState.mk state.heap1 state.heap2 (state.heap3 - state.heap1 - state.heap2)
  | _ => state  -- Invalid operations return the original state

/-- Defines the initial state of the heaps -/
def initialState : HeapState := HeapState.mk 1993 199 19

/-- Theorem stating that it's impossible to make a heap empty -/
theorem impossible_to_empty_heap :
  ∀ (operations : List Operation),
    let finalState := operations.foldl applyOperation initialState
    ¬(finalState.heap1 = 0 ∨ finalState.heap2 = 0 ∨ finalState.heap3 = 0) :=
by
  sorry


end impossible_to_empty_heap_l2696_269617


namespace f_monotone_increasing_iff_a_range_l2696_269613

/-- A function f(x) = 2x^2 - ax + 5 that is monotonically increasing on [1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - a * x + 5

/-- The property of f being monotonically increasing on [1, +∞) -/
def monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f a x < f a y

/-- The theorem stating the range of a for which f is monotonically increasing on [1, +∞) -/
theorem f_monotone_increasing_iff_a_range :
  ∀ a : ℝ, monotone_increasing a ↔ a ≤ 4 :=
sorry

end f_monotone_increasing_iff_a_range_l2696_269613


namespace antiderivative_of_f_l2696_269620

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x

-- Define the antiderivative F
def F (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem antiderivative_of_f (x : ℝ) : 
  (deriv F x = f x) ∧ (F 1 = 3) := by sorry

end antiderivative_of_f_l2696_269620


namespace impossible_single_piece_on_center_l2696_269697

/-- Represents a square on the solitaire board -/
inductive Square
| One
| Two
| Three

/-- Represents the state of the solitaire board -/
structure BoardState where
  occupied_ones : Nat
  occupied_twos : Nat

/-- Represents a valid move in the solitaire game -/
inductive Move
| HorizontalMove
| VerticalMove

/-- Defines K as the sum of occupied 1-squares and 2-squares -/
def K (state : BoardState) : Nat :=
  state.occupied_ones + state.occupied_twos

/-- The initial state of the board -/
def initial_state : BoardState :=
  { occupied_ones := 15, occupied_twos := 15 }

/-- Applies a move to the board state -/
def apply_move (state : BoardState) (move : Move) : BoardState :=
  sorry

/-- Theorem stating that it's impossible to end with a single piece on the central square -/
theorem impossible_single_piece_on_center :
  ∀ (moves : List Move),
    let final_state := moves.foldl apply_move initial_state
    ¬(K final_state = 1 ∧ final_state.occupied_ones + final_state.occupied_twos = 1) :=
  sorry

end impossible_single_piece_on_center_l2696_269697


namespace snowfall_sum_l2696_269655

/-- The total snowfall recorded during a three-day snowstorm -/
def total_snowfall (wednesday thursday friday : ℝ) : ℝ :=
  wednesday + thursday + friday

/-- Proof that the total snowfall is 0.88 cm given the daily measurements -/
theorem snowfall_sum :
  total_snowfall 0.33 0.33 0.22 = 0.88 := by
  sorry

end snowfall_sum_l2696_269655


namespace sum_difference_remainder_mod_two_l2696_269648

theorem sum_difference_remainder_mod_two : 
  let n := 100
  let sum_remainder_one := (Finset.range n).sum (fun i => if i % 2 = 1 then i + 1 else 0)
  let sum_remainder_zero := (Finset.range n).sum (fun i => if i % 2 = 0 then i + 1 else 0)
  sum_remainder_zero - sum_remainder_one = 50 := by
sorry

end sum_difference_remainder_mod_two_l2696_269648


namespace inequality_proof_l2696_269654

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / b + 1 / c ≥ 9 / (a + b + c) := by
  sorry

end inequality_proof_l2696_269654


namespace max_prism_pyramid_elements_l2696_269649

/-- A shape formed by fusing a rectangular prism with a pyramid on one of its faces -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_new_faces : ℕ
  pyramid_new_edges : ℕ
  pyramid_new_vertex : ℕ

/-- The sum of exterior faces, vertices, and edges of a PrismPyramid -/
def total_elements (pp : PrismPyramid) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_new_faces) +
  (pp.prism_edges + pp.pyramid_new_edges) +
  (pp.prism_vertices + pp.pyramid_new_vertex)

/-- Theorem stating that the maximum sum of exterior faces, vertices, and edges is 34 -/
theorem max_prism_pyramid_elements :
  ∃ (pp : PrismPyramid), 
    pp.prism_faces = 6 ∧
    pp.prism_edges = 12 ∧
    pp.prism_vertices = 8 ∧
    pp.pyramid_new_faces = 4 ∧
    pp.pyramid_new_edges = 4 ∧
    pp.pyramid_new_vertex = 1 ∧
    total_elements pp = 34 ∧
    ∀ (pp' : PrismPyramid), total_elements pp' ≤ 34 :=
  sorry

end max_prism_pyramid_elements_l2696_269649


namespace withdrawal_theorem_l2696_269670

/-- Calculates the number of bills received when withdrawing from two banks -/
def number_of_bills (withdrawal_per_bank : ℕ) (bill_denomination : ℕ) : ℕ :=
  (2 * withdrawal_per_bank) / bill_denomination

/-- Theorem: Withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdrawal_theorem : number_of_bills 300 20 = 30 := by
  sorry

end withdrawal_theorem_l2696_269670


namespace function_symmetry_l2696_269659

/-- Given a polynomial function f(x) = ax^7 + bx^5 + cx^3 + dx + 5 where a, b, c, d are constants,
    if f(-7) = -7, then f(7) = 17 -/
theorem function_symmetry (a b c d : ℝ) :
  let f := fun x : ℝ => a * x^7 + b * x^5 + c * x^3 + d * x + 5
  (f (-7) = -7) → (f 7 = 17) := by
  sorry

end function_symmetry_l2696_269659


namespace cross_section_area_is_40_div_3_l2696_269601

/-- Right prism with isosceles triangle base -/
structure RightPrism where
  -- Base triangle
  AB : ℝ
  BC : ℝ
  angleABC : ℝ
  -- Intersection points
  AD_ratio : ℝ
  EC1_ratio : ℝ
  -- Conditions
  isIsosceles : AB = BC
  baseLength : AB = 5
  angleCondition : angleABC = 2 * Real.arcsin (3/5)
  adIntersection : AD_ratio = 1/3
  ec1Intersection : EC1_ratio = 1/3

/-- The area of the cross-section of the prism -/
def crossSectionArea (p : RightPrism) : ℝ :=
  sorry -- Actual calculation would go here

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_is_40_div_3 (p : RightPrism) :
  crossSectionArea p = 40/3 := by
  sorry

#check cross_section_area_is_40_div_3

end cross_section_area_is_40_div_3_l2696_269601


namespace perfect_squares_condition_l2696_269662

theorem perfect_squares_condition (n : ℕ) : 
  (∃ k m : ℕ, 2 * n + 1 = k^2 ∧ 3 * n + 1 = m^2) ↔ 
  (∃ a b : ℕ, n + 1 = a^2 + (a + 1)^2 ∧ ∃ c : ℕ, n + 1 = c^2 + 2 * (c + 1)^2) :=
by sorry

end perfect_squares_condition_l2696_269662


namespace fourth_ball_black_prob_l2696_269656

/-- A box containing colored balls. -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box. -/
def prob_black_ball (box : Box) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The theorem stating that the probability of the fourth ball being black
    is equal to the probability of selecting a black ball from the box. -/
theorem fourth_ball_black_prob (box : Box) (h1 : box.red_balls = 2) (h2 : box.black_balls = 5) :
  prob_black_ball box = 5 / 7 := by
  sorry

end fourth_ball_black_prob_l2696_269656


namespace df_length_is_six_l2696_269665

/-- Represents a triangle with side lengths and an angle --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle : ℝ

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Given two triangles ABC and DEF with the specified properties,
    prove that the length of DF is 6 cm --/
theorem df_length_is_six 
  (ABC : Triangle)
  (DEF : Triangle)
  (angle_relation : ABC.angle = 2 * DEF.angle)
  (ab_length : ABC.side1 = 4)
  (ac_length : ABC.side2 = 6)
  (de_length : DEF.side1 = 2)
  (perimeter_relation : perimeter ABC = 2 * perimeter DEF) :
  DEF.side2 = 6 := by
  sorry


end df_length_is_six_l2696_269665


namespace quadratic_function_properties_l2696_269695

def f (a c x : ℝ) : ℝ := x^2 - a*x + c

theorem quadratic_function_properties (a c : ℝ) :
  (∀ x, f a c x > 1 ↔ x < -1 ∨ x > 3) →
  (∀ x m, m^2 - 4*m < f a c (2^x)) →
  (∀ x₁ x₂, x₁ ∈ [-1, 5] → x₂ ∈ [-1, 5] → |f a c x₁ - f a c x₂| ≤ 10) →
  (a = 2 ∧ c = -2) ∧
  (∀ m, m > 1 ∧ m < 3) ∧
  (a ≥ 10 - 2*Real.sqrt 10 ∧ a ≤ -2 + 2*Real.sqrt 10) :=
by sorry

end quadratic_function_properties_l2696_269695


namespace danny_bottle_caps_l2696_269636

/-- Proves the number of bottle caps Danny found at the park -/
theorem danny_bottle_caps 
  (thrown_away : ℕ) 
  (current_total : ℕ) 
  (found_more_than_thrown : ℕ) : 
  thrown_away = 35 → 
  current_total = 22 → 
  found_more_than_thrown = 1 → 
  ∃ (previous_total : ℕ) (found : ℕ), 
    found = thrown_away + found_more_than_thrown ∧ 
    current_total = previous_total - thrown_away + found ∧
    found = 36 := by
  sorry

end danny_bottle_caps_l2696_269636


namespace lychee_production_increase_l2696_269691

theorem lychee_production_increase (x : ℝ) : 
  let increase_factor := 1 + x / 100
  let two_year_increase := increase_factor ^ 2 - 1
  two_year_increase = ((1 + x / 100) ^ 2 - 1) :=
by sorry

end lychee_production_increase_l2696_269691


namespace arithmetic_sequence_sum_l2696_269646

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_3 = 3 and S_6 = 7, prove S_9 = 12 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h3 : a.S 3 = 3) 
  (h6 : a.S 6 = 7) : 
  a.S 9 = 12 := by
sorry

end arithmetic_sequence_sum_l2696_269646


namespace solution_set_f_neg_x_l2696_269607

/-- Given a function f(x) = (ax-1)(x-b) where the solution set of f(x) > 0 is (-1,3),
    prove that the solution set of f(-x) < 0 is (-∞,-3)∪(1,+∞) -/
theorem solution_set_f_neg_x (a b : ℝ) : 
  (∀ x, (a * x - 1) * (x - b) > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, (a * (-x) - 1) * (-x - b) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end solution_set_f_neg_x_l2696_269607


namespace larger_number_problem_l2696_269622

theorem larger_number_problem (smaller larger : ℚ) : 
  smaller = 48 → 
  larger - smaller = (1 : ℚ) / 3 * larger →
  larger = 72 := by
sorry

end larger_number_problem_l2696_269622


namespace min_value_abc_min_value_abc_achieved_l2696_269687

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^3 * b^2 * c ≥ 64/729 := by
  sorry

theorem min_value_abc_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  1/a + 1/b + 1/c = 9 ∧
  a^3 * b^2 * c < 64/729 + ε := by
  sorry

end min_value_abc_min_value_abc_achieved_l2696_269687


namespace max_value_trig_expression_l2696_269641

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
  sorry

end max_value_trig_expression_l2696_269641


namespace parallel_vectors_x_value_l2696_269600

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 := by
sorry

end parallel_vectors_x_value_l2696_269600


namespace polygon_interior_angle_sum_l2696_269624

theorem polygon_interior_angle_sum (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 144 → 
  (n - 2) * 180 = n * interior_angle :=
by
  sorry

#check polygon_interior_angle_sum

end polygon_interior_angle_sum_l2696_269624


namespace log_product_equals_five_l2696_269678

theorem log_product_equals_five :
  (Real.log 4 / Real.log 2) * (Real.log 8 / Real.log 4) *
  (Real.log 16 / Real.log 8) * (Real.log 32 / Real.log 16) = 5 := by
  sorry

end log_product_equals_five_l2696_269678


namespace mikes_remaining_books_l2696_269647

theorem mikes_remaining_books (initial_books sold_books : ℕ) :
  initial_books = 51 →
  sold_books = 45 →
  initial_books - sold_books = 6 :=
by sorry

end mikes_remaining_books_l2696_269647


namespace three_day_trip_mileage_l2696_269692

theorem three_day_trip_mileage (total_miles : ℕ) (day1_miles : ℕ) (day2_miles : ℕ) 
  (h1 : total_miles = 493) 
  (h2 : day1_miles = 125) 
  (h3 : day2_miles = 223) : 
  total_miles - (day1_miles + day2_miles) = 145 := by
  sorry

end three_day_trip_mileage_l2696_269692


namespace disjoint_equal_sum_subsets_l2696_269633

theorem disjoint_equal_sum_subsets (S : Finset ℕ) 
  (h1 : S ⊆ Finset.range 2018)
  (h2 : S.card = 68) :
  ∃ (A B C : Finset ℕ), 
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    A.card = B.card ∧ B.card = C.card ∧
    A.sum id = B.sum id ∧ B.sum id = C.sum id :=
by sorry

end disjoint_equal_sum_subsets_l2696_269633


namespace theater_empty_showtime_l2696_269603

/-- Represents a theater --/
structure Theater :=
  (id : Nat)

/-- Represents a student --/
structure Student :=
  (id : Nat)

/-- Represents a showtime --/
structure Showtime :=
  (id : Nat)

/-- Represents the attendance of students at a theater for a specific showtime --/
def Attendance := Theater → Showtime → Finset Student

theorem theater_empty_showtime 
  (students : Finset Student) 
  (theaters : Finset Theater) 
  (showtimes : Finset Showtime) 
  (attendance : Attendance) :
  (students.card = 7) →
  (theaters.card = 7) →
  (showtimes.card = 8) →
  (∀ s : Showtime, ∃! t : Theater, (attendance t s).card = 6) →
  (∀ s : Showtime, ∃! t : Theater, (attendance t s).card = 1) →
  (∀ stud : Student, ∀ t : Theater, ∃ s : Showtime, stud ∈ attendance t s) →
  (∀ t : Theater, ∃ s : Showtime, (attendance t s).card = 0) :=
by sorry

end theater_empty_showtime_l2696_269603


namespace maplewood_elementary_difference_l2696_269616

theorem maplewood_elementary_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 20 →
  guinea_pigs_per_class = 3 →
  num_classes = 4 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 68 :=
by
  sorry

end maplewood_elementary_difference_l2696_269616


namespace min_value_problem_l2696_269651

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log x + Real.log y = 2) :
  5 * x + 2 * y ≥ 20 * Real.sqrt 10 := by
sorry

end min_value_problem_l2696_269651


namespace rabbit_walk_distance_l2696_269684

/-- The perimeter of a square park -/
def park_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- The theorem stating that a rabbit walking along the perimeter of a square park
    with a side length of 13 meters walks 52 meters -/
theorem rabbit_walk_distance : park_perimeter 13 = 52 := by
  sorry

end rabbit_walk_distance_l2696_269684


namespace floral_shop_sale_l2696_269696

/-- Represents the number of bouquets sold on each day of a three-day sale. -/
structure SaleData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Theorem stating the conditions of the sale and the result to be proven. -/
theorem floral_shop_sale (sale : SaleData) : 
  sale.tuesday = 3 * sale.monday ∧ 
  sale.wednesday = sale.tuesday / 3 ∧
  sale.monday + sale.tuesday + sale.wednesday = 60 →
  sale.monday = 12 := by
  sorry

end floral_shop_sale_l2696_269696


namespace g_is_even_symmetry_axes_increasing_function_l2696_269619

variable (f : ℝ → ℝ)

-- f is not constant
axiom not_constant : ∃ x y, f x ≠ f y

-- Definition of g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: g is even
theorem g_is_even : ∀ x, g f x = g f (-x) := by sorry

-- Definition of odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem 2: If f is odd and f(x) + f(2 + x) = 0, then f has axes of symmetry at x = 2n + 1
theorem symmetry_axes (h_odd : is_odd f) (h_sum : ∀ x, f x + f (2 + x) = 0) :
  ∀ n : ℤ, ∀ x : ℝ, f (2 * n + 1 + x) = -f (2 * n + 1 - x) := by sorry

-- Theorem 3: If (f(x₁) - f(x₂))/(x₁ - x₂) > 0 for x₁ ≠ x₂, then f is increasing
theorem increasing_function (h : ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) :
  ∀ x y, x < y → f x < f y := by sorry

end g_is_even_symmetry_axes_increasing_function_l2696_269619


namespace combined_savings_equal_individual_savings_l2696_269610

/-- Represents the number of windows in a bundle -/
def bundle_size : ℕ := 7

/-- Represents the number of windows paid for in a bundle -/
def paid_windows_per_bundle : ℕ := 5

/-- Represents the cost of a single window -/
def window_cost : ℕ := 100

/-- Calculates the number of bundles needed for a given number of windows -/
def bundles_needed (windows : ℕ) : ℕ :=
  (windows + bundle_size - 1) / bundle_size

/-- Calculates the cost of windows with the promotion -/
def promotional_cost (windows : ℕ) : ℕ :=
  bundles_needed windows * paid_windows_per_bundle * window_cost

/-- Calculates the savings for a given number of windows -/
def savings (windows : ℕ) : ℕ :=
  windows * window_cost - promotional_cost windows

/-- Dave's required number of windows -/
def dave_windows : ℕ := 12

/-- Doug's required number of windows -/
def doug_windows : ℕ := 10

theorem combined_savings_equal_individual_savings :
  savings (dave_windows + doug_windows) = savings dave_windows + savings doug_windows :=
by sorry

end combined_savings_equal_individual_savings_l2696_269610


namespace angle_symmetry_l2696_269625

theorem angle_symmetry (α : Real) : 
  (∃ k : ℤ, α = π/3 + 2*k*π) →  -- Condition 1 (symmetry implies α = π/3 + 2kπ)
  α ∈ Set.Ioo (-4*π) (-2*π) →   -- Condition 2
  (α = -11*π/3 ∨ α = -5*π/3) :=
by sorry

end angle_symmetry_l2696_269625


namespace tiles_needed_is_100_l2696_269640

/-- Calculates the number of tiles needed to cover a rectangular room with a central pillar -/
def calculate_tiles (room_length room_width pillar_side border_tile_side central_tile_side : ℕ) : ℕ :=
  let border_tiles := 2 * room_width
  let central_area := room_length * (room_width - 2) - pillar_side^2
  let central_tiles := (central_area + central_tile_side^2 - 1) / central_tile_side^2
  border_tiles + central_tiles

/-- The total number of tiles needed for the specific room configuration is 100 -/
theorem tiles_needed_is_100 : calculate_tiles 30 20 2 1 3 = 100 := by sorry

end tiles_needed_is_100_l2696_269640


namespace used_books_count_l2696_269680

def total_books : ℕ := 30
def new_books : ℕ := 15

theorem used_books_count : total_books - new_books = 15 := by
  sorry

end used_books_count_l2696_269680


namespace race_catchup_time_l2696_269645

theorem race_catchup_time 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (head_start : ℝ) 
  (h1 : cristina_speed = 6)
  (h2 : nicky_speed = 3)
  (h3 : head_start = 36) :
  ∃ t : ℝ, t = 12 ∧ cristina_speed * t = head_start + nicky_speed * t := by
sorry

end race_catchup_time_l2696_269645


namespace sqrt_neg_one_is_plus_minus_i_l2696_269643

theorem sqrt_neg_one_is_plus_minus_i :
  ∃ (z : ℂ), z * z = -1 ∧ (z = Complex.I ∨ z = -Complex.I) :=
by sorry

end sqrt_neg_one_is_plus_minus_i_l2696_269643
