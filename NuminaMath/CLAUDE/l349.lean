import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_n_l349_34946

/-- Given two natural numbers m and n, where mn = 34^8 and m has a units digit of 4,
    prove that the units digit of n is 4. -/
theorem units_digit_of_n (m n : ℕ) 
  (h1 : m * n = 34^8)
  (h2 : m % 10 = 4) : 
  n % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l349_34946


namespace NUMINAMATH_CALUDE_f_properties_l349_34999

noncomputable def f (x : ℝ) := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) - 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi/3) (Real.pi/12), ∀ y ∈ Set.Icc (-Real.pi/3) (Real.pi/12), x ≤ y → f y ≤ f x) ∧
  (∀ x ∈ Set.Icc (Real.pi/12) (Real.pi/3), ∀ y ∈ Set.Icc (Real.pi/12) (Real.pi/3), x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l349_34999


namespace NUMINAMATH_CALUDE_base_12_addition_l349_34987

/-- Addition in base 12 --/
def base_12_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 12 --/
def to_base_12 (n : ℕ) : ℕ := sorry

/-- Conversion from base 12 to base 10 --/
def from_base_12 (n : ℕ) : ℕ := sorry

theorem base_12_addition :
  base_12_add (from_base_12 528) (from_base_12 274) = to_base_12 940 :=
sorry

end NUMINAMATH_CALUDE_base_12_addition_l349_34987


namespace NUMINAMATH_CALUDE_inequality_system_solution_l349_34997

theorem inequality_system_solution :
  ∀ x : ℝ, (x - 7 < 5 * (x - 1) ∧ 4/3 * x + 3 ≥ 1 - 2/3 * x) ↔ x > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l349_34997


namespace NUMINAMATH_CALUDE_cubic_function_properties_l349_34949

-- Define the cubic function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem stating the properties of f(x)
theorem cubic_function_properties :
  (∀ x, f' x = 0 ↔ x = 1 ∨ x = -1) ∧
  f (-2) = -4 ∧
  f (-1) = 0 ∧
  f 1 = -4 ∧
  (∀ x, x < -1 → f' x > 0) ∧
  (∀ x, x > 1 → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l349_34949


namespace NUMINAMATH_CALUDE_money_distribution_l349_34958

theorem money_distribution (a b c total : ℕ) : 
  (a + b + c = total) →  -- total is the sum of all shares
  (2 * b = 3 * a) →      -- ratio between a and b is 2:3
  (3 * c = 4 * b) →      -- ratio between b and c is 3:4
  (b = 1800) →           -- b's share is $1800
  (total = 5400) :=      -- prove that total is $5400
by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l349_34958


namespace NUMINAMATH_CALUDE_michaels_weight_loss_l349_34991

/-- Michael's weight loss problem -/
theorem michaels_weight_loss 
  (total_goal : ℝ) 
  (april_loss : ℝ) 
  (may_goal : ℝ) 
  (h1 : total_goal = 10) 
  (h2 : april_loss = 4) 
  (h3 : may_goal = 3) : 
  total_goal - (april_loss + may_goal) = 3 := by
sorry

end NUMINAMATH_CALUDE_michaels_weight_loss_l349_34991


namespace NUMINAMATH_CALUDE_product_of_roots_l349_34988

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → 
  ∃ x₁ x₂ : ℝ, x₁ * x₂ = -34 ∧ (x₁ + 3) * (x₁ - 4) = 22 ∧ (x₂ + 3) * (x₂ - 4) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l349_34988


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l349_34983

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l349_34983


namespace NUMINAMATH_CALUDE_triangle_perimeter_l349_34935

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define the properties of the triangle
def TriangleProperties (A B C : ℝ × ℝ) :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  BC = AC - 1 ∧ AC = AB - 1 ∧ (AB^2 + AC^2 - BC^2) / (2 * AB * AC) = 3/5

-- Theorem statement
theorem triangle_perimeter (A B C : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hp : TriangleProperties A B C) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB + BC + AC = 42 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l349_34935


namespace NUMINAMATH_CALUDE_heechul_most_books_l349_34995

/-- The number of books each person has -/
structure BookCollection where
  heejin : ℕ
  heechul : ℕ
  dongkyun : ℕ

/-- Conditions of the book collection -/
def valid_collection (bc : BookCollection) : Prop :=
  bc.heechul = bc.heejin + 2 ∧ bc.dongkyun < bc.heejin

/-- Heechul has the most books -/
def heechul_has_most (bc : BookCollection) : Prop :=
  bc.heechul > bc.heejin ∧ bc.heechul > bc.dongkyun

/-- Theorem: If the collection is valid, then Heechul has the most books -/
theorem heechul_most_books (bc : BookCollection) :
  valid_collection bc → heechul_has_most bc := by
  sorry

end NUMINAMATH_CALUDE_heechul_most_books_l349_34995


namespace NUMINAMATH_CALUDE_roberto_outfits_l349_34957

/-- The number of different outfits that can be created from a given number of trousers, shirts, and jackets. -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ :=
  trousers * shirts * jackets

/-- Theorem stating that with 5 trousers, 6 shirts, and 4 jackets, the number of possible outfits is 120. -/
theorem roberto_outfits :
  number_of_outfits 5 6 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l349_34957


namespace NUMINAMATH_CALUDE_functional_equation_problem_l349_34971

/-- A function satisfying f(a+b) = f(a)f(b) for all a and b -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b, f (a + b) = f a * f b

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) 
  (h2 : f 1 = 2) : 
  (f 1)^2 / f 1 + f 2 / f 1 + 
  (f 2)^2 / f 3 + f 4 / f 3 + 
  (f 3)^2 / f 5 + f 6 / f 5 + 
  (f 4)^2 / f 7 + f 8 / f 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l349_34971


namespace NUMINAMATH_CALUDE_tino_jellybean_count_l349_34916

/-- The number of jellybeans each person has. -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem. -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.tino = j.lee + 24 ∧ 
  j.arnold * 2 = j.lee ∧ 
  j.arnold = 5

/-- The theorem stating that Tino has 34 jellybeans under the given conditions. -/
theorem tino_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.tino = 34 := by
  sorry

end NUMINAMATH_CALUDE_tino_jellybean_count_l349_34916


namespace NUMINAMATH_CALUDE_abs_diff_inequality_l349_34927

theorem abs_diff_inequality (x : ℝ) : |x + 3| - |x - 1| > 0 ↔ x > -1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_inequality_l349_34927


namespace NUMINAMATH_CALUDE_fraction_simplification_l349_34986

variable (x : ℝ)

theorem fraction_simplification :
  (x^3 + 4*x^2 + 7*x + 4) / (x^3 + 2*x^2 + x - 4) = (x + 1) / (x - 1) ∧
  2 * (24*x^3 + 46*x^2 + 33*x + 9) / (24*x^3 + 10*x^2 - 9*x - 9) = (4*x + 3) / (4*x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l349_34986


namespace NUMINAMATH_CALUDE_apartment_number_exists_unique_l349_34978

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def contains_digit (n d : ℕ) : Prop :=
  ∃ k m, n = 100 * k + 10 * m + d ∧ 0 ≤ k ∧ k < 10 ∧ 0 ≤ m ∧ m < 10

theorem apartment_number_exists_unique :
  ∃! n : ℕ, is_three_digit n ∧
            n % 11 = 0 ∧
            n % 2 = 0 ∧
            n % 5 = 0 ∧
            ¬ contains_digit n 7 :=
sorry

end NUMINAMATH_CALUDE_apartment_number_exists_unique_l349_34978


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l349_34952

theorem greatest_common_divisor_under_60 : 
  ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 540 ∧ 
  n < 60 ∧ 
  n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 → m < 60 → m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l349_34952


namespace NUMINAMATH_CALUDE_chess_game_probability_l349_34984

/-- The probability of a chess game resulting in a draw -/
def prob_draw : ℚ := 1/2

/-- The probability of player A winning the chess game -/
def prob_a_win : ℚ := 1/3

/-- The probability of player A not losing the chess game -/
def prob_a_not_lose : ℚ := prob_draw + prob_a_win

theorem chess_game_probability : prob_a_not_lose = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l349_34984


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l349_34966

theorem geometric_progression_problem :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (b / a = c / b) ∧
    (a + b + c = 65) ∧
    (a * b * c = 3375) ∧
    ((a = 5 ∧ b = 15 ∧ c = 45) ∨ (a = 45 ∧ b = 15 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l349_34966


namespace NUMINAMATH_CALUDE_allstar_seating_arrangements_l349_34961

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_allstars : ℕ := 9
def num_cubs : ℕ := 3
def num_redsox : ℕ := 3
def num_yankees : ℕ := 2
def num_dodgers : ℕ := 1
def num_teams : ℕ := 4

theorem allstar_seating_arrangements :
  (factorial num_teams) * (factorial num_cubs) * (factorial num_redsox) * 
  (factorial num_yankees) * (factorial num_dodgers) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_allstar_seating_arrangements_l349_34961


namespace NUMINAMATH_CALUDE_prob_odd_1_49_is_0_12_l349_34910

/-- The total number of balls in the box -/
def total_balls : ℕ := 200

/-- The number of odd-numbered balls in the range 1-49 -/
def odd_balls_1_49 : ℕ := 24

/-- The probability of selecting an odd-numbered ball within the range of 1-49 -/
def prob_odd_1_49 : ℚ := odd_balls_1_49 / total_balls

/-- The conditions for the ball selection -/
structure BallSelection where
  num_selected : ℕ
  at_least_three_odd : Bool
  at_least_one_50_100 : Bool
  no_consecutive : Bool

/-- The theorem stating the probability of selecting an odd-numbered ball within the range of 1-49 -/
theorem prob_odd_1_49_is_0_12 (selection : BallSelection) :
  selection.num_selected = 5 →
  selection.at_least_three_odd = true →
  selection.at_least_one_50_100 = true →
  selection.no_consecutive = true →
  prob_odd_1_49 = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_1_49_is_0_12_l349_34910


namespace NUMINAMATH_CALUDE_pioneer_assignment_l349_34940

structure Pioneer where
  lastName : String
  firstName : String
  age : Nat

def Burov : Pioneer := sorry
def Gridnev : Pioneer := sorry
def Klimenko : Pioneer := sorry

axiom burov_not_kolya : Burov.firstName ≠ "Kolya"
axiom petya_school_start : ∃ p : Pioneer, p.firstName = "Petya" ∧ p.age = 12
axiom gridnev_grisha_older : Gridnev.age = (Klimenko.age + 1) ∧ Burov.age = (Klimenko.age + 1)

theorem pioneer_assignment :
  (Burov.firstName = "Grisha" ∧ Burov.age = 13) ∧
  (Gridnev.firstName = "Kolya" ∧ Gridnev.age = 13) ∧
  (Klimenko.firstName = "Petya" ∧ Klimenko.age = 12) :=
sorry

end NUMINAMATH_CALUDE_pioneer_assignment_l349_34940


namespace NUMINAMATH_CALUDE_hotel_flat_fee_l349_34956

/-- Given a hotel's pricing structure and two customer payments, prove the flat fee for the first night. -/
theorem hotel_flat_fee (f n : ℝ) 
  (ann_payment : f + n = 120)
  (bob_payment : f + 6 * n = 330) :
  f = 78 := by sorry

end NUMINAMATH_CALUDE_hotel_flat_fee_l349_34956


namespace NUMINAMATH_CALUDE_lost_people_problem_l349_34909

/-- Calculates the number of people in the second group of lost people --/
def second_group_size (initial_group : ℕ) (initial_days : ℕ) (days_passed : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_food := initial_group * initial_days
  let remaining_food := total_food - (initial_group * days_passed)
  let total_people := remaining_food / remaining_days
  total_people - initial_group

/-- Theorem stating that given the problem conditions, the second group has 3 people --/
theorem lost_people_problem :
  second_group_size 9 5 1 3 = 3 := by
  sorry


end NUMINAMATH_CALUDE_lost_people_problem_l349_34909


namespace NUMINAMATH_CALUDE_point_on_modified_graph_l349_34904

/-- Given a function g : ℝ → ℝ where (3, 9) is on its graph,
    prove that (1, 10) is on the graph of 3y = 4g(3x) - 6 -/
theorem point_on_modified_graph (g : ℝ → ℝ) (h : g 3 = 9) :
  3 * 10 = 4 * g (3 * 1) - 6 := by
  sorry

end NUMINAMATH_CALUDE_point_on_modified_graph_l349_34904


namespace NUMINAMATH_CALUDE_n_mod_nine_eq_six_l349_34954

/-- The sum of specific numbers -/
def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

/-- Theorem stating that n mod 9 equals 6 -/
theorem n_mod_nine_eq_six : n % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_n_mod_nine_eq_six_l349_34954


namespace NUMINAMATH_CALUDE_total_flowers_l349_34989

theorem total_flowers (total_vases : Nat) (vases_with_five : Nat) (flowers_in_four : Nat) (flowers_in_one : Nat) : 
  total_vases = 5 → vases_with_five = 4 → flowers_in_four = 5 → flowers_in_one = 6 → 
  vases_with_five * flowers_in_four + (total_vases - vases_with_five) * flowers_in_one = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l349_34989


namespace NUMINAMATH_CALUDE_feeding_sequence_count_l349_34905

/-- Represents the number of animal pairs in the safari park -/
def num_pairs : ℕ := 5

/-- Calculates the number of possible feeding sequences -/
def feeding_sequences : ℕ := 
  (num_pairs)  -- choices for first female
  * (num_pairs - 1)  -- choices for second male
  * (num_pairs - 1)  -- choices for second female
  * (num_pairs - 2)  -- choices for third male
  * (num_pairs - 2)  -- choices for third female
  * (num_pairs - 3)  -- choices for fourth male
  * (num_pairs - 3)  -- choices for fourth female
  * (num_pairs - 4)  -- choices for fifth male
  * (num_pairs - 4)  -- choices for fifth female

theorem feeding_sequence_count : feeding_sequences = 2880 := by
  sorry

end NUMINAMATH_CALUDE_feeding_sequence_count_l349_34905


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l349_34941

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_diff : a 3 - 3 * a 2 = 2)
  (h_mean : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l349_34941


namespace NUMINAMATH_CALUDE_absolute_value_reciprocal_graph_l349_34965

theorem absolute_value_reciprocal_graph (x : ℝ) (x_nonzero : x ≠ 0) :
  (1 / |x|) = if x > 0 then 1 / x else -1 / x :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_reciprocal_graph_l349_34965


namespace NUMINAMATH_CALUDE_solution_characterization_l349_34933

/-- Represents a 3-digit integer abc --/
structure ThreeDigitInt where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a < 10
  h3 : b < 10
  h4 : c < 10

/-- Converts a ThreeDigitInt to its numerical value --/
def toInt (n : ThreeDigitInt) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Checks if a ThreeDigitInt satisfies the given equation --/
def satisfiesEquation (n : ThreeDigitInt) : Prop :=
  n.b * (10 * n.a + n.c) = n.c * (10 * n.a + n.b) + 10

/-- The set of all ThreeDigitInt that satisfy the equation --/
def solutionSet : Set ThreeDigitInt :=
  {n : ThreeDigitInt | satisfiesEquation n}

/-- The theorem to be proved --/
theorem solution_characterization :
  solutionSet = {
    ⟨1, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 2, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 3, 4, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 4, 5, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 5, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 6, 7, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 7, 8, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 8, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩
  } := by sorry

#eval toInt ⟨1, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 2, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 3, 4, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 4, 5, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 5, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 6, 7, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 7, 8, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 8, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_solution_characterization_l349_34933


namespace NUMINAMATH_CALUDE_min_students_in_class_l349_34929

theorem min_students_in_class (b g : ℕ) : 
  b = 2 * g →  -- ratio of boys to girls is 2:1
  (3 * b) / 5 = (5 * g) / 8 →  -- number of boys who passed equals number of girls who passed
  b + g ≥ 120 ∧ ∀ n < 120, ¬(∃ b' g', b' = 2 * g' ∧ (3 * b') / 5 = (5 * g') / 8 ∧ b' + g' = n) :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_class_l349_34929


namespace NUMINAMATH_CALUDE_speeding_percentage_l349_34964

/-- The percentage of motorists who exceed the speed limit and receive tickets -/
def ticketed_speeders : ℝ := 20

/-- The percentage of speeding motorists who do not receive tickets -/
def unticketed_speeder_percentage : ℝ := 20

/-- The total percentage of motorists who exceed the speed limit -/
def total_speeders : ℝ := 25

theorem speeding_percentage :
  ticketed_speeders * (100 - unticketed_speeder_percentage) / 100 = total_speeders * (100 - unticketed_speeder_percentage) / 100 := by
  sorry

end NUMINAMATH_CALUDE_speeding_percentage_l349_34964


namespace NUMINAMATH_CALUDE_division_of_powers_l349_34985

theorem division_of_powers (x : ℝ) (h : x ≠ 0) : (-6 * x^3) / (-2 * x^2) = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l349_34985


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l349_34914

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem f_max_min_on_interval :
  let a := -3
  let b := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 18 ∧ f x_min = -18 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l349_34914


namespace NUMINAMATH_CALUDE_special_function_omega_value_l349_34981

/-- A function f with the properties described in the problem -/
structure SpecialFunction (ω : ℝ) where
  f : ℝ → ℝ
  eq : ∀ x, f x = 3 * Real.sin (ω * x + π / 3)
  positive_ω : ω > 0
  equal_values : f (π / 6) = f (π / 3)
  min_no_max : ∃ x₀ ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x₀ ≤ f x
             ∧ ¬∃ x₁ ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x ≤ f x₁

/-- The main theorem stating that ω must be 14/3 -/
theorem special_function_omega_value {ω : ℝ} (sf : SpecialFunction ω) : ω = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_omega_value_l349_34981


namespace NUMINAMATH_CALUDE_red_subset_existence_l349_34982

theorem red_subset_existence (n k m : ℕ) (X : Finset ℕ) 
  (red_subsets : Finset (Finset ℕ)) :
  n > 0 → k > 0 → k < n →
  Finset.card X = n →
  (∀ A ∈ red_subsets, Finset.card A = k ∧ A ⊆ X) →
  Finset.card red_subsets = m →
  m > ((k - 1) * (n - k) + k) / (k^2 : ℚ) * (Nat.choose n (k - 1)) →
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Finset.card Y = k + 1 ∧
    ∀ Z : Finset ℕ, Z ⊆ Y → Finset.card Z = k → Z ∈ red_subsets :=
by sorry


end NUMINAMATH_CALUDE_red_subset_existence_l349_34982


namespace NUMINAMATH_CALUDE_ball_drawing_probability_l349_34934

theorem ball_drawing_probability : 
  let total_balls : ℕ := 25
  let black_balls : ℕ := 10
  let white_balls : ℕ := 10
  let red_balls : ℕ := 5
  let drawn_balls : ℕ := 4

  let probability : ℚ := 
    (Nat.choose black_balls 2 * Nat.choose white_balls 2 + 
     Nat.choose black_balls 2 * Nat.choose red_balls 2 + 
     Nat.choose white_balls 2 * Nat.choose red_balls 2) / 
    Nat.choose total_balls drawn_balls

  probability = 195 / 841 := by
sorry

end NUMINAMATH_CALUDE_ball_drawing_probability_l349_34934


namespace NUMINAMATH_CALUDE_lottery_probability_l349_34953

theorem lottery_probability 
  (sharpBallCount : Nat) 
  (prizeBallCount : Nat) 
  (prizeBallsDrawn : Nat) 
  (h1 : sharpBallCount = 30)
  (h2 : prizeBallCount = 50)
  (h3 : prizeBallsDrawn = 6) :
  (1 : ℚ) / sharpBallCount * (1 : ℚ) / Nat.choose prizeBallCount prizeBallsDrawn = 1 / 476721000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l349_34953


namespace NUMINAMATH_CALUDE_baseball_cards_count_l349_34907

theorem baseball_cards_count (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 87)
  (h2 : additional_cards = 13) :
  initial_cards + additional_cards = 100 :=
by sorry

end NUMINAMATH_CALUDE_baseball_cards_count_l349_34907


namespace NUMINAMATH_CALUDE_unique_solution_for_all_real_b_l349_34903

/-- The equation has exactly one real solution in x for all real b, 
    unless b forces other roots or violates the unimodality of solution. -/
theorem unique_solution_for_all_real_b : 
  ∀ b : ℝ, ∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_real_b_l349_34903


namespace NUMINAMATH_CALUDE_maria_boxes_count_l349_34924

def eggs_per_box : ℕ := 7
def total_eggs : ℕ := 21

theorem maria_boxes_count : 
  total_eggs / eggs_per_box = 3 := by sorry

end NUMINAMATH_CALUDE_maria_boxes_count_l349_34924


namespace NUMINAMATH_CALUDE_sum_equals_zero_l349_34948

theorem sum_equals_zero : 1 + 1 - 2 + 3 + 5 - 8 + 13 + 21 - 34 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l349_34948


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l349_34993

/-- The probability of selecting two non-defective pens without replacement from a box of 12 pens, where 3 are defective, is 6/11. -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 3) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 6 / 11 := by
  sorry

#check prob_two_non_defective_pens

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l349_34993


namespace NUMINAMATH_CALUDE_divisibility_proof_l349_34928

theorem divisibility_proof (n : ℕ) (x : ℝ) :
  ∃ P : ℝ → ℝ, (x + 1)^(2*n) - x^(2*n) - 2*x - 1 = x * (x + 1) * (2*x + 1) * P x :=
by sorry

end NUMINAMATH_CALUDE_divisibility_proof_l349_34928


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l349_34996

/-- Given a man and his son, where the man is 34 years older than his son
    and the son's current age is 32, proves that the ratio of their ages
    in two years is 2:1. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
  son_age = 32 →
  man_age = son_age + 34 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l349_34996


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l349_34973

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), (x = 6 + (182 : ℚ) / 999) ∧ (1000 * x - x = 6182 - 6) :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l349_34973


namespace NUMINAMATH_CALUDE_square_root_identity_specific_square_roots_l349_34945

theorem square_root_identity (n : ℕ) :
  Real.sqrt (1 - (2 * n + 1) / ((n + 1) ^ 2)) = n / (n + 1) :=
sorry

theorem specific_square_roots :
  Real.sqrt (1 - 9 / 25) = 4 / 5 ∧ Real.sqrt (1 - 15 / 64) = 7 / 8 :=
sorry

end NUMINAMATH_CALUDE_square_root_identity_specific_square_roots_l349_34945


namespace NUMINAMATH_CALUDE_percent_less_than_l349_34994

theorem percent_less_than (N M : ℝ) (h1 : 0 < N) (h2 : 0 < M) (h3 : N < M) :
  (M - N) / M * 100 = 100 * (1 - N / M) := by
  sorry

end NUMINAMATH_CALUDE_percent_less_than_l349_34994


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l349_34900

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 0, p x) ↔ ∀ x > 0, ¬ p x := by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 0, Real.log x > x - 1) ↔ (∀ x > 0, Real.log x ≤ x - 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l349_34900


namespace NUMINAMATH_CALUDE_estimate_event_knowledge_chengdu_games_knowledge_estimate_l349_34944

/-- Estimates the number of people in a population who know about an event,
    given a sample survey result. -/
theorem estimate_event_knowledge (total_population : ℕ) 
                                  (sample_size : ℕ) 
                                  (sample_positive : ℕ) : ℕ :=
  let estimate := (sample_positive * total_population) / sample_size
  estimate

/-- Proves that the estimated number of people who know about the event
    in a population of 10,000, given 125 out of 200 know in a sample, is 6250. -/
theorem chengdu_games_knowledge_estimate :
  estimate_event_knowledge 10000 200 125 = 6250 := by
  sorry

end NUMINAMATH_CALUDE_estimate_event_knowledge_chengdu_games_knowledge_estimate_l349_34944


namespace NUMINAMATH_CALUDE_polygon_properties_l349_34915

theorem polygon_properties (n : ℕ) (internal_angle : ℝ) (external_angle : ℝ) :
  (internal_angle + external_angle = 180) →
  (external_angle = (2/3) * internal_angle) →
  (360 / external_angle = n) →
  (n > 2) →
  (n = 5 ∧ (n - 2) * 180 = 540) :=
by sorry

end NUMINAMATH_CALUDE_polygon_properties_l349_34915


namespace NUMINAMATH_CALUDE_sum_not_exceeding_eight_probability_most_probable_sum_probability_of_most_probable_sum_l349_34920

def ball_count : ℕ := 8

def ball_labels : Finset ℕ := Finset.range ball_count

def sum_of_pair (i j : ℕ) : ℕ := i + j

def valid_pairs : Finset (ℕ × ℕ) :=
  (ball_labels.product ball_labels).filter (λ p => p.1 < p.2)

def pairs_with_sum (n : ℕ) : Finset (ℕ × ℕ) :=
  valid_pairs.filter (λ p => sum_of_pair p.1 p.2 = n)

def probability (favorable : Finset (ℕ × ℕ)) : ℚ :=
  favorable.card / valid_pairs.card

theorem sum_not_exceeding_eight_probability :
  probability (valid_pairs.filter (λ p => sum_of_pair p.1 p.2 ≤ 8)) = 3/7 := by sorry

theorem most_probable_sum :
  ∃ n : ℕ, n = 9 ∧ 
    ∀ m : ℕ, probability (pairs_with_sum n) ≥ probability (pairs_with_sum m) := by sorry

theorem probability_of_most_probable_sum :
  probability (pairs_with_sum 9) = 1/7 := by sorry

end NUMINAMATH_CALUDE_sum_not_exceeding_eight_probability_most_probable_sum_probability_of_most_probable_sum_l349_34920


namespace NUMINAMATH_CALUDE_divisible_by_ten_l349_34921

theorem divisible_by_ten : ∃ k : ℤ, 43^43 - 17^17 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l349_34921


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_2718_and_gcd_l349_34980

theorem largest_four_digit_divisible_by_2718_and_gcd : ∃ (n : ℕ), n ≤ 9999 ∧ n % 2718 = 0 ∧ 
  (∀ m : ℕ, m ≤ 9999 ∧ m % 2718 = 0 → m ≤ n) ∧
  n = 8154 ∧
  Nat.gcd n 8640 = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_2718_and_gcd_l349_34980


namespace NUMINAMATH_CALUDE_combine_terms_implies_zero_sum_l349_34972

theorem combine_terms_implies_zero_sum (a b x y : ℝ) : 
  (∃ k : ℝ, -3 * a^(2*x-1) * b = k * 5 * a * b^(y+4)) → 
  (x - 2)^2016 + (y + 2)^2017 = 0 := by
sorry

end NUMINAMATH_CALUDE_combine_terms_implies_zero_sum_l349_34972


namespace NUMINAMATH_CALUDE_area_T_prime_l349_34998

/-- A transformation matrix -/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, 5]

/-- The area of the original region T -/
def area_T : ℝ := 9

/-- The theorem stating the area of the transformed region T' -/
theorem area_T_prime : 
  let det_A := Matrix.det A
  area_T * det_A = 207 := by sorry

end NUMINAMATH_CALUDE_area_T_prime_l349_34998


namespace NUMINAMATH_CALUDE_x_squared_plus_k_factorization_l349_34950

theorem x_squared_plus_k_factorization (k : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + k = (x - a) * (x - b)) ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_k_factorization_l349_34950


namespace NUMINAMATH_CALUDE_small_kite_area_l349_34967

/-- A kite is defined by its four vertices -/
structure Kite where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a kite -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The specific kite from the problem -/
def smallKite : Kite :=
  { v1 := (3, 0)
    v2 := (0, 5)
    v3 := (3, 7)
    v4 := (6, 5) }

/-- Theorem stating that the area of the small kite is 21 square inches -/
theorem small_kite_area : kiteArea smallKite = 21 := by sorry

end NUMINAMATH_CALUDE_small_kite_area_l349_34967


namespace NUMINAMATH_CALUDE_sampledInInterval_eq_three_l349_34968

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalPopulation : ℕ
  sampleSize : ℕ
  intervalStart : ℕ
  intervalEnd : ℕ

/-- Calculates the number of sampled individuals within a given interval -/
def sampledInInterval (s : SystematicSampling) : ℕ :=
  let stride := s.totalPopulation / s.sampleSize
  let firstSample := s.intervalStart + (stride - s.intervalStart % stride) % stride
  if firstSample > s.intervalEnd then 0
  else (s.intervalEnd - firstSample) / stride + 1

/-- Theorem stating that for the given systematic sampling scenario, 
    the number of sampled individuals in the interval [61, 120] is 3 -/
theorem sampledInInterval_eq_three :
  let s : SystematicSampling := {
    totalPopulation := 840,
    sampleSize := 42,
    intervalStart := 61,
    intervalEnd := 120
  }
  sampledInInterval s = 3 := by sorry

end NUMINAMATH_CALUDE_sampledInInterval_eq_three_l349_34968


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l349_34908

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det ((B ^ 2) - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l349_34908


namespace NUMINAMATH_CALUDE_lyka_initial_money_l349_34962

/-- Calculates the initial amount of money Lyka has given the cost of a smartphone,
    the saving period in weeks, and the weekly saving rate. -/
def initial_money (smartphone_cost : ℕ) (saving_period : ℕ) (saving_rate : ℕ) : ℕ :=
  smartphone_cost - saving_period * saving_rate

/-- Proves that given a smartphone cost of $160, a saving period of 8 weeks,
    and a saving rate of $15 per week, the initial amount of money Lyka has is $40. -/
theorem lyka_initial_money :
  initial_money 160 8 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lyka_initial_money_l349_34962


namespace NUMINAMATH_CALUDE_speed_of_sound_calculation_l349_34911

-- Define the time between hearing the blasts in seconds
def time_between_hearing : ℝ := 30 * 60 + 20

-- Define the time between the blasts occurring in seconds
def time_between_blasts : ℝ := 30 * 60

-- Define the distance traveled by the sound of the second blast in meters
def distance : ℝ := 6600

-- Define the speed of sound in meters per second
def speed_of_sound : ℝ := 330

-- Theorem statement
theorem speed_of_sound_calculation :
  speed_of_sound = distance / (time_between_hearing - time_between_blasts) := by
  sorry

end NUMINAMATH_CALUDE_speed_of_sound_calculation_l349_34911


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l349_34902

theorem quadratic_is_square_of_binomial (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 - 25 * x + 9 = (r * x + s)^2) →
  a = 625 / 36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l349_34902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_property_l349_34963

/-- Given an arithmetic sequence {a_n} with common difference d (d ≠ 0) and sum of first n terms S_n,
    if {√(S_n + n)} is also an arithmetic sequence with common difference d, then d = 1/2. -/
theorem arithmetic_sequence_special_property (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) - a n = d) ∧
  (∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * d) ∧
  (∀ n : ℕ, Real.sqrt (S (n + 1) + (n + 1)) - Real.sqrt (S n + n) = d) →
  d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_property_l349_34963


namespace NUMINAMATH_CALUDE_positive_real_solution_l349_34926

theorem positive_real_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : 3 * Real.sqrt (x^2 + x) + 3 * Real.sqrt (x^2 - x) = 6 * Real.sqrt 2) : 
  x = 4 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_l349_34926


namespace NUMINAMATH_CALUDE_fraction_addition_l349_34939

theorem fraction_addition : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l349_34939


namespace NUMINAMATH_CALUDE_race_total_length_l349_34942

/-- The total length of a race with four parts -/
def race_length (part1 part2 part3 part4 : ℝ) : ℝ :=
  part1 + part2 + part3 + part4

theorem race_total_length :
  race_length 15.5 21.5 21.5 16 = 74.5 := by
  sorry

end NUMINAMATH_CALUDE_race_total_length_l349_34942


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l349_34925

theorem hyperbola_eccentricity (m : ℝ) :
  (∃ e : ℝ, e > Real.sqrt 2 ∧
    ∀ x y : ℝ, x^2 - y^2/m = 1 → e = Real.sqrt (1 + m)) ↔
  m > 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l349_34925


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l349_34947

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 6 ∧ 3 * x * y = 6 →
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) ∧
    a + b + c + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l349_34947


namespace NUMINAMATH_CALUDE_dhoni_leftover_percentage_l349_34937

/-- Represents the percentage of Dhoni's earnings spent on rent -/
def rent_percentage : ℝ := 20

/-- Represents the difference in percentage between rent and dishwasher expenses -/
def dishwasher_difference : ℝ := 5

/-- Calculates the percentage of earnings spent on the dishwasher -/
def dishwasher_percentage : ℝ := rent_percentage - dishwasher_difference

/-- Calculates the total percentage of earnings spent -/
def total_spent_percentage : ℝ := rent_percentage + dishwasher_percentage

/-- Represents the total percentage (100%) -/
def total_percentage : ℝ := 100

/-- Theorem: The percentage of Dhoni's earning left over is 65% -/
theorem dhoni_leftover_percentage : 
  total_percentage - total_spent_percentage = 65 := by
  sorry

end NUMINAMATH_CALUDE_dhoni_leftover_percentage_l349_34937


namespace NUMINAMATH_CALUDE_circle_center_coordinates_sum_l349_34918

theorem circle_center_coordinates_sum (x y : ℝ) : 
  x^2 + y^2 - 12*x + 10*y = 40 → (x - 6)^2 + (y + 5)^2 = 101 ∧ x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_sum_l349_34918


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l349_34919

theorem four_digit_number_with_specific_remainders :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 131 = 112 ∧ n % 132 = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l349_34919


namespace NUMINAMATH_CALUDE_profit_percentage_approx_l349_34969

/-- Calculates the profit percentage for a given purchase and sale scenario. -/
def profit_percentage (items_bought : ℕ) (price_paid : ℕ) (discount : ℚ) : ℚ :=
  let cost := price_paid
  let selling_price := (items_bought : ℚ) * (1 - discount)
  let profit := selling_price - (cost : ℚ)
  (profit / cost) * 100

/-- Theorem stating that the profit percentage for the given scenario is approximately 11.91%. -/
theorem profit_percentage_approx (ε : ℚ) (h_ε : ε > 0) :
  ∃ δ : ℚ, abs (profit_percentage 52 46 (1/100) - 911/7650) < ε :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_approx_l349_34969


namespace NUMINAMATH_CALUDE_rectangle_area_l349_34970

/-- Given a rectangle with width 42 inches, where ten such rectangles placed end to end
    reach a length of 390 inches, prove that its area is 1638 square inches. -/
theorem rectangle_area (width : ℝ) (total_length : ℝ) (h1 : width = 42)
    (h2 : total_length = 390) : width * (total_length / 10) = 1638 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l349_34970


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l349_34951

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 3 and a₂ + a₃ = 6, 
    prove that the 7th term a₇ = 64. -/
theorem geometric_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 2 + a 3 = 6) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) : 
  a 7 = 64 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l349_34951


namespace NUMINAMATH_CALUDE_infinite_square_root_equals_three_l349_34955

theorem infinite_square_root_equals_three :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (3 + 2 * x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_square_root_equals_three_l349_34955


namespace NUMINAMATH_CALUDE_locker_count_l349_34901

/-- The cost of a single digit in dollars -/
def digit_cost : ℚ := 0.03

/-- The total cost of labeling all lockers in dollars -/
def total_cost : ℚ := 206.91

/-- Calculates the cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let one_digit := min n 9
  let two_digit := min (n - 9) 90
  let three_digit := min (n - 99) 900
  let four_digit := max (n - 999) 0
  digit_cost * (one_digit + 2 * two_digit + 3 * three_digit + 4 * four_digit)

/-- The theorem stating that 2001 lockers can be labeled with the given total cost -/
theorem locker_count : labeling_cost 2001 = total_cost := by
  sorry

end NUMINAMATH_CALUDE_locker_count_l349_34901


namespace NUMINAMATH_CALUDE_root_triple_relation_l349_34975

theorem root_triple_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_root_triple_relation_l349_34975


namespace NUMINAMATH_CALUDE_horner_method_eval_l349_34932

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem horner_method_eval :
  f 4 = horner_eval [3, -2, 5, -2.5, 1.5, -0.7] 4 ∧
  horner_eval [3, -2, 5, -2.5, 1.5, -0.7] 4 = 2845.3 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_eval_l349_34932


namespace NUMINAMATH_CALUDE_parallel_condition_l349_34917

/-- Two lines are parallel if and only if they have the same slope -/
def parallel (m1 a1 b1 : ℝ) (m2 a2 b2 : ℝ) : Prop :=
  m1 = m2

/-- The line l1 with equation ax + 2y - 3 = 0 -/
def l1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y - 3 = 0

/-- The line l2 with equation 2x + y - a = 0 -/
def l2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x + y - a = 0

/-- The statement that a = 4 is a necessary and sufficient condition for l1 to be parallel to l2 -/
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, parallel (-a/2) 0 0 (-2) 0 0) ↔ a = 4 :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l349_34917


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l349_34938

/-- Proves that Maxwell walks for 2 hours before meeting Brad -/
theorem maxwell_brad_meeting_time
  (distance : ℝ)
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (head_start : ℝ)
  (h_distance : distance = 14)
  (h_maxwell_speed : maxwell_speed = 4)
  (h_brad_speed : brad_speed = 6)
  (h_head_start : head_start = 1) :
  ∃ (t : ℝ), t + head_start = 2 ∧ maxwell_speed * (t + head_start) + brad_speed * t = distance :=
by sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l349_34938


namespace NUMINAMATH_CALUDE_circle_theorem_l349_34960

/-- Circle C -/
def circle_C (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

/-- Circle D -/
def circle_D (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 1)^2 = 16

/-- Line l -/
def line_l (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

/-- Circles C and D are externally tangent -/
def externally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y m ∧ circle_D x y

theorem circle_theorem :
  ∀ m : ℝ,
  (∀ x y : ℝ, circle_C x y m → m < 5) ∧
  (externally_tangent m → m = 4) ∧
  (m = 4 →
    ∃ chord_length : ℝ,
      chord_length = 4 * Real.sqrt 5 / 5 ∧
      ∀ x y : ℝ,
        circle_C x y m ∧ line_l x y →
        ∃ x' y' : ℝ,
          circle_C x' y' m ∧ line_l x' y' ∧
          (x - x')^2 + (y - y')^2 = chord_length^2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_theorem_l349_34960


namespace NUMINAMATH_CALUDE_platform_length_calculation_l349_34930

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a post, prove that the length of the platform is 350 meters. -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_post : ℝ)
    (h1 : train_length = 300)
    (h2 : time_platform = 39)
    (h3 : time_post = 18) :
    let train_speed := train_length / time_post
    let platform_length := train_speed * time_platform - train_length
    platform_length = 350 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l349_34930


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l349_34990

/-- The trajectory of the center of a moving circle that is externally tangent to two fixed circles -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ (r : ℝ), 
    -- First fixed circle
    (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 + 4*x₁ + 3 = 0 ∧ 
      -- Moving circle is externally tangent to the first fixed circle
      (x - x₁)^2 + (y - y₁)^2 = (r + 1)^2) ∧ 
    -- Second fixed circle
    (∃ (x₂ y₂ : ℝ), x₂^2 + y₂^2 - 4*x₂ - 5 = 0 ∧ 
      -- Moving circle is externally tangent to the second fixed circle
      (x - x₂)^2 + (y - y₂)^2 = (r + 3)^2)) →
  -- The trajectory of the center of the moving circle
  x^2 - 3*y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l349_34990


namespace NUMINAMATH_CALUDE_sum_digits_1944_base9_l349_34936

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits of 1944 in base 9 is 8 -/
theorem sum_digits_1944_base9 : sumDigits (toBase9 1944) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_1944_base9_l349_34936


namespace NUMINAMATH_CALUDE_aeroplane_transaction_loss_l349_34923

theorem aeroplane_transaction_loss : 
  let selling_price : ℝ := 600
  let profit_percentage : ℝ := 0.2
  let loss_percentage : ℝ := 0.2
  let cost_price_profit : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_loss : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price_profit + cost_price_loss
  let total_revenue : ℝ := 2 * selling_price
  total_cost - total_revenue = 50 := by
sorry


end NUMINAMATH_CALUDE_aeroplane_transaction_loss_l349_34923


namespace NUMINAMATH_CALUDE_school_total_students_l349_34977

/-- The total number of students in a school with a given number of grades and students per grade -/
def total_students (num_grades : ℕ) (students_per_grade : ℕ) : ℕ :=
  num_grades * students_per_grade

/-- Theorem stating that the total number of students in a school with 304 grades and 75 students per grade is 22800 -/
theorem school_total_students :
  total_students 304 75 = 22800 := by
  sorry

end NUMINAMATH_CALUDE_school_total_students_l349_34977


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l349_34931

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l349_34931


namespace NUMINAMATH_CALUDE_floor_equality_implies_abs_diff_less_than_one_l349_34974

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_equality_implies_abs_diff_less_than_one (x y : ℝ) :
  floor x = floor y → |x - y| < 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_implies_abs_diff_less_than_one_l349_34974


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l349_34922

theorem student_multiplication_problem (x : ℝ) : 40 * x - 138 = 102 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l349_34922


namespace NUMINAMATH_CALUDE_derivative_at_two_l349_34913

/-- Given a function f(x) = ax³ + bx² + 3 where b = f'(2), prove that if f'(1) = -5, then f'(2) = -5 -/
theorem derivative_at_two (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + b * x^2 + 3)
  (h2 : b = (deriv f) 2) (h3 : (deriv f) 1 = -5) : (deriv f) 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l349_34913


namespace NUMINAMATH_CALUDE_box_packing_problem_l349_34906

theorem box_packing_problem (n : ℕ) (h : n = 301) :
  (n % 3 = 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1) ∧
  (∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n % k = 0) :=
by sorry

end NUMINAMATH_CALUDE_box_packing_problem_l349_34906


namespace NUMINAMATH_CALUDE_find_number_l349_34959

theorem find_number : ∃ x : ℚ, x = 15 ∧ (4/5) * x + 20 = (80/100) * 40 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l349_34959


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l349_34992

theorem nested_fraction_equality : 
  1 / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l349_34992


namespace NUMINAMATH_CALUDE_picnic_problem_l349_34943

theorem picnic_problem (total : ℕ) (men_excess : ℕ) (adult_excess : ℕ) :
  total = 240 →
  men_excess = 80 →
  adult_excess = 80 →
  ∃ (men women adults children : ℕ),
    men = women + men_excess ∧
    adults = children + adult_excess ∧
    men + women = adults ∧
    adults + children = total ∧
    men = 120 := by
  sorry

end NUMINAMATH_CALUDE_picnic_problem_l349_34943


namespace NUMINAMATH_CALUDE_point_C_coordinates_l349_34979

/-- Given points A(-1,2) and B(2,8), if vector AB = 3 * vector AC, 
    then C has coordinates (0,4) -/
theorem point_C_coordinates 
  (A B C : ℝ × ℝ) 
  (h1 : A = (-1, 2)) 
  (h2 : B = (2, 8)) 
  (h3 : B - A = 3 • (C - A)) : 
  C = (0, 4) := by
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l349_34979


namespace NUMINAMATH_CALUDE_max_value_of_f_on_S_l349_34912

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the constraint set
def S : Set ℝ := {x : ℝ | x^4 + 36 ≤ 13*x^2}

-- Theorem statement
theorem max_value_of_f_on_S :
  ∃ (M : ℝ), M = 18 ∧ ∀ x ∈ S, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_S_l349_34912


namespace NUMINAMATH_CALUDE_cost_reduction_per_meter_l349_34976

/-- Proves that the reduction in cost per meter is 1 Rs -/
theorem cost_reduction_per_meter
  (original_cost : ℝ)
  (original_length : ℝ)
  (new_length : ℝ)
  (h_original_cost : original_cost = 35)
  (h_original_length : original_length = 10)
  (h_new_length : new_length = 14)
  (h_total_cost_unchanged : original_cost = new_length * (original_cost / original_length - x))
  : x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_reduction_per_meter_l349_34976
