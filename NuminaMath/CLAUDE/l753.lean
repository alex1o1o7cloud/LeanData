import Mathlib

namespace NUMINAMATH_CALUDE_solution_numbers_l753_75378

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem solution_numbers : 
  {n : ℕ | n + sumOfDigits n = 2021} = {2014, 1996} := by sorry

end NUMINAMATH_CALUDE_solution_numbers_l753_75378


namespace NUMINAMATH_CALUDE_die_roll_average_l753_75382

def die_rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]
def next_roll : Nat := 2
def total_rolls : Nat := die_rolls.length + 1

theorem die_roll_average :
  (die_rolls.sum + next_roll) / total_rolls = 3 := by
sorry

end NUMINAMATH_CALUDE_die_roll_average_l753_75382


namespace NUMINAMATH_CALUDE_total_pages_bought_l753_75364

def total_spent : ℚ := 10
def cost_per_notepad : ℚ := 5/4  -- $1.25 expressed as a rational number
def pages_per_notepad : ℕ := 60

theorem total_pages_bought : ℕ := by
  -- Proof goes here
  sorry

#check total_pages_bought = 480

end NUMINAMATH_CALUDE_total_pages_bought_l753_75364


namespace NUMINAMATH_CALUDE_jane_circle_impossibility_l753_75357

theorem jane_circle_impossibility : ¬ ∃ (a : Fin 2024 → ℕ+),
  (∀ i : Fin 2024, ∃ j : Fin 2024, a i * a (i + 1) = Nat.factorial (j + 1)) ∧
  (∀ k : Fin 2024, ∃ i : Fin 2024, a i * a (i + 1) = Nat.factorial (k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_jane_circle_impossibility_l753_75357


namespace NUMINAMATH_CALUDE_unique_a_value_l753_75379

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

theorem unique_a_value : 
  ∃! a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ (Set.univ \ B)) ∧ 
            (∀ x : ℝ, x ∈ (Set.univ \ B) → a - 1 < x ∧ x < a + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l753_75379


namespace NUMINAMATH_CALUDE_three_digit_base_nine_to_base_three_digit_count_sum_l753_75389

theorem three_digit_base_nine_to_base_three_digit_count_sum :
  ∀ n : ℕ,
  (3^4 ≤ n ∧ n < 3^6) →
  (∃ e : ℕ, (3^(e-1) ≤ n ∧ n < 3^e) ∧ (e = 5 ∨ e = 6 ∨ e = 7)) ∧
  (5 + 6 + 7 = 18) :=
sorry

end NUMINAMATH_CALUDE_three_digit_base_nine_to_base_three_digit_count_sum_l753_75389


namespace NUMINAMATH_CALUDE_rationality_of_expressions_l753_75318

theorem rationality_of_expressions :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt (Real.exp 2) = p / q) ∧
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ (0.64 : ℝ) ^ (1/3) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (0.0256 : ℝ) ^ (1/4) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-8 : ℝ) ^ (1/3) * Real.sqrt ((0.25 : ℝ)⁻¹) = p / q) :=
by sorry

end NUMINAMATH_CALUDE_rationality_of_expressions_l753_75318


namespace NUMINAMATH_CALUDE_problem_solution_l753_75381

/-- Represents a three-digit number in the form abc --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem problem_solution :
  ∀ (a b : Nat),
  let n1 := ThreeDigitNumber.mk 3 a 7 (by sorry)
  let n2 := ThreeDigitNumber.mk 6 b 1 (by sorry)
  (n1.toNat + 294 = n2.toNat) →
  (n2.toNat % 7 = 0) →
  a + b = 8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l753_75381


namespace NUMINAMATH_CALUDE_ln_equation_solution_l753_75324

theorem ln_equation_solution :
  ∃ y : ℝ, (Real.log y - 3 * Real.log 2 = -1) ∧ (abs (y - 2.94) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ln_equation_solution_l753_75324


namespace NUMINAMATH_CALUDE_sequence_property_l753_75351

def sequence_a (n : ℕ) : ℕ := sorry

theorem sequence_property : 
  ∃ (b c d : ℤ), 
    (∀ n : ℕ, n > 0 → sequence_a n = b * Int.floor (Real.sqrt (n + c)) + d) ∧ 
    sequence_a 1 = 1 ∧ 
    b + c + d = 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l753_75351


namespace NUMINAMATH_CALUDE_tan_cot_15_sum_even_l753_75336

theorem tan_cot_15_sum_even (n : ℕ+) : 
  ∃ k : ℤ, (2 - Real.sqrt 3) ^ n.val + (2 + Real.sqrt 3) ^ n.val = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_tan_cot_15_sum_even_l753_75336


namespace NUMINAMATH_CALUDE_min_n_for_inequality_l753_75342

theorem min_n_for_inequality : ∃ (n : ℕ),
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), x^2 + y^2 + z^2 > m * (x^4 + y^4 + z^4)) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_inequality_l753_75342


namespace NUMINAMATH_CALUDE_disk_ratio_theorem_l753_75374

/-- Represents a disk with a center point and a radius. -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two disks are tangent to each other. -/
def areTangent (d1 d2 : Disk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 = (d1.radius + d2.radius)^2

/-- Checks if two disks have disjoint interiors. -/
def haveDisjointInteriors (d1 d2 : Disk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 > (d1.radius + d2.radius)^2

theorem disk_ratio_theorem (d1 d2 d3 d4 : Disk) 
  (h_equal_size : d1.radius = d2.radius ∧ d2.radius = d3.radius)
  (h_smaller : d4.radius < d1.radius)
  (h_tangent : areTangent d1 d2 ∧ areTangent d2 d3 ∧ areTangent d3 d1 ∧ 
               areTangent d1 d4 ∧ areTangent d2 d4 ∧ areTangent d3 d4)
  (h_disjoint : haveDisjointInteriors d1 d2 ∧ haveDisjointInteriors d2 d3 ∧ 
                haveDisjointInteriors d3 d1 ∧ haveDisjointInteriors d1 d4 ∧ 
                haveDisjointInteriors d2 d4 ∧ haveDisjointInteriors d3 d4) :
  d4.radius / d1.radius = (2 * Real.sqrt 3 - 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_disk_ratio_theorem_l753_75374


namespace NUMINAMATH_CALUDE_power_function_through_2_4_l753_75341

/-- A power function passing through the point (2, 4) has exponent 2 -/
theorem power_function_through_2_4 :
  ∀ a : ℝ, (2 : ℝ) ^ a = 4 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_2_4_l753_75341


namespace NUMINAMATH_CALUDE_blue_marbles_difference_marble_difference_theorem_l753_75367

/-- Represents the number of marbles in each color for a jar -/
structure JarContents where
  blue : ℚ
  green : ℚ

/-- Represents the contents of both jars -/
structure TwoJars where
  jar1 : JarContents
  jar2 : JarContents

def blue_green_ratio (jar : JarContents) : ℚ :=
  jar.blue / jar.green

theorem blue_marbles_difference (jars : TwoJars) : ℚ :=
  jars.jar1.blue - jars.jar2.blue

/-- The main theorem to be proved -/
theorem marble_difference_theorem (jars : TwoJars) : 
    jars.jar1.blue + jars.jar1.green = jars.jar2.blue + jars.jar2.green →
    blue_green_ratio jars.jar1 = 7/3 →
    blue_green_ratio jars.jar2 = 5/1 →
    jars.jar1.green + jars.jar2.green = 120 →
    blue_marbles_difference jars = -34 := by
  sorry


end NUMINAMATH_CALUDE_blue_marbles_difference_marble_difference_theorem_l753_75367


namespace NUMINAMATH_CALUDE_peters_newspaper_delivery_l753_75391

/-- Peter's newspaper delivery problem -/
theorem peters_newspaper_delivery :
  let total_weekend := 110
  let saturday := 45
  let sunday := 65
  sunday > saturday →
  sunday - saturday = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_newspaper_delivery_l753_75391


namespace NUMINAMATH_CALUDE_classroom_ratio_l753_75343

theorem classroom_ratio :
  ∀ (boys girls : ℕ),
  boys + girls = 36 →
  boys = girls + 6 →
  (boys : ℚ) / girls = 7 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l753_75343


namespace NUMINAMATH_CALUDE_puppy_weight_is_2_5_l753_75386

/-- The weight of the puppy in pounds -/
def puppy_weight : ℝ := 2.5

/-- The weight of the smaller cat in pounds -/
def smaller_cat_weight : ℝ := 7.5

/-- The weight of the larger cat in pounds -/
def larger_cat_weight : ℝ := 20

/-- Theorem stating that the weight of the puppy is 2.5 pounds given the conditions -/
theorem puppy_weight_is_2_5 :
  (puppy_weight + smaller_cat_weight + larger_cat_weight = 30) ∧
  (puppy_weight + larger_cat_weight = 3 * smaller_cat_weight) ∧
  (puppy_weight + smaller_cat_weight = larger_cat_weight - 10) →
  puppy_weight = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_is_2_5_l753_75386


namespace NUMINAMATH_CALUDE_family_park_cost_l753_75326

/-- Calculates the total cost for a family to visit a park and one attraction -/
def total_cost (park_fee : ℕ) (child_attraction_fee : ℕ) (adult_attraction_fee : ℕ) 
                (num_children : ℕ) (num_parents : ℕ) (num_grandparents : ℕ) : ℕ :=
  let total_family_members := num_children + num_parents + num_grandparents
  let total_adults := num_parents + num_grandparents
  let park_cost := total_family_members * park_fee
  let children_attraction_cost := num_children * child_attraction_fee
  let adult_attraction_cost := total_adults * adult_attraction_fee
  park_cost + children_attraction_cost + adult_attraction_cost

/-- Theorem: The total cost for the specified family composition is $55 -/
theorem family_park_cost : 
  total_cost 5 2 4 4 2 1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_family_park_cost_l753_75326


namespace NUMINAMATH_CALUDE_smallest_n_for_milly_victory_l753_75360

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ a b c d : ℕ, a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n →
    (coloring a = coloring b ∧ coloring b = coloring c ∧ coloring c = coloring d) →
    a + b + c ≠ d

theorem smallest_n_for_milly_victory : 
  (∀ n < 11, ∃ coloring : ℕ → Bool, is_valid_coloring n coloring) ∧
  (∀ coloring : ℕ → Bool, ¬ is_valid_coloring 11 coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_milly_victory_l753_75360


namespace NUMINAMATH_CALUDE_product_of_positive_reals_l753_75334

theorem product_of_positive_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 24 * (3 ^ (1/4)))
  (h2 : x * z = 42 * (3 ^ (1/4)))
  (h3 : y * z = 21 * (3 ^ (1/4))) :
  x * y * z = Real.sqrt 63504 := by
sorry

end NUMINAMATH_CALUDE_product_of_positive_reals_l753_75334


namespace NUMINAMATH_CALUDE_product_of_sums_geq_product_l753_75302

theorem product_of_sums_geq_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_of_sums_geq_product_l753_75302


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l753_75373

theorem y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l753_75373


namespace NUMINAMATH_CALUDE_even_product_probability_spinners_l753_75383

/-- Represents a spinner with sections labeled by natural numbers -/
structure Spinner :=
  (sections : List ℕ)

/-- The probability of getting an even product when spinning two spinners -/
def evenProductProbability (spinnerA spinnerB : Spinner) : ℚ :=
  sorry

/-- Spinner A with 6 equal sections: 1, 1, 2, 2, 3, 4 -/
def spinnerA : Spinner :=
  ⟨[1, 1, 2, 2, 3, 4]⟩

/-- Spinner B with 4 equal sections: 1, 3, 5, 6 -/
def spinnerB : Spinner :=
  ⟨[1, 3, 5, 6]⟩

/-- Theorem stating that the probability of getting an even product
    when spinning spinnerA and spinnerB is 5/8 -/
theorem even_product_probability_spinners :
  evenProductProbability spinnerA spinnerB = 5/8 :=
sorry

end NUMINAMATH_CALUDE_even_product_probability_spinners_l753_75383


namespace NUMINAMATH_CALUDE_selection_theorem_l753_75358

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of male students --/
def num_males : ℕ := 5

/-- The number of female students --/
def num_females : ℕ := 4

/-- The total number of representatives to be selected --/
def total_representatives : ℕ := 4

/-- The number of ways to select representatives satisfying the given conditions --/
def num_ways : ℕ := 
  choose num_males 2 * choose num_females 2 + 
  choose num_males 3 * choose num_females 1

theorem selection_theorem : num_ways = 100 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l753_75358


namespace NUMINAMATH_CALUDE_tons_approximation_l753_75371

/-- Two real numbers are approximately equal if their absolute difference is less than 0.5 -/
def approximately_equal (x y : ℝ) : Prop := |x - y| < 0.5

/-- 1 ton is defined as 1000 kilograms -/
def ton : ℝ := 1000

theorem tons_approximation : approximately_equal (29.6 * ton) (30 * ton) := by sorry

end NUMINAMATH_CALUDE_tons_approximation_l753_75371


namespace NUMINAMATH_CALUDE_football_cost_l753_75307

/-- The cost of a football given the total amount paid, change received, and cost of a baseball. -/
theorem football_cost (total_paid : ℝ) (change : ℝ) (baseball_cost : ℝ) 
  (h1 : total_paid = 20)
  (h2 : change = 4.05)
  (h3 : baseball_cost = 6.81) : 
  total_paid - change - baseball_cost = 9.14 := by
  sorry

#check football_cost

end NUMINAMATH_CALUDE_football_cost_l753_75307


namespace NUMINAMATH_CALUDE_product_125_sum_31_l753_75330

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 125 → 
  a + b + c = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l753_75330


namespace NUMINAMATH_CALUDE_fraction_simplification_l753_75301

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l753_75301


namespace NUMINAMATH_CALUDE_complex_fourth_power_l753_75300

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l753_75300


namespace NUMINAMATH_CALUDE_total_legs_in_collection_l753_75332

/-- The number of legs for a spider -/
def spider_legs : ℕ := 8

/-- The number of legs for an ant -/
def ant_legs : ℕ := 6

/-- The number of spiders in the collection -/
def num_spiders : ℕ := 8

/-- The number of ants in the collection -/
def num_ants : ℕ := 12

/-- Theorem stating that the total number of legs in the collection is 136 -/
theorem total_legs_in_collection : 
  num_spiders * spider_legs + num_ants * ant_legs = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_collection_l753_75332


namespace NUMINAMATH_CALUDE_mrs_sheridans_cats_l753_75344

theorem mrs_sheridans_cats (initial_cats additional_cats : ℕ) :
  initial_cats = 17 →
  additional_cats = 14 →
  initial_cats + additional_cats = 31 :=
by sorry

end NUMINAMATH_CALUDE_mrs_sheridans_cats_l753_75344


namespace NUMINAMATH_CALUDE_min_value_of_function_l753_75327

theorem min_value_of_function (x : ℝ) (h : x > 10) : x^2 / (x - 10) ≥ 40 ∧ ∃ y > 10, y^2 / (y - 10) = 40 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l753_75327


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l753_75395

theorem positive_integer_solutions_count : 
  (Finset.filter (fun (x : ℕ × ℕ × ℕ × ℕ) => x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.range 10))))).card = 84 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l753_75395


namespace NUMINAMATH_CALUDE_sin_balanceable_same_balancing_pair_for_square_and_exp_cos_squared_balancing_pair_range_l753_75345

/-- A function f is balanceable if there exist real numbers m and k (m ≠ 0) such that
    m * f x = f (x + k) + f (x - k) for all x in the domain of f. -/
def Balanceable (f : ℝ → ℝ) : Prop :=
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

/-- A balancing pair for a function f is a pair (m, k) that satisfies the balanceable condition. -/
def BalancingPair (f : ℝ → ℝ) (m k : ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

theorem sin_balanceable :
  ∃ n : ℤ, BalancingPair Real.sin 1 (2 * π * n + π / 3) ∨ BalancingPair Real.sin 1 (2 * π * n - π / 3) :=
sorry

theorem same_balancing_pair_for_square_and_exp :
  ∀ a : ℝ, a ≠ 0 →
  (BalancingPair (fun x ↦ x^2) 2 0 ∧ BalancingPair (fun x ↦ a + 2^x) 2 0) :=
sorry

theorem cos_squared_balancing_pair_range :
  ∃ m₁ m₂ : ℝ,
  BalancingPair (fun x ↦ Real.cos x ^ 2) m₁ (π / 2) ∧
  BalancingPair (fun x ↦ Real.cos x ^ 2) m₂ (π / 4) ∧
  ∀ x, 0 ≤ x ∧ x ≤ π / 4 → 1 ≤ m₁^2 + m₂^2 ∧ m₁^2 + m₂^2 ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_sin_balanceable_same_balancing_pair_for_square_and_exp_cos_squared_balancing_pair_range_l753_75345


namespace NUMINAMATH_CALUDE_boss_contribution_l753_75372

def gift_cost : ℝ := 100
def employee_contribution : ℝ := 11
def num_employees : ℕ := 5

theorem boss_contribution :
  ∃ (boss_amount : ℝ),
    boss_amount = 15 ∧
    ∃ (todd_amount : ℝ),
      todd_amount = 2 * boss_amount ∧
      boss_amount + todd_amount + (num_employees : ℝ) * employee_contribution = gift_cost :=
by sorry

end NUMINAMATH_CALUDE_boss_contribution_l753_75372


namespace NUMINAMATH_CALUDE_probability_three_players_complete_theorem_l753_75337

/-- The probability that after N matches in a 5-player round-robin tournament,
    there are at least 3 players who have played all their matches against each other. -/
def probability_three_players_complete (N : ℕ) : ℚ :=
  if N < 3 then 0
  else if N = 3 then 1/12
  else if N = 4 then 1/3
  else if N = 5 then 5/7
  else if N = 6 then 20/21
  else 1

/-- The theorem stating the probability of having at least 3 players who have played
    all their matches against each other after N matches in a 5-player round-robin tournament. -/
theorem probability_three_players_complete_theorem (N : ℕ) (h : 3 ≤ N ∧ N ≤ 10) :
  probability_three_players_complete N =
    if N < 3 then 0
    else if N = 3 then 1/12
    else if N = 4 then 1/3
    else if N = 5 then 5/7
    else if N = 6 then 20/21
    else 1 := by sorry

end NUMINAMATH_CALUDE_probability_three_players_complete_theorem_l753_75337


namespace NUMINAMATH_CALUDE_remainder_3m_mod_5_l753_75366

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3m_mod_5_l753_75366


namespace NUMINAMATH_CALUDE_fishing_line_sections_l753_75322

theorem fishing_line_sections (num_reels : ℕ) (reel_length : ℕ) (section_length : ℕ) : 
  num_reels = 3 → reel_length = 100 → section_length = 10 → 
  (num_reels * reel_length) / section_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_fishing_line_sections_l753_75322


namespace NUMINAMATH_CALUDE_sum_xyz_equals_twenty_ninths_l753_75312

theorem sum_xyz_equals_twenty_ninths 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + 4*y^2 + 9*z^2 = 4)
  (eq2 : 2*x + 4*y + 3*z = 6) :
  x + y + z = 20/9 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_twenty_ninths_l753_75312


namespace NUMINAMATH_CALUDE_bayonet_third_draw_probability_l753_75384

/-- Represents the total number of bulbs in the box -/
def total_bulbs : ℕ := 10

/-- Represents the number of screw base bulbs in the box -/
def screw_bulbs : ℕ := 3

/-- Represents the number of bayonet base bulbs in the box -/
def bayonet_bulbs : ℕ := 7

/-- Represents the probability of selecting a bayonet base bulb on the third draw without replacement -/
def prob_bayonet_third_draw : ℚ := 7 / 120

theorem bayonet_third_draw_probability :
  (screw_bulbs / total_bulbs) * ((screw_bulbs - 1) / (total_bulbs - 1)) * (bayonet_bulbs / (total_bulbs - 2)) = prob_bayonet_third_draw :=
by sorry

end NUMINAMATH_CALUDE_bayonet_third_draw_probability_l753_75384


namespace NUMINAMATH_CALUDE_work_completion_theorem_l753_75316

theorem work_completion_theorem (work : ℕ) (days1 days2 men1 : ℕ) 
  (h1 : work = men1 * days1)
  (h2 : work = 24 * (work / (men1 * days1) * men1))
  (h3 : men1 = 16)
  (h4 : days1 = 30)
  : work / (men1 * days1) * men1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l753_75316


namespace NUMINAMATH_CALUDE_total_students_l753_75394

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 300) :
  boys + girls = 780 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l753_75394


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l753_75361

theorem max_value_x_sqrt_1_minus_4x_squared (x : ℝ) :
  0 < x → x < 1/2 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l753_75361


namespace NUMINAMATH_CALUDE_divisibility_property_l753_75331

theorem divisibility_property (q : ℕ) (h1 : q > 1) (h2 : Odd q) :
  ∃ k : ℕ, (q + 1) ^ ((q + 1) / 2) = (q + 1) * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l753_75331


namespace NUMINAMATH_CALUDE_arithmetic_seq_bicolored_l753_75362

/-- A coloring function for natural numbers -/
def coloring (n : ℕ) : Bool :=
  let segment := (Nat.sqrt (8 * n + 1) - 1) / 2
  segment % 2 = 0

/-- Definition of an arithmetic sequence -/
def isArithmeticSeq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + r

/-- Theorem stating that every infinite arithmetic sequence is bi-colored -/
theorem arithmetic_seq_bicolored :
  ∀ (a : ℕ → ℕ) (r : ℕ), isArithmeticSeq a r →
  (∃ k, coloring (a k) = true) ∧ (∃ m, coloring (a m) = false) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_seq_bicolored_l753_75362


namespace NUMINAMATH_CALUDE_absolute_difference_sequence_l753_75390

/-- Given three non-negative real numbers x, y, z, where z = 1, and after n steps of taking pairwise
    absolute differences, the sequence stabilizes with x_n = x, y_n = y, z_n = z, 
    then (x, y) = (0, 1) or (1, 0). -/
theorem absolute_difference_sequence (x y z : ℝ) (n : ℕ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z = 1 →
  (∃ (x_seq y_seq z_seq : ℕ → ℝ),
    (∀ k, k < n → 
      x_seq (k+1) = |x_seq k - y_seq k| ∧
      y_seq (k+1) = |y_seq k - z_seq k| ∧
      z_seq (k+1) = |z_seq k - x_seq k|) ∧
    x_seq 0 = x ∧ y_seq 0 = y ∧ z_seq 0 = z ∧
    x_seq n = x ∧ y_seq n = y ∧ z_seq n = z) →
  (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_absolute_difference_sequence_l753_75390


namespace NUMINAMATH_CALUDE_james_age_l753_75393

theorem james_age (dan_age james_age : ℕ) : 
  (dan_age : ℚ) / james_age = 6 / 5 →
  dan_age + 4 = 28 →
  james_age = 20 := by
sorry

end NUMINAMATH_CALUDE_james_age_l753_75393


namespace NUMINAMATH_CALUDE_polygon_perimeter_sum_tan_greater_than_x_l753_75359

theorem polygon_perimeter_sum (R : ℝ) (h : R > 0) :
  let n : ℕ := 1985
  let θ : ℝ := 2 * Real.pi / n
  let inner_side := 2 * R * Real.sin (θ / 2)
  let outer_side := 2 * R * Real.tan (θ / 2)
  let inner_perimeter := n * inner_side
  let outer_perimeter := n * outer_side
  inner_perimeter + outer_perimeter ≥ 4 * Real.pi * R :=
by
  sorry

theorem tan_greater_than_x (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  Real.tan x ≥ x :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_perimeter_sum_tan_greater_than_x_l753_75359


namespace NUMINAMATH_CALUDE_minimum_bookmarks_l753_75310

def is_divisible_by (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem minimum_bookmarks : 
  ∀ (n : ℕ), n > 0 → 
  (is_divisible_by n 3 ∧ 
   is_divisible_by n 4 ∧ 
   is_divisible_by n 5 ∧ 
   is_divisible_by n 7 ∧ 
   is_divisible_by n 8) → 
  n ≥ 840 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_bookmarks_l753_75310


namespace NUMINAMATH_CALUDE_geometric_sum_example_l753_75369

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Proof that the sum of the first 8 terms of the geometric sequence
    with first term 1/3 and common ratio 1/3 is 9840/19683 -/
theorem geometric_sum_example :
  geometric_sum (1/3) (1/3) 8 = 9840/19683 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l753_75369


namespace NUMINAMATH_CALUDE_xy_power_2018_l753_75380

theorem xy_power_2018 (x y : ℝ) (h : |x - 1/2| + (y + 2)^2 = 0) : (x*y)^2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_power_2018_l753_75380


namespace NUMINAMATH_CALUDE_sophie_germain_prime_units_digit_l753_75328

/-- A positive prime number p is a Sophie Germain prime if 2p + 1 is also prime. -/
def SophieGermainPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem sophie_germain_prime_units_digit (p : ℕ) (h : SophieGermainPrime p) (h_gt : p > 6) :
  unitsDigit p = 1 ∨ unitsDigit p = 3 :=
sorry

end NUMINAMATH_CALUDE_sophie_germain_prime_units_digit_l753_75328


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l753_75346

/-- Given a geometric sequence with common ratio 2, prove that S_4 / a_1 = 15 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  (a 0 * (1 - 2^4)) / (a 0 * (1 - 2)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l753_75346


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_count_l753_75339

/-- The number of ways to choose 4 points from 10 distinct points on a circle's circumference to form convex quadrilaterals -/
def convex_quadrilaterals_from_circle_points (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Theorem stating that the number of convex quadrilaterals formed from 10 points on a circle is 210 -/
theorem convex_quadrilaterals_count :
  convex_quadrilaterals_from_circle_points 10 4 = 210 := by
  sorry

#eval convex_quadrilaterals_from_circle_points 10 4

end NUMINAMATH_CALUDE_convex_quadrilaterals_count_l753_75339


namespace NUMINAMATH_CALUDE_no_real_solutions_l753_75396

/-- The quadratic equation x^2 + 2x + 3 = 0 has no real solutions -/
theorem no_real_solutions : ¬∃ (x : ℝ), x^2 + 2*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l753_75396


namespace NUMINAMATH_CALUDE_decimal_division_proof_l753_75321

theorem decimal_division_proof : (0.045 : ℝ) / (0.005 : ℝ) = 9 := by sorry

end NUMINAMATH_CALUDE_decimal_division_proof_l753_75321


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l753_75306

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (m n : Line) (α : Plane) :
  parallel m n → perpendicular m α → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l753_75306


namespace NUMINAMATH_CALUDE_focus_of_hyperbola_l753_75385

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- A point is a focus of the hyperbola if its x-coordinate squared equals a^2 + b^2 -/
def is_focus (x y : ℝ) : Prop :=
  x^2 = 3 + 1 ∧ y = 0

theorem focus_of_hyperbola :
  is_focus 2 0 :=
sorry

end NUMINAMATH_CALUDE_focus_of_hyperbola_l753_75385


namespace NUMINAMATH_CALUDE_another_two_digit_prime_digit_number_l753_75335

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem another_two_digit_prime_digit_number : 
  ∃ n : ℕ, is_two_digit n ∧ 
           is_prime (n / 10) ∧ 
           is_prime (n % 10) ∧ 
           n ≠ 23 :=
sorry

end NUMINAMATH_CALUDE_another_two_digit_prime_digit_number_l753_75335


namespace NUMINAMATH_CALUDE_completing_square_l753_75354

theorem completing_square (x : ℝ) : x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l753_75354


namespace NUMINAMATH_CALUDE_min_filtrations_correct_l753_75338

/-- The initial concentration of pollutants in mg/cm³ -/
def initial_concentration : ℝ := 1.2

/-- The reduction factor for each filtration -/
def reduction_factor : ℝ := 0.8

/-- The target concentration of pollutants in mg/cm³ -/
def target_concentration : ℝ := 0.2

/-- The minimum number of filtrations needed to reach the target concentration -/
def min_filtrations : ℕ := 8

theorem min_filtrations_correct :
  (∀ n : ℕ, n < min_filtrations → initial_concentration * reduction_factor ^ n > target_concentration) ∧
  initial_concentration * reduction_factor ^ min_filtrations ≤ target_concentration :=
by sorry

end NUMINAMATH_CALUDE_min_filtrations_correct_l753_75338


namespace NUMINAMATH_CALUDE_distance_between_locations_l753_75333

/-- The distance between two locations A and B given two cars meeting conditions --/
theorem distance_between_locations (speed_B : ℝ) (h1 : speed_B > 0) : 
  let speed_A := 1.2 * speed_B
  let midpoint_to_meeting := 8
  let time := 2 * midpoint_to_meeting / (speed_A - speed_B)
  (speed_A + speed_B) * time = 176 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_locations_l753_75333


namespace NUMINAMATH_CALUDE_distance_between_centers_is_sqrt_5_l753_75352

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle with sides 6, 8, and 10
def rightTriangle : Triangle := { a := 6, b := 8, c := 10 }

-- Define the distance between centers of inscribed and circumscribed circles
def distanceBetweenCenters (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem distance_between_centers_is_sqrt_5 :
  distanceBetweenCenters rightTriangle = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_centers_is_sqrt_5_l753_75352


namespace NUMINAMATH_CALUDE_andrew_mango_purchase_l753_75392

-- Define the given constants
def grape_quantity : ℕ := 6
def grape_price : ℕ := 74
def mango_price : ℕ := 59
def total_paid : ℕ := 975

-- Define the function to calculate the mango quantity
def mango_quantity : ℕ := (total_paid - grape_quantity * grape_price) / mango_price

-- Theorem statement
theorem andrew_mango_purchase :
  mango_quantity = 9 := by
  sorry

end NUMINAMATH_CALUDE_andrew_mango_purchase_l753_75392


namespace NUMINAMATH_CALUDE_total_difference_l753_75387

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def correct_discount : ℝ := 0.25
def charlie_discount : ℝ := 0.15

def anne_total : ℝ := original_price * (1 + sales_tax_rate) * (1 - correct_discount)
def ben_total : ℝ := original_price * (1 - correct_discount) * (1 + sales_tax_rate)
def charlie_total : ℝ := original_price * (1 - charlie_discount) * (1 + sales_tax_rate)

theorem total_difference : anne_total - ben_total - charlie_total = -12.96 := by
  sorry

end NUMINAMATH_CALUDE_total_difference_l753_75387


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l753_75388

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l753_75388


namespace NUMINAMATH_CALUDE_tank_filling_time_l753_75317

theorem tank_filling_time (a b c : ℝ) (h1 : c = 2 * b) (h2 : b = 2 * a) (h3 : a + b + c = 1 / 8) :
  1 / a = 56 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l753_75317


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l753_75309

theorem rectangle_area_with_hole (x : ℝ) : 
  (x + 7) * (x + 5) - (x + 1) * (x + 4) = 7 * x + 31 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l753_75309


namespace NUMINAMATH_CALUDE_track_length_is_400_l753_75303

/-- Represents a circular running track -/
structure Track :=
  (length : ℝ)

/-- Represents a runner on the track -/
structure Runner :=
  (speed : ℝ)
  (initialPosition : ℝ)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (position : ℝ)
  (time : ℝ)

/-- The scenario of two runners on a circular track -/
def runningScenario (t : Track) (r1 r2 : Runner) (m1 m2 : Meeting) : Prop :=
  r1.initialPosition = 0 ∧
  r2.initialPosition = t.length / 2 ∧
  r1.speed > 0 ∧
  r2.speed < 0 ∧
  m1.position = 100 ∧
  m2.position - m1.position = 150 ∧
  m1.time * r1.speed = 100 ∧
  m1.time * r2.speed = t.length / 2 - 100 ∧
  m2.time * r1.speed = t.length / 2 - 50 ∧
  m2.time * r2.speed = t.length / 2 + 50

theorem track_length_is_400 (t : Track) (r1 r2 : Runner) (m1 m2 : Meeting) :
  runningScenario t r1 r2 m1 m2 → t.length = 400 :=
by sorry

end NUMINAMATH_CALUDE_track_length_is_400_l753_75303


namespace NUMINAMATH_CALUDE_double_burger_cost_l753_75329

/-- The cost of a double burger given the following conditions:
  - Total spent: $68.50
  - Total number of hamburgers: 50
  - Single burger cost: $1.00 each
  - Number of double burgers: 37
-/
theorem double_burger_cost :
  let total_spent : ℚ := 68.5
  let total_burgers : ℕ := 50
  let single_burger_cost : ℚ := 1
  let double_burgers : ℕ := 37
  let single_burgers : ℕ := total_burgers - double_burgers
  let double_burger_cost : ℚ := (total_spent - (single_burgers : ℚ) * single_burger_cost) / (double_burgers : ℚ)
  double_burger_cost = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_double_burger_cost_l753_75329


namespace NUMINAMATH_CALUDE_trees_in_yard_l753_75325

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 300) (h2 : tree_distance = 12) : 
  yard_length / tree_distance + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l753_75325


namespace NUMINAMATH_CALUDE_inequality_solution_set_l753_75349

def solution_set (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 3 then {x | x < -3/a ∨ x > -1}
  else if a = 3 then {x | x ≠ -1}
  else if a > 3 then {x | x < -1 ∨ x > -3/a}
  else if a = 0 then {x | x > -1}
  else if a < 0 then {x | -1 < x ∧ x < -3/a}
  else ∅

theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  a * x^2 + 3 * x + 2 > -a * x - 1 ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l753_75349


namespace NUMINAMATH_CALUDE_tommy_initial_balloons_l753_75319

/-- The number of balloons Tommy's mom gave him -/
def balloons_from_mom : ℕ := 34

/-- The total number of balloons Tommy had after receiving more from his mom -/
def total_balloons : ℕ := 60

/-- The number of balloons Tommy had to start with -/
def initial_balloons : ℕ := total_balloons - balloons_from_mom

theorem tommy_initial_balloons : initial_balloons = 26 := by
  sorry

end NUMINAMATH_CALUDE_tommy_initial_balloons_l753_75319


namespace NUMINAMATH_CALUDE_binary_10111_is_23_l753_75315

/-- Converts a binary number represented as a list of bits (0 or 1) to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Nat := [1, 0, 1, 1, 1]

/-- Theorem stating that the decimal representation of (10111)₂ is 23 -/
theorem binary_10111_is_23 : binary_to_decimal binary_number = 23 := by
  sorry

end NUMINAMATH_CALUDE_binary_10111_is_23_l753_75315


namespace NUMINAMATH_CALUDE_dennis_pants_purchase_l753_75314

def pants_price : ℝ := 110
def socks_price : ℝ := 60
def discount_rate : ℝ := 0.3
def num_socks : ℕ := 2
def total_spent : ℝ := 392

def discounted_pants_price : ℝ := pants_price * (1 - discount_rate)
def discounted_socks_price : ℝ := socks_price * (1 - discount_rate)

theorem dennis_pants_purchase :
  ∃ (num_pants : ℕ),
    num_pants * discounted_pants_price + num_socks * discounted_socks_price = total_spent ∧
    num_pants = 4 := by
  sorry

end NUMINAMATH_CALUDE_dennis_pants_purchase_l753_75314


namespace NUMINAMATH_CALUDE_zeros_sum_greater_than_2a_l753_75398

/-- The function f(x) = ln x + a/x - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 2

/-- Theorem: If x₁ and x₂ are the two zeros of f(x) with x₁ < x₂, then x₁ + x₂ > 2a -/
theorem zeros_sum_greater_than_2a (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂)
  (h₄ : f a x₁ = 0) (h₅ : f a x₂ = 0) :
  x₁ + x₂ > 2 * a := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_greater_than_2a_l753_75398


namespace NUMINAMATH_CALUDE_difference_of_x_and_y_l753_75375

theorem difference_of_x_and_y (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_x_and_y_l753_75375


namespace NUMINAMATH_CALUDE_root_transformation_l753_75305

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 9 = 0) → 
  ((3*r₁)^3 - 12*(3*r₁)^2 + 243 = 0) ∧ 
  ((3*r₂)^3 - 12*(3*r₂)^2 + 243 = 0) ∧ 
  ((3*r₃)^3 - 12*(3*r₃)^2 + 243 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l753_75305


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l753_75370

theorem opposite_of_negative_five : 
  ∃ x : ℤ, (x + (-5) = 0 ∧ x = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l753_75370


namespace NUMINAMATH_CALUDE_middle_part_length_l753_75356

/-- Given a road of length 28 km divided into three parts, if the distance between
    the midpoints of the outer parts is 16 km, then the length of the middle part is 4 km. -/
theorem middle_part_length
  (total_length : ℝ)
  (part1 part2 part3 : ℝ)
  (h_total : total_length = 28)
  (h_parts : part1 + part2 + part3 = total_length)
  (h_midpoints : |((part1 + part2 + part3/2) - part1/2)| = 16) :
  part2 = 4 := by
sorry

end NUMINAMATH_CALUDE_middle_part_length_l753_75356


namespace NUMINAMATH_CALUDE_buns_per_student_fourth_class_l753_75323

/-- Calculates the number of buns per student in the fourth class --/
def bunsPerStudentInFourthClass (
  numClasses : ℕ)
  (studentsPerClass : ℕ)
  (bunsPerPackage : ℕ)
  (packagesClass1 : ℕ)
  (packagesClass2 : ℕ)
  (packagesClass3 : ℕ)
  (packagesClass4 : ℕ)
  (staleBuns : ℕ)
  (uneatenBuns : ℕ) : ℕ :=
  let totalBunsClass4 := packagesClass4 * bunsPerPackage
  let totalUneatenBuns := staleBuns + uneatenBuns
  let uneatenBunsPerClass := totalUneatenBuns / numClasses
  let availableBunsClass4 := totalBunsClass4 - uneatenBunsPerClass
  availableBunsClass4 / studentsPerClass

/-- Theorem: Given the conditions, the number of buns per student in the fourth class is 9 --/
theorem buns_per_student_fourth_class :
  bunsPerStudentInFourthClass 4 30 8 20 25 30 35 16 20 = 9 := by
  sorry

end NUMINAMATH_CALUDE_buns_per_student_fourth_class_l753_75323


namespace NUMINAMATH_CALUDE_cone_base_circumference_l753_75353

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = (1/3) * π * r^2 * h →
  V = 27 * π →
  h = 9 →
  2 * π * r = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l753_75353


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l753_75340

theorem trigonometric_simplification :
  (Real.sqrt (1 + 2 * Real.sin (610 * π / 180) * Real.cos (430 * π / 180))) /
  (Real.sin (250 * π / 180) + Real.cos (790 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l753_75340


namespace NUMINAMATH_CALUDE_complex_root_equation_l753_75355

theorem complex_root_equation (z : ℂ) : 
  (∃ a b : ℝ, z = a + b * I) → 
  z^2 = -3 - 4 * I → 
  z = -1 + 2 * I := by sorry

end NUMINAMATH_CALUDE_complex_root_equation_l753_75355


namespace NUMINAMATH_CALUDE_two_equal_roots_sum_l753_75313

theorem two_equal_roots_sum (a : ℝ) (α β : ℝ) :
  (∃! (x : ℝ), x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin x + 4 * Real.cos x = a) →
  (α ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin α + 4 * Real.cos α = a) →
  (β ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin β + 4 * Real.cos β = a) →
  (α + β = Real.pi - 2 * Real.arcsin (4/5) ∨ α + β = 3 * Real.pi - 2 * Real.arcsin (4/5)) :=
by sorry


end NUMINAMATH_CALUDE_two_equal_roots_sum_l753_75313


namespace NUMINAMATH_CALUDE_bmw_cars_sold_l753_75365

/-- The total number of cars sold -/
def total_cars : ℕ := 250

/-- The percentage of Audi cars sold -/
def audi_percent : ℚ := 10 / 100

/-- The percentage of Toyota cars sold -/
def toyota_percent : ℚ := 20 / 100

/-- The percentage of Acura cars sold -/
def acura_percent : ℚ := 15 / 100

/-- The percentage of Ford cars sold -/
def ford_percent : ℚ := 25 / 100

/-- The percentage of BMW cars sold -/
def bmw_percent : ℚ := 1 - (audi_percent + toyota_percent + acura_percent + ford_percent)

theorem bmw_cars_sold : 
  ⌊(bmw_percent * total_cars : ℚ)⌋ = 75 := by sorry

end NUMINAMATH_CALUDE_bmw_cars_sold_l753_75365


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l753_75397

-- Define sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem stating that "a ∈ M" is a necessary but not sufficient condition for "a ∈ N"
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry


end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l753_75397


namespace NUMINAMATH_CALUDE_division_result_l753_75377

theorem division_result : (4.036 : ℝ) / 0.04 = 100.9 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l753_75377


namespace NUMINAMATH_CALUDE_second_number_equality_l753_75311

theorem second_number_equality : ∃ x : ℤ, (9548 + x = 3362 + 13500) ∧ (x = 7314) := by
  sorry

end NUMINAMATH_CALUDE_second_number_equality_l753_75311


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l753_75363

/-- A quadratic function f(x) = x^2 + bx + 3 where b is a real number -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- The theorem stating that if the range of f is [0, +∞) and the solution set of f(x) < c
    is an open interval of length 8, then c = 16 -/
theorem quadratic_function_theorem (b : ℝ) (c : ℝ) :
  (∀ x, f b x ≥ 0) →
  (∃ m, ∀ x, f b x < c ↔ m - 8 < x ∧ x < m) →
  c = 16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l753_75363


namespace NUMINAMATH_CALUDE_modulus_of_z_l753_75399

-- Define the complex number z
def z : ℂ := Complex.I * (3 + 2 * Complex.I)

-- State the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l753_75399


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l753_75348

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_single_digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d ≤ 9

theorem seven_digit_divisible_by_11 (m n : ℕ) :
  is_single_digit m →
  is_single_digit n →
  is_divisible_by_11 (742 * 10000 + m * 1000 + 83 * 10 + n) →
  m + n = 1 :=
by sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l753_75348


namespace NUMINAMATH_CALUDE_tan_negative_two_implies_fraction_l753_75347

theorem tan_negative_two_implies_fraction (θ : Real) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_two_implies_fraction_l753_75347


namespace NUMINAMATH_CALUDE_exam_marks_problem_l753_75320

/-- Examination marks problem -/
theorem exam_marks_problem (full_marks : ℕ) (a_marks b_marks c_marks d_marks : ℕ) :
  full_marks = 500 →
  a_marks = (9 : ℕ) * b_marks / 10 →
  c_marks = (4 : ℕ) * d_marks / 5 →
  a_marks = 360 →
  d_marks = (4 : ℕ) * full_marks / 5 →
  b_marks - c_marks = c_marks / 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_problem_l753_75320


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identity_l753_75308

theorem triangle_trigonometric_identity (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 - 
  2 * Real.cos A * Real.cos B * Real.cos C = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identity_l753_75308


namespace NUMINAMATH_CALUDE_principal_is_10000_l753_75350

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- Theorem stating that given the conditions, the principal is 10000 -/
theorem principal_is_10000 (loan : SimpleLoan) 
  (h_rate : loan.rate = 12)
  (h_time : loan.time = 3)
  (h_interest : loan.interest = 3600)
  (h_simple_interest : loan.interest = loan.principal * loan.rate * loan.time / 100) :
  loan.principal = 10000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_10000_l753_75350


namespace NUMINAMATH_CALUDE_sin_cos_45_degrees_l753_75376

theorem sin_cos_45_degrees : 
  let θ : Real := Real.pi / 4  -- 45 degrees in radians
  Real.sin θ = 1 / Real.sqrt 2 ∧ Real.cos θ = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_45_degrees_l753_75376


namespace NUMINAMATH_CALUDE_robot_trajectory_constraint_l753_75304

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The trajectory of the robot -/
def robotTrajectory : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The line x = -1 -/
def verticalLine : Line :=
  { slope := 0, yIntercept := -1 }

/-- The point F(1, 0) -/
def pointF : Point :=
  { x := 1, y := 0 }

/-- The point P(-1, 0) -/
def pointP : Point :=
  { x := -1, y := 0 }

/-- The line passing through P(-1, 0) with slope k -/
def lineThroughP (k : ℝ) : Line :=
  { slope := k, yIntercept := k }

/-- The robot's trajectory does not intersect the line through P -/
def noIntersection (k : ℝ) : Prop :=
  ∀ p : Point, p ∈ robotTrajectory → p ∉ {p : Point | p.y = (lineThroughP k).slope * (p.x + 1)}

theorem robot_trajectory_constraint (k : ℝ) :
  noIntersection k ↔ k > 1 ∨ k < -1 :=
sorry

end NUMINAMATH_CALUDE_robot_trajectory_constraint_l753_75304


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l753_75368

theorem complex_fraction_sum (a b : ℝ) :
  (2 : ℂ) / (1 - I) = a + b * I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l753_75368
