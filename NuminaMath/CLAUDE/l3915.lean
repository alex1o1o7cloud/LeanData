import Mathlib

namespace position_of_2010_l3915_391585

/-- The sum of the first n terms of the arithmetic sequence representing the number of integers in each row -/
def rowSum (n : ℕ) : ℕ := n^2

/-- The first number in the nth row -/
def firstInRow (n : ℕ) : ℕ := rowSum (n - 1) + 1

/-- The position of a number in the table -/
structure Position where
  row : ℕ
  column : ℕ

/-- Find the position of a number in the table -/
def findPosition (num : ℕ) : Position :=
  let row := (Nat.sqrt (num - 1) + 1)
  let column := num - firstInRow row + 1
  ⟨row, column⟩

theorem position_of_2010 : findPosition 2010 = ⟨45, 74⟩ := by
  sorry

end position_of_2010_l3915_391585


namespace sin_cos_extrema_l3915_391558

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  let m := Real.sin x - Real.cos y ^ 2
  (∀ a b, Real.sin a + Real.sin b = 1/3 → m ≤ Real.sin a - Real.cos b ^ 2) ∧
  (∃ x' y', Real.sin x' + Real.sin y' = 1/3 ∧ m = 4/9) ∧
  (∀ a b, Real.sin a + Real.sin b = 1/3 → Real.sin a - Real.cos b ^ 2 ≤ m) ∧
  (∃ x'' y'', Real.sin x'' + Real.sin y'' = 1/3 ∧ m = -11/16) := by
  sorry

end sin_cos_extrema_l3915_391558


namespace group_size_problem_l3915_391572

theorem group_size_problem (T : ℝ) 
  (hat_wearers : ℝ → ℝ)
  (shoe_wearers : ℝ → ℝ)
  (both_wearers : ℝ → ℝ)
  (h1 : hat_wearers T = 0.40 * T + 60)
  (h2 : shoe_wearers T = 0.25 * T)
  (h3 : both_wearers T = 0.20 * T)
  (h4 : both_wearers T = hat_wearers T - shoe_wearers T) :
  T = 1200 := by
sorry

end group_size_problem_l3915_391572


namespace derivative_of_exp_ax_l3915_391583

theorem derivative_of_exp_ax (a : ℝ) (x : ℝ) :
  deriv (fun x => Real.exp (a * x)) x = a * Real.exp (a * x) := by
  sorry

end derivative_of_exp_ax_l3915_391583


namespace oregano_basil_difference_l3915_391538

theorem oregano_basil_difference (basil : ℕ) (total : ℕ) (oregano : ℕ) :
  basil = 5 →
  total = 17 →
  oregano > 2 * basil →
  total = basil + oregano →
  oregano - 2 * basil = 2 := by
  sorry

end oregano_basil_difference_l3915_391538


namespace equal_hikes_in_64_weeks_l3915_391595

/-- The number of weeks it takes for Camila to have hiked as many times as Steven -/
def weeks_to_equal_hikes : ℕ :=
  let camila_initial := 7
  let amanda_initial := 8 * camila_initial
  let steven_initial := amanda_initial + 15
  let david_initial := 2 * steven_initial
  let elizabeth_initial := david_initial - 10
  let camila_weekly := 4
  let amanda_weekly := 2
  let steven_weekly := 3
  let david_weekly := 5
  let elizabeth_weekly := 1
  64

theorem equal_hikes_in_64_weeks :
  let camila_initial := 7
  let amanda_initial := 8 * camila_initial
  let steven_initial := amanda_initial + 15
  let david_initial := 2 * steven_initial
  let elizabeth_initial := david_initial - 10
  let camila_weekly := 4
  let amanda_weekly := 2
  let steven_weekly := 3
  let david_weekly := 5
  let elizabeth_weekly := 1
  let w := weeks_to_equal_hikes
  camila_initial + camila_weekly * w = steven_initial + steven_weekly * w :=
by sorry

end equal_hikes_in_64_weeks_l3915_391595


namespace circle_center_not_constructible_with_straightedge_l3915_391508

-- Define a circle on a plane
def Circle : Type := sorry

-- Define a straightedge
def Straightedge : Type := sorry

-- Define a point on a plane
def Point : Type := sorry

-- Define the concept of constructing a point using a straightedge
def constructible (p : Point) (s : Straightedge) : Prop := sorry

-- Define the center of a circle
def center (c : Circle) : Point := sorry

-- Theorem statement
theorem circle_center_not_constructible_with_straightedge (c : Circle) (s : Straightedge) :
  ¬(constructible (center c) s) := by sorry

end circle_center_not_constructible_with_straightedge_l3915_391508


namespace cats_eating_mice_l3915_391574

/-- If n cats eat n mice in n hours, then p cats eat (p^2 / n) mice in p hours -/
theorem cats_eating_mice (n p : ℕ) (h : n ≠ 0) : 
  (n : ℚ) * (n : ℚ) / (n : ℚ) = n → (p : ℚ) * (p : ℚ) / (n : ℚ) = p^2 / n := by
  sorry

end cats_eating_mice_l3915_391574


namespace fraction_equality_l3915_391546

theorem fraction_equality : (3/7 + 5/8) / (5/12 + 2/15) = 295/154 := by
  sorry

end fraction_equality_l3915_391546


namespace gcd_lcm_sum_problem_l3915_391562

theorem gcd_lcm_sum_problem : Nat.gcd 40 60 + 2 * Nat.lcm 20 15 = 140 := by
  sorry

end gcd_lcm_sum_problem_l3915_391562


namespace cake_muffin_probability_l3915_391590

/-- The probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 17) :
  (total - (cake + muffin - both)) / total = 27 / 100 :=
by sorry

end cake_muffin_probability_l3915_391590


namespace angle_expression_value_l3915_391587

theorem angle_expression_value (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (θ - π) = -1/2) :
  Real.sqrt ((1 + Real.cos θ) / (1 - Real.sin (π/2 - θ))) - 
  Real.sqrt ((1 - Real.cos θ) / (1 + Real.sin (θ - 3*π/2))) = -4 := by
  sorry

end angle_expression_value_l3915_391587


namespace unique_solution_l3915_391563

theorem unique_solution : 
  ∃! x : ℝ, -1 < x ∧ x ≤ 2 ∧ 
  Real.sqrt (2 - x) + Real.sqrt (2 + 2*x) = 
  Real.sqrt ((x^4 + 1) / (x^2 + 1)) + (x + 3) / (x + 1) ∧ 
  x = 1 := by sorry

end unique_solution_l3915_391563


namespace union_M_N_l3915_391547

-- Define the universe set U
def U : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N_in_U : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N_in_U

-- Theorem to prove
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end union_M_N_l3915_391547


namespace roots_of_quadratic_l3915_391531

/-- A quadratic equation with roots 1 and -2 -/
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + x - 2 = 0

theorem roots_of_quadratic : 
  (quadratic_equation 1) ∧ (quadratic_equation (-2)) := by sorry

end roots_of_quadratic_l3915_391531


namespace y_is_75_percent_of_x_l3915_391522

-- Define variables
variable (x y z p : ℝ)

-- Define the theorem
theorem y_is_75_percent_of_x
  (h1 : 0.45 * z = 0.9 * y)
  (h2 : z = 1.5 * x)
  (h3 : y = p * x)
  : y = 0.75 * x :=
by sorry

end y_is_75_percent_of_x_l3915_391522


namespace point_movement_l3915_391556

def move_point (start : ℤ) (distance : ℤ) : ℤ := start + distance

theorem point_movement (A B : ℤ) :
  A = -3 →
  move_point A 4 = B →
  B = 1 := by sorry

end point_movement_l3915_391556


namespace system_of_equations_solutions_l3915_391513

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x - 2*y = 8 ∧ x = -2 ∧ y = -7) ∧
  -- System 2
  (∃ x y : ℝ, 3*x + 4*y = 5 ∧ 5*x - 2*y = 30 ∧ x = 5 ∧ y = -5/2) :=
by sorry

end system_of_equations_solutions_l3915_391513


namespace mia_wins_two_l3915_391502

/-- Represents a player in the chess tournament -/
inductive Player : Type
  | Sarah : Player
  | Ryan : Player
  | Mia : Player

/-- Represents the number of games won by a player -/
def wins : Player → ℕ
  | Player.Sarah => 5
  | Player.Ryan => 2
  | Player.Mia => 2  -- This is what we want to prove

/-- Represents the number of games lost by a player -/
def losses : Player → ℕ
  | Player.Sarah => 1
  | Player.Ryan => 4
  | Player.Mia => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := 6

theorem mia_wins_two : wins Player.Mia = 2 := by
  sorry

#check mia_wins_two

end mia_wins_two_l3915_391502


namespace consecutive_sum_product_l3915_391544

theorem consecutive_sum_product (start : ℕ) :
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) = 33) →
  (start * (start + 1) * (start + 2) * (start + 3) * (start + 4) * (start + 5) = 20160) :=
by
  sorry

end consecutive_sum_product_l3915_391544


namespace unique_prime_n_l3915_391560

def f (n : ℕ+) : ℤ := -n^4 + n^3 - 4*n^2 + 18*n - 19

theorem unique_prime_n : ∃! (n : ℕ+), Nat.Prime (Int.natAbs (f n)) :=
sorry

end unique_prime_n_l3915_391560


namespace new_car_distance_l3915_391553

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (speed_increase : ℝ) :
  old_car_distance = 150 →
  speed_increase = 0.3 →
  old_car_speed * (1 + speed_increase) * (old_car_distance / old_car_speed) = 195 :=
by sorry

end new_car_distance_l3915_391553


namespace trigonometric_identities_l3915_391515

theorem trigonometric_identities :
  (Real.sin (30 * π / 180) + Real.cos (45 * π / 180) = (1 + Real.sqrt 2) / 2) ∧
  (Real.sin (60 * π / 180) ^ 2 + Real.cos (60 * π / 180) ^ 2 - Real.tan (45 * π / 180) = 0) := by
  sorry

end trigonometric_identities_l3915_391515


namespace quadratic_root_range_l3915_391548

theorem quadratic_root_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + a*x + a^2 - 1 = 0 ∧ y^2 + a*y + a^2 - 1 = 0) →
  -1 < a ∧ a < 1 := by
sorry

end quadratic_root_range_l3915_391548


namespace golden_rabbit_cards_count_l3915_391577

/-- The total number of possible four-digit combinations -/
def total_combinations : ℕ := 10000

/-- The number of digits that are not 6 or 8 -/
def available_digits : ℕ := 8

/-- The number of digits in the combination -/
def combination_length : ℕ := 4

/-- The number of combinations without 6 or 8 -/
def combinations_without_6_or_8 : ℕ := available_digits ^ combination_length

/-- The number of "Golden Rabbit Cards" -/
def golden_rabbit_cards : ℕ := total_combinations - combinations_without_6_or_8

theorem golden_rabbit_cards_count : golden_rabbit_cards = 5904 := by
  sorry

end golden_rabbit_cards_count_l3915_391577


namespace order_of_operations_l3915_391567

theorem order_of_operations (a b c : ℕ) : a - b * c = a - (b * c) := by
  sorry

end order_of_operations_l3915_391567


namespace solve_system_l3915_391589

theorem solve_system (x y z : ℝ) 
  (eq1 : x + 2*y = 10)
  (eq2 : y = 3)
  (eq3 : x - 3*y + z = 7) :
  z = 12 := by
  sorry

end solve_system_l3915_391589


namespace complex_division_equality_l3915_391584

theorem complex_division_equality : (3 - Complex.I) / Complex.I = -1 - 3 * Complex.I := by
  sorry

end complex_division_equality_l3915_391584


namespace permutation_sum_consecutive_l3915_391596

theorem permutation_sum_consecutive (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → Fin n),
    Function.Bijective a ∧ Function.Bijective b ∧
    ∃ (k : ℕ), ∀ i : Fin n, (a i).val + (b i).val = k + i.val) ↔
  Odd n :=
sorry

end permutation_sum_consecutive_l3915_391596


namespace third_term_value_l3915_391578

-- Define the sequence sum function
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (n + 2 : ℚ)

-- Define the sequence term function
def a (n : ℕ) : ℚ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- Theorem statement
theorem third_term_value : a 3 = 1 / 20 := by
  sorry

end third_term_value_l3915_391578


namespace spermatogenesis_experiment_verification_l3915_391540

-- Define the available materials and tools
inductive Material
| MouseLiver
| Testes
| Kidneys

inductive Stain
| SudanIII
| AceticOrcein
| JanusGreen

inductive Tool
| DissociationFixative

-- Define the experiment steps
structure ExperimentSteps where
  material : Material
  fixative : Tool
  stain : Stain

-- Define the experiment result
structure ExperimentResult where
  cellTypesObserved : Nat

-- Define the correct experiment setup and result
def correctExperiment : ExperimentSteps := {
  material := Material.Testes,
  fixative := Tool.DissociationFixative,
  stain := Stain.AceticOrcein
}

def correctResult : ExperimentResult := {
  cellTypesObserved := 3
}

-- Theorem statement
theorem spermatogenesis_experiment_verification :
  ∀ (setup : ExperimentSteps) (result : ExperimentResult),
  setup = correctExperiment ∧ result = correctResult →
  setup.material = Material.Testes ∧
  setup.fixative = Tool.DissociationFixative ∧
  setup.stain = Stain.AceticOrcein ∧
  result.cellTypesObserved = 3 :=
by sorry

end spermatogenesis_experiment_verification_l3915_391540


namespace apps_files_difference_l3915_391550

/-- Represents the state of Dave's phone --/
structure PhoneState where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone --/
def initial_state : PhoneState := { apps := 15, files := 24 }

/-- The final state of Dave's phone --/
def final_state : PhoneState := { apps := 21, files := 4 }

/-- Theorem stating the difference between apps and files in the final state --/
theorem apps_files_difference :
  final_state.apps - final_state.files = 17 := by
  sorry

end apps_files_difference_l3915_391550


namespace simple_interest_increase_l3915_391536

/-- Given that the simple interest on $2000 increases by $40 when the time increases by x years,
    and the rate percent per annum is 0.5, prove that x = 4. -/
theorem simple_interest_increase (x : ℝ) : 
  (2000 * 0.5 * x) / 100 = 40 → x = 4 := by sorry

end simple_interest_increase_l3915_391536


namespace option_A_is_incorrect_l3915_391569

-- Define the set of angles whose terminal sides lie on y=x
def AnglesOnYEqualsX : Set ℝ := {β | ∃ n : ℤ, β = 45 + n * 180}

-- Define the set given in option A
def OptionASet : Set ℝ := {β | ∃ k : ℤ, β = 45 + k * 360 ∨ β = -45 + k * 360}

-- Theorem statement
theorem option_A_is_incorrect : OptionASet ≠ AnglesOnYEqualsX := by
  sorry

end option_A_is_incorrect_l3915_391569


namespace smallest_g_for_square_3150_l3915_391523

theorem smallest_g_for_square_3150 : 
  ∃ (g : ℕ), g > 0 ∧ 
  (∃ (n : ℕ), 3150 * g = n^2) ∧ 
  (∀ (k : ℕ), k > 0 → k < g → ¬∃ (m : ℕ), 3150 * k = m^2) ∧
  g = 14 := by
sorry

end smallest_g_for_square_3150_l3915_391523


namespace givenPoint_in_first_quadrant_l3915_391500

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point -/
def givenPoint : Point2D :=
  { x := 6, y := 2 }

/-- Theorem stating that the given point is in the first quadrant -/
theorem givenPoint_in_first_quadrant :
  isInFirstQuadrant givenPoint :=
by
  sorry

end givenPoint_in_first_quadrant_l3915_391500


namespace jacket_price_calculation_l3915_391580

def calculate_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (coupon : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_coupon := price_after_discount2 - coupon
  let final_price := price_after_coupon * (1 + tax_rate)
  final_price

theorem jacket_price_calculation :
  calculate_final_price 150 0.25 0.10 10 0.10 = 100.38 := by
  sorry

end jacket_price_calculation_l3915_391580


namespace divisibility_by_nine_l3915_391529

theorem divisibility_by_nine (n : ℕ) (h : 900 ≤ n ∧ n ≤ 999) : 
  (n % 9 = 0) ↔ ((n / 100 + (n / 10) % 10 + n % 10) % 9 = 0) := by
  sorry

end divisibility_by_nine_l3915_391529


namespace inequality_solution_l3915_391581

theorem inequality_solution (x : ℝ) : 
  (x^2 + 3*x + 3)^(5*x^3 - 3*x^2) ≤ (x^2 + 3*x + 3)^(3*x^3 + 5*x) ↔ 
  x ≤ -2 ∨ x = -1 ∨ (0 ≤ x ∧ x ≤ 5/2) := by
sorry

end inequality_solution_l3915_391581


namespace monic_polynomial_value_theorem_l3915_391566

theorem monic_polynomial_value_theorem (p : ℤ → ℤ) (a b c d : ℤ) :
  (∀ x, p x = p (x + 1) - p x) →  -- p is monic
  (∀ x, ∃ k, p x = k) →  -- p has integer coefficients
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct values
  p a = 5 ∧ p b = 5 ∧ p c = 5 ∧ p d = 5 →  -- p takes value 5 at four distinct integers
  ∀ x : ℤ, p x ≠ 8 :=
by
  sorry

#check monic_polynomial_value_theorem

end monic_polynomial_value_theorem_l3915_391566


namespace smallest_n_less_than_one_hundredth_l3915_391521

/-- The probability of stopping after drawing exactly n marbles -/
def Q (n : ℕ+) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 100

theorem smallest_n_less_than_one_hundredth :
  (∀ k : ℕ+, k < 10 → Q k ≥ 1/100) ∧
  (Q 10 < 1/100) ∧
  (∀ n : ℕ+, n ≤ num_boxes → Q n < 1/100 → n ≥ 10) :=
sorry

end smallest_n_less_than_one_hundredth_l3915_391521


namespace cream_ratio_is_15_23_l3915_391530

/-- The ratio of cream in Joe's coffee to JoAnn's coffee -/
def cream_ratio : ℚ := sorry

/-- Initial amount of coffee for both Joe and JoAnn -/
def initial_coffee : ℚ := 20

/-- Amount of cream added by both Joe and JoAnn -/
def cream_added : ℚ := 3

/-- Amount of mixture Joe drank -/
def joe_drank : ℚ := 4

/-- Amount of coffee JoAnn drank before adding cream -/
def joann_drank : ℚ := 4

/-- Theorem stating the ratio of cream in Joe's coffee to JoAnn's coffee -/
theorem cream_ratio_is_15_23 : cream_ratio = 15 / 23 := by sorry

end cream_ratio_is_15_23_l3915_391530


namespace sum_of_fractions_l3915_391518

theorem sum_of_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 9 = (7 : ℚ) / 12 := by
  sorry

end sum_of_fractions_l3915_391518


namespace bench_capacity_l3915_391532

theorem bench_capacity (num_benches : ℕ) (people_sitting : ℕ) (spaces_available : ℕ) 
  (h1 : num_benches = 50)
  (h2 : people_sitting = 80)
  (h3 : spaces_available = 120) :
  (num_benches * 4 = people_sitting + spaces_available) ∧ 
  (4 = (people_sitting + spaces_available) / num_benches) :=
by sorry

end bench_capacity_l3915_391532


namespace distinct_sides_not_isosceles_l3915_391543

-- Define a triangle with sides a, b, and c
structure Triangle (α : Type*) :=
  (a b c : α)

-- Define what it means for a triangle to be isosceles
def is_isosceles {α : Type*} [PartialOrder α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Theorem statement
theorem distinct_sides_not_isosceles {α : Type*} [LinearOrder α] 
  (t : Triangle α) (h_distinct : t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c) :
  ¬(is_isosceles t) :=
sorry

end distinct_sides_not_isosceles_l3915_391543


namespace quadratic_symmetry_and_point_l3915_391517

def f (x : ℝ) := (x - 2)^2 - 3

theorem quadratic_symmetry_and_point :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧ f 0 = 1 := by
  sorry

end quadratic_symmetry_and_point_l3915_391517


namespace max_servings_is_eight_l3915_391519

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (eggs : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Represents Lisa's available ingredients --/
structure Available :=
  (eggs : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Calculates the maximum number of servings possible for a given ingredient --/
def max_servings_for_ingredient (recipe_amount : ℚ) (available_amount : ℚ) : ℚ :=
  (available_amount / recipe_amount) * 4

/-- Finds the maximum number of servings possible given the recipe and available ingredients --/
def max_servings (recipe : Recipe) (available : Available) : ℚ :=
  min (max_servings_for_ingredient recipe.eggs available.eggs)
    (min (max_servings_for_ingredient recipe.sugar available.sugar)
      (max_servings_for_ingredient recipe.milk available.milk))

theorem max_servings_is_eight :
  let recipe := Recipe.mk 3 (1/2) 2
  let available := Available.mk 10 1 9
  max_servings recipe available = 8 := by
  sorry

#eval max_servings (Recipe.mk 3 (1/2) 2) (Available.mk 10 1 9)

end max_servings_is_eight_l3915_391519


namespace intersection_singleton_iff_a_in_range_l3915_391568

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * |p.1|}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + a}

theorem intersection_singleton_iff_a_in_range (a : ℝ) :
  (∃! p : ℝ × ℝ, p ∈ set_A a ∩ set_B a) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end intersection_singleton_iff_a_in_range_l3915_391568


namespace problem_solution_l3915_391549

theorem problem_solution (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) :
  (a * b < b * c) ∧ (a * c < b * c) ∧ (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end problem_solution_l3915_391549


namespace square_area_ratio_l3915_391555

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := s₂ * Real.sqrt 2
  (s₁ ^ 2) / (s₂ ^ 2) = 2 := by
sorry

end square_area_ratio_l3915_391555


namespace rotation_sum_65_l3915_391597

/-- Triangle in 2D space defined by three points -/
structure Triangle where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ

/-- Rotation in 2D space defined by an angle and a center point -/
structure Rotation where
  angle : ℝ
  center : ℝ × ℝ

/-- Check if two triangles are congruent under rotation -/
def isCongruentUnderRotation (t1 t2 : Triangle) (r : Rotation) : Prop :=
  sorry

theorem rotation_sum_65 (xyz x'y'z' : Triangle) (r : Rotation) :
  xyz.x = (0, 0) →
  xyz.y = (0, 15) →
  xyz.z = (20, 0) →
  x'y'z'.x = (30, 10) →
  x'y'z'.y = (40, 10) →
  x'y'z'.z = (30, 0) →
  isCongruentUnderRotation xyz x'y'z' r →
  r.angle ≤ r'.angle → isCongruentUnderRotation xyz x'y'z' r' →
  r.angle + r.center.1 + r.center.2 = 65 := by
  sorry

end rotation_sum_65_l3915_391597


namespace translation_of_line_segment_l3915_391503

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_of_line_segment :
  let A : Point := { x := 1, y := 1 }
  let B : Point := { x := -2, y := 0 }
  let A' : Point := { x := 4, y := 0 }
  let t : Translation := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' : Point := applyTranslation t B
  B'.x = 1 ∧ B'.y = -1 := by
  sorry

end translation_of_line_segment_l3915_391503


namespace geometric_series_sum_special_series_sum_l3915_391592

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, a * r^n = a / (1 - r) :=
sorry

/-- The sum of the infinite series 5 + 6(1/1000) + 7(1/1000)^2 + 8(1/1000)^3 + ... is 4995005/998001 -/
theorem special_series_sum :
  ∑' n : ℕ, (n + 5 : ℝ) * (1/1000)^(n-1) = 4995005 / 998001 :=
sorry

end geometric_series_sum_special_series_sum_l3915_391592


namespace tom_fishing_probability_l3915_391576

-- Define the weather conditions
inductive Weather
  | Sunny
  | Rainy
  | Cloudy

-- Define the probability of Tom going fishing for each weather condition
def fishing_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Sunny => 0.7
  | Weather.Rainy => 0.3
  | Weather.Cloudy => 0.5

-- Define the probability of each weather condition
def weather_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Sunny => 0.3
  | Weather.Rainy => 0.5
  | Weather.Cloudy => 0.2

-- Theorem stating the probability of Tom going fishing
theorem tom_fishing_probability :
  (fishing_prob Weather.Sunny * weather_prob Weather.Sunny +
   fishing_prob Weather.Rainy * weather_prob Weather.Rainy +
   fishing_prob Weather.Cloudy * weather_prob Weather.Cloudy) = 0.46 := by
  sorry


end tom_fishing_probability_l3915_391576


namespace circle_equation_l3915_391501

/-- Given a circle with center (2, -3) intercepted by the line 2x + 3y - 8 = 0
    with a chord length of 4√3, prove that its standard equation is (x-2)² + (y+3)² = 25 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -3)
  let line (x y : ℝ) : ℝ := 2*x + 3*y - 8
  let chord_length : ℝ := 4 * Real.sqrt 3
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 ↔ 
      ((p.1 - 2)^2 + (p.2 + 3)^2 = 25 ∧ 
       ∃ (q : ℝ × ℝ), line q.1 q.2 = 0 ∧ 
         (q.1 - p.1)^2 + (q.2 - p.2)^2 ≤ chord_length^2)) := by
  sorry

end circle_equation_l3915_391501


namespace toys_sold_l3915_391535

theorem toys_sold (selling_price : ℕ) (cost_price : ℕ) (gain : ℕ) :
  selling_price = 27300 →
  gain = 3 * cost_price →
  cost_price = 1300 →
  selling_price = (selling_price - gain) / cost_price * cost_price + gain →
  (selling_price - gain) / cost_price = 18 :=
by sorry

end toys_sold_l3915_391535


namespace trip_time_proof_l3915_391520

-- Define the distances
def freeway_distance : ℝ := 60
def mountain_distance : ℝ := 20

-- Define the time spent on mountain pass
def mountain_time : ℝ := 40

-- Define the speed ratio
def speed_ratio : ℝ := 4

-- Define the total trip time
def total_trip_time : ℝ := 70

-- Theorem statement
theorem trip_time_proof :
  let mountain_speed := mountain_distance / mountain_time
  let freeway_speed := speed_ratio * mountain_speed
  let freeway_time := freeway_distance / freeway_speed
  mountain_time + freeway_time = total_trip_time := by sorry

end trip_time_proof_l3915_391520


namespace charlie_has_largest_answer_l3915_391537

def starting_number : ℕ := 15

def alice_operation (n : ℕ) : ℕ := ((n - 2)^2 + 3)

def bob_operation (n : ℕ) : ℕ := (n^2 - 2 + 3)

def charlie_operation (n : ℕ) : ℕ := ((n - 2 + 3)^2)

theorem charlie_has_largest_answer :
  charlie_operation starting_number > alice_operation starting_number ∧
  charlie_operation starting_number > bob_operation starting_number := by
  sorry

end charlie_has_largest_answer_l3915_391537


namespace modulo_congruence_problem_l3915_391506

theorem modulo_congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 49325 % 31 = n % 31 ∧ n = 2 := by
  sorry

end modulo_congruence_problem_l3915_391506


namespace age_height_not_function_l3915_391575

-- Define a type for people
structure Person where
  age : ℕ
  height : ℝ

-- Define what it means for a relation to be a function
def is_function (R : α → β → Prop) : Prop :=
  ∀ a : α, ∃! b : β, R a b

-- State the theorem
theorem age_height_not_function :
  ¬ is_function (λ (p : Person) (h : ℝ) => p.height = h) :=
sorry

end age_height_not_function_l3915_391575


namespace line_through_point_l3915_391533

theorem line_through_point (k : ℚ) :
  (1 - k * 5 = -2 * (-4)) → k = -7/5 := by
  sorry

end line_through_point_l3915_391533


namespace quotient_remainder_difference_l3915_391561

theorem quotient_remainder_difference (N : ℕ) : 
  N ≥ 75 → 
  N % 5 = 0 → 
  (∀ m : ℕ, m ≥ 75 ∧ m % 5 = 0 → m ≥ N) →
  (N / 5) - (N % 34) = 8 := by
  sorry

end quotient_remainder_difference_l3915_391561


namespace jim_juice_consumption_l3915_391593

theorem jim_juice_consumption (susan_juice : ℚ) (jim_fraction : ℚ) :
  susan_juice = 3/8 →
  jim_fraction = 5/6 →
  jim_fraction * susan_juice = 5/16 := by
  sorry

end jim_juice_consumption_l3915_391593


namespace lukes_trays_l3915_391510

/-- Given that Luke can carry 4 trays at a time, made 9 trips, and picked up 16 trays from the second table,
    prove that he picked up 20 trays from the first table. -/
theorem lukes_trays (trays_per_trip : ℕ) (total_trips : ℕ) (trays_second_table : ℕ)
    (h1 : trays_per_trip = 4)
    (h2 : total_trips = 9)
    (h3 : trays_second_table = 16) :
    trays_per_trip * total_trips - trays_second_table = 20 :=
by sorry

end lukes_trays_l3915_391510


namespace only_proposition2_correct_l3915_391598

-- Define the propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q) → (p ∨ q) ∧ ¬((p ∨ q) → (p ∧ q))

def proposition2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  let not_p := ∀ x : ℝ, x^2 + 2*x > 0
  (¬p) ↔ not_p

def proposition3 : Prop := ∀ (p q : Prop), p ∧ ¬q → (p ∧ ¬q) ∧ (¬p ∨ q)

def proposition4 : Prop := ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that only proposition2 is correct
theorem only_proposition2_correct :
  ¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4 :=
sorry

end only_proposition2_correct_l3915_391598


namespace trivia_team_score_l3915_391516

theorem trivia_team_score : 
  let total_members : ℕ := 30
  let absent_members : ℕ := 8
  let points_per_member : ℕ := 4
  let deduction_per_incorrect : ℕ := 2
  let total_incorrect : ℕ := 6
  let bonus_multiplier : ℚ := 3/2

  let present_members : ℕ := total_members - absent_members
  let initial_points : ℕ := present_members * points_per_member
  let total_deductions : ℕ := total_incorrect * deduction_per_incorrect
  let points_after_deductions : ℕ := initial_points - total_deductions
  let final_score : ℚ := (points_after_deductions : ℚ) * bonus_multiplier

  final_score = 114 := by sorry

end trivia_team_score_l3915_391516


namespace line_graph_shows_trends_l3915_391526

-- Define the types of statistical graphs
inductive StatGraph
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

-- Define the properties of statistical graphs
def comparesQuantities (g : StatGraph) : Prop :=
  g = StatGraph.BarGraph

def showsTrends (g : StatGraph) : Prop :=
  g = StatGraph.LineGraph

def displaysParts (g : StatGraph) : Prop :=
  g = StatGraph.PieChart

def showsDistribution (g : StatGraph) : Prop :=
  g = StatGraph.Histogram

-- Define the set of common statistical graphs
def commonGraphs : Set StatGraph :=
  {StatGraph.BarGraph, StatGraph.LineGraph, StatGraph.PieChart, StatGraph.Histogram}

-- Theorem: The line graph is the type that can display the trend of data
theorem line_graph_shows_trends :
  ∃ (g : StatGraph), g ∈ commonGraphs ∧ showsTrends g ∧
    ∀ (h : StatGraph), h ∈ commonGraphs → showsTrends h → h = g :=
  sorry

end line_graph_shows_trends_l3915_391526


namespace binomial_expansion_property_l3915_391512

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end binomial_expansion_property_l3915_391512


namespace x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive_l3915_391573

theorem x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive :
  (∀ x : ℝ, x + |x| > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ x + |x| ≤ 0) :=
by sorry

end x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive_l3915_391573


namespace quadratic_inequality_range_l3915_391541

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ k ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end quadratic_inequality_range_l3915_391541


namespace pounds_in_ton_l3915_391559

theorem pounds_in_ton (ounces_per_pound : ℕ) (num_packets : ℕ) (packet_weight_pounds : ℕ) 
  (packet_weight_ounces : ℕ) (bag_capacity_tons : ℕ) :
  ounces_per_pound = 16 →
  num_packets = 1680 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  bag_capacity_tons = 13 →
  ∃ (pounds_per_ton : ℕ), pounds_per_ton = 2100 :=
by
  sorry

#check pounds_in_ton

end pounds_in_ton_l3915_391559


namespace isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3915_391599

/-- An isosceles triangle with congruent sides of length 10 and perimeter 35 has a base of length 15 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruentSide := (10 : ℝ)
    let perimeter := (35 : ℝ)
    (2 * congruentSide + base = perimeter) →
    (base = 15)

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 15 := by
  sorry

end isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3915_391599


namespace salary_calculation_l3915_391571

theorem salary_calculation (salary : ℝ) : 
  (salary * (1/5 : ℝ) + salary * (1/10 : ℝ) + salary * (3/5 : ℝ) + 15000 = salary) → 
  salary = 150000 := by
sorry

end salary_calculation_l3915_391571


namespace mountain_distance_l3915_391514

/-- Represents the mountain climbing scenario -/
structure MountainClimb where
  /-- Distance from bottom to top of the mountain in meters -/
  total_distance : ℝ
  /-- A's ascending speed in meters per hour -/
  speed_a_up : ℝ
  /-- B's ascending speed in meters per hour -/
  speed_b_up : ℝ
  /-- Distance from top where A and B meet in meters -/
  meeting_point : ℝ
  /-- Assumption that descending speed is 3 times ascending speed -/
  descent_speed_multiplier : ℝ
  /-- Assumption that A reaches bottom when B is halfway down -/
  b_halfway_when_a_bottom : Bool

/-- Main theorem: The distance from bottom to top is 1550 meters -/
theorem mountain_distance (climb : MountainClimb) 
  (h1 : climb.meeting_point = 150)
  (h2 : climb.descent_speed_multiplier = 3)
  (h3 : climb.b_halfway_when_a_bottom = true) :
  climb.total_distance = 1550 := by
  sorry

end mountain_distance_l3915_391514


namespace two_machines_copies_l3915_391509

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Theorem: Two copy machines working together make 2550 copies in 30 minutes -/
theorem two_machines_copies : 
  let machine1 : CopyMachine := ⟨30⟩
  let machine2 : CopyMachine := ⟨55⟩
  let total_time : ℕ := 30
  copies_made machine1 total_time + copies_made machine2 total_time = 2550 := by
  sorry


end two_machines_copies_l3915_391509


namespace given_number_scientific_notation_l3915_391570

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number in meters -/
def given_number : ℝ := 0.0000084

/-- The expected scientific notation representation -/
def expected_representation : ScientificNotation := {
  coefficient := 8.4
  exponent := -6
  is_normalized := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_scientific_notation : 
  given_number = expected_representation.coefficient * (10 : ℝ) ^ expected_representation.exponent := by
  sorry

end given_number_scientific_notation_l3915_391570


namespace function_depends_on_one_arg_l3915_391507

/-- A function that depends only on one of its arguments -/
def DependsOnOneArg {α : Type*} {k : ℕ} (f : (Fin k → α) → α) : Prop :=
  ∃ i : Fin k, ∀ x y : Fin k → α, (x i = y i) → (f x = f y)

/-- The main theorem -/
theorem function_depends_on_one_arg
  {n : ℕ} (h_n : n ≥ 3) (k : ℕ) (f : (Fin k → Fin n) → Fin n)
  (h_f : ∀ x y : Fin k → Fin n, (∀ i, x i ≠ y i) → f x ≠ f y) :
  DependsOnOneArg f := by
  sorry

end function_depends_on_one_arg_l3915_391507


namespace fruit_shop_quantities_l3915_391534

/-- Represents the quantities and prices of fruits in a shop --/
structure FruitShop where
  apple_quantity : ℝ
  pear_quantity : ℝ
  apple_price : ℝ
  pear_price : ℝ
  apple_profit_rate : ℝ
  pear_price_ratio : ℝ

/-- Theorem stating the correct quantities of apples and pears purchased --/
theorem fruit_shop_quantities (shop : FruitShop) 
  (total_weight : shop.apple_quantity + shop.pear_quantity = 200)
  (apple_price : shop.apple_price = 15)
  (pear_price : shop.pear_price = 10)
  (apple_profit : shop.apple_profit_rate = 0.4)
  (pear_price_ratio : shop.pear_price_ratio = 2/3)
  (total_profit : 
    shop.apple_quantity * shop.apple_price * shop.apple_profit_rate + 
    shop.pear_quantity * (shop.apple_price * (1 + shop.apple_profit_rate) * shop.pear_price_ratio - shop.pear_price) = 1020) :
  shop.apple_quantity = 110 ∧ shop.pear_quantity = 90 := by
  sorry

end fruit_shop_quantities_l3915_391534


namespace cos_225_degrees_l3915_391524

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l3915_391524


namespace geometric_sequence_third_term_l3915_391551

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ+ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_prod : a 2 * a 4 = 9) :
  a 3 = 3 := by
  sorry

end geometric_sequence_third_term_l3915_391551


namespace ellipse_x_intersection_l3915_391579

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y - 3)^2) + Real.sqrt ((x - 4)^2 + (y - 1)^2) = 10

/-- Theorem stating the other x-axis intersection point of the ellipse -/
theorem ellipse_x_intersection :
  ∃ x : ℝ, x = 1 + Real.sqrt 40 ∧ ellipse x 0 ∧ ellipse 0 0 := by sorry

end ellipse_x_intersection_l3915_391579


namespace perpendicular_lines_theorem_l3915_391557

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection operation between planes
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem perpendicular_lines_theorem 
  (α β : Plane) (a b l : Line) 
  (h1 : perp_planes α β)
  (h2 : intersection α β = l)
  (h3 : parallel_line_plane a α)
  (h4 : perp_line_plane b β) :
  perp_lines b l :=
sorry

end perpendicular_lines_theorem_l3915_391557


namespace no_solution_for_part_a_unique_solution_for_part_b_l3915_391527

-- Define S(x) as the sum of digits of a natural number
def S (x : ℕ) : ℕ := sorry

-- Theorem for part (a)
theorem no_solution_for_part_a :
  ¬ ∃ x : ℕ, x + S x + S (S x) = 1993 := by sorry

-- Theorem for part (b)
theorem unique_solution_for_part_b :
  ∃! x : ℕ, x + S x + S (S x) + S (S (S x)) = 1993 ∧ x = 1963 := by sorry

end no_solution_for_part_a_unique_solution_for_part_b_l3915_391527


namespace arithmetic_sequence_nth_term_l3915_391539

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def nth_term (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_nth_term 
  (x : ℝ) (n : ℕ) 
  (h1 : arithmetic_sequence (3*x - 4) (7*x - 14) (4*x + 5))
  (h2 : ∃ (a d : ℝ), nth_term a d n = 4013 ∧ a = 3*x - 4 ∧ d = (7*x - 14) - (3*x - 4)) :
  n = 610 := by
sorry

end arithmetic_sequence_nth_term_l3915_391539


namespace batsman_average_l3915_391554

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℚ) : 
  total_innings = 25 →
  last_innings_score = 95 →
  average_increase = 5/2 →
  (∃ (previous_average : ℚ), 
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings = previous_average + average_increase) →
  (∃ (final_average : ℚ), final_average = 35) := by
sorry

end batsman_average_l3915_391554


namespace shopping_remainder_l3915_391591

def initial_amount : ℕ := 109
def shirt_cost : ℕ := 11
def num_shirts : ℕ := 2
def pants_cost : ℕ := 13

theorem shopping_remainder :
  initial_amount - (shirt_cost * num_shirts + pants_cost) = 74 := by
  sorry

end shopping_remainder_l3915_391591


namespace optimal_allocation_l3915_391505

/-- Represents the allocation of workers to different parts -/
structure WorkerAllocation where
  partA : ℕ
  partB : ℕ
  partC : ℕ

/-- Checks if the given allocation satisfies the total worker constraint -/
def satisfiesTotalWorkers (allocation : WorkerAllocation) : Prop :=
  allocation.partA + allocation.partB + allocation.partC = 45

/-- Checks if the given allocation produces parts in the required ratio -/
def satisfiesProductionRatio (allocation : WorkerAllocation) : Prop :=
  30 * allocation.partA = 25 * allocation.partB * 3 / 5 ∧
  30 * allocation.partA = 20 * allocation.partC * 3 / 4

/-- The main theorem stating that the given allocation satisfies all constraints -/
theorem optimal_allocation :
  let allocation := WorkerAllocation.mk 9 18 18
  satisfiesTotalWorkers allocation ∧ satisfiesProductionRatio allocation :=
by sorry

end optimal_allocation_l3915_391505


namespace maker_funds_and_loan_repayment_l3915_391565

/-- Represents the remaining funds after n months -/
def remaining_funds (n : ℕ) : ℝ := sorry

/-- The initial borrowed capital -/
def initial_capital : ℝ := 100000

/-- Monthly profit rate -/
def profit_rate : ℝ := 0.2

/-- Monthly expense rate (rent and tax) -/
def expense_rate : ℝ := 0.1

/-- Monthly fixed expenses -/
def fixed_expenses : ℝ := 3000

/-- Annual interest rate for the bank loan -/
def annual_interest_rate : ℝ := 0.05

/-- Number of months in a year -/
def months_in_year : ℕ := 12

theorem maker_funds_and_loan_repayment :
  remaining_funds months_in_year = 194890 ∧
  remaining_funds months_in_year > initial_capital * (1 + annual_interest_rate) :=
sorry

end maker_funds_and_loan_repayment_l3915_391565


namespace simple_interest_calculation_l3915_391511

/-- Calculate the total amount owed after one year with simple interest. -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 75) 
  (h2 : rate = 0.07) 
  (h3 : time = 1) : 
  principal * (1 + rate * time) = 80.25 := by
sorry

end simple_interest_calculation_l3915_391511


namespace function_extrema_sum_l3915_391582

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 0

-- State the theorem
theorem function_extrema_sum (m : ℝ) :
  (∃ (max min : ℝ), 
    max ∈ Set.image (f m) interval ∧ 
    min ∈ Set.image (f m) interval ∧
    (∀ y ∈ Set.image (f m) interval, y ≤ max ∧ y ≥ min) ∧
    max + min = -14) →
  m = 1 :=
sorry

end function_extrema_sum_l3915_391582


namespace equation_solution_l3915_391594

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 ↔ 
  x = (-17 + Real.sqrt 569) / 4 ∨ x = (-17 - Real.sqrt 569) / 4 :=
by sorry

end equation_solution_l3915_391594


namespace pentagon_enclosure_percentage_l3915_391504

/-- Represents a tiling pattern on a plane -/
structure TilingPattern where
  smallSquaresPerLargeSquare : ℕ
  pentagonsPerLargeSquare : ℕ

/-- Calculates the percentage of the plane enclosed by pentagons -/
def percentEnclosedByPentagons (pattern : TilingPattern) : ℚ :=
  (pattern.pentagonsPerLargeSquare : ℚ) / (pattern.smallSquaresPerLargeSquare : ℚ) * 100

/-- The specific tiling pattern described in the problem -/
def problemPattern : TilingPattern :=
  { smallSquaresPerLargeSquare := 16
  , pentagonsPerLargeSquare := 5 }

theorem pentagon_enclosure_percentage :
  percentEnclosedByPentagons problemPattern = 31.25 := by
  sorry

end pentagon_enclosure_percentage_l3915_391504


namespace congruence_solution_l3915_391586

theorem congruence_solution (n : ℤ) : 
  4 ≤ n ∧ n ≤ 10 ∧ n ≡ 11783 [ZMOD 7] → n = 5 := by sorry

end congruence_solution_l3915_391586


namespace wednesday_sales_l3915_391545

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40
def unsold_percentage : ℚ := 80.57142857142857 / 100

theorem wednesday_sales :
  let unsold := (initial_stock : ℚ) * unsold_percentage
  let other_days_sales := monday_sales + tuesday_sales + thursday_sales + friday_sales
  initial_stock - (unsold.floor + other_days_sales) = 60 := by
  sorry

end wednesday_sales_l3915_391545


namespace spot_fraction_l3915_391542

theorem spot_fraction (rover_spots cisco_spots granger_spots total_spots : ℕ) 
  (f : ℚ) : 
  rover_spots = 46 →
  granger_spots = 5 * cisco_spots →
  granger_spots + cisco_spots = total_spots →
  total_spots = 108 →
  cisco_spots = f * 46 - 5 →
  f = 1/2 := by
  sorry

end spot_fraction_l3915_391542


namespace necessary_not_sufficient_condition_l3915_391525

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is an infinite set -/
def InfiniteDomain (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, f x ≠ 0

/-- There exist infinitely many real numbers x in the domain such that f(-x) = f(x) -/
def InfinitelyManySymmetricPoints (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, f (-x) = f x

theorem necessary_not_sufficient_condition (f : ℝ → ℝ) :
  InfiniteDomain f →
  (IsEven f → InfinitelyManySymmetricPoints f) ∧
  ¬(InfinitelyManySymmetricPoints f → IsEven f) :=
by sorry

end necessary_not_sufficient_condition_l3915_391525


namespace matrix_multiplication_and_scalar_l3915_391588

theorem matrix_multiplication_and_scalar : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 2]
  2 • (A * B) = !![34, -14; 32, -32] := by sorry

end matrix_multiplication_and_scalar_l3915_391588


namespace new_average_weight_l3915_391528

def original_team_size : ℕ := 7
def original_average_weight : ℝ := 121
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60

theorem new_average_weight :
  let total_original_weight : ℝ := original_team_size * original_average_weight
  let new_total_weight : ℝ := total_original_weight + new_player1_weight + new_player2_weight
  let new_team_size : ℕ := original_team_size + 2
  (new_total_weight / new_team_size : ℝ) = 113 := by sorry

end new_average_weight_l3915_391528


namespace rectangular_plot_breadth_l3915_391564

/-- Properties of a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_is_thrice_breadth : length = 3 * breadth
  area_formula : area = length * breadth

/-- Theorem: The breadth of a rectangular plot with given properties is 11 meters -/
theorem rectangular_plot_breadth (plot : RectangularPlot) 
  (h : plot.area = 363) : plot.breadth = 11 := by
  sorry

end rectangular_plot_breadth_l3915_391564


namespace sqrt_equation_solution_l3915_391552

theorem sqrt_equation_solution : ∃ x : ℝ, x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end sqrt_equation_solution_l3915_391552
