import Mathlib

namespace decimal_place_150_of_5_11_l2827_282775

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of the decimal representation of a rational number -/
def decimal_period (q : ℚ) : ℕ := sorry

theorem decimal_place_150_of_5_11 :
  decimal_representation (5/11) 150 = 5 := by sorry

end decimal_place_150_of_5_11_l2827_282775


namespace least_subtraction_for_divisibility_l2827_282795

theorem least_subtraction_for_divisibility (n : ℕ) : 
  (∃ (x : ℕ), x = 46 ∧ 
   (∀ (y : ℕ), y < x → ¬(5 ∣ (9671 - y) ∧ 7 ∣ (9671 - y) ∧ 11 ∣ (9671 - y))) ∧
   (5 ∣ (9671 - x) ∧ 7 ∣ (9671 - x) ∧ 11 ∣ (9671 - x))) := by
  sorry

end least_subtraction_for_divisibility_l2827_282795


namespace p_necessary_not_sufficient_for_q_l2827_282723

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x > 0, Real.exp x - a * x < 1

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(a - 1)^x) > (-(a - 1)^y)

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬(q a)) := by
  sorry

end p_necessary_not_sufficient_for_q_l2827_282723


namespace complex_function_property_l2827_282706

theorem complex_function_property (a b : ℝ) :
  (∀ z : ℂ, Complex.abs ((a + b * Complex.I) * z^2 - z) = Complex.abs ((a + b * Complex.I) * z^2)) →
  Complex.abs (a + b * Complex.I) = 5 →
  b^2 = 99/4 := by sorry

end complex_function_property_l2827_282706


namespace unique_g_30_equals_48_l2827_282745

def sumOfDivisors (n : ℕ) : ℕ := sorry

def g₁ (n : ℕ) : ℕ := 4 * sumOfDivisors n

def g (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j+1 => g₁ (g j n)

theorem unique_g_30_equals_48 :
  ∃! n : ℕ, n ≤ 30 ∧ g 30 n = 48 := by sorry

end unique_g_30_equals_48_l2827_282745


namespace absolute_value_simplification_l2827_282781

theorem absolute_value_simplification : |-4^2 - 6| = 22 := by
  sorry

end absolute_value_simplification_l2827_282781


namespace wilsons_theorem_l2827_282792

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) : ℤ) ≡ -1 [ZMOD p] := by
  sorry

end wilsons_theorem_l2827_282792


namespace average_study_time_difference_l2827_282711

def asha_times : List ℝ := [40, 60, 50, 70, 30, 55, 45]
def sasha_times : List ℝ := [50, 70, 40, 100, 10, 55, 0]

theorem average_study_time_difference :
  (List.sum (List.zipWith (·-·) sasha_times asha_times)) / asha_times.length = -25 / 7 := by
  sorry

end average_study_time_difference_l2827_282711


namespace schoolchildren_mushroom_picking_l2827_282754

theorem schoolchildren_mushroom_picking (n : ℕ) 
  (h_max : ∃ (child : ℕ), child ≤ n ∧ child * 5 = n) 
  (h_min : ∃ (child : ℕ), child ≤ n ∧ child * 7 = n) : 
  5 < n ∧ n < 7 := by
  sorry

#check schoolchildren_mushroom_picking

end schoolchildren_mushroom_picking_l2827_282754


namespace blackboard_final_product_l2827_282726

/-- Represents the count of each number on the blackboard -/
structure BoardState :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (fours : ℕ)

/-- Represents a single operation on the board -/
inductive Operation
  | erase_123_add_4
  | erase_124_add_3
  | erase_134_add_2
  | erase_234_add_1

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_123_add_4 => ⟨state.ones - 1, state.twos - 1, state.threes - 1, state.fours + 2⟩
  | Operation.erase_124_add_3 => ⟨state.ones - 1, state.twos - 1, state.threes + 2, state.fours - 1⟩
  | Operation.erase_134_add_2 => ⟨state.ones - 1, state.twos + 2, state.threes - 1, state.fours - 1⟩
  | Operation.erase_234_add_1 => ⟨state.ones + 2, state.twos - 1, state.threes - 1, state.fours - 1⟩

/-- Checks if a board state is in the final condition (only 3 numbers left) -/
def is_final_state (state : BoardState) : Prop :=
  (state.ones = 0 ∧ state.twos = 2 ∧ state.threes = 1 ∧ state.fours = 0) ∨
  (state.ones = 0 ∧ state.twos = 1 ∧ state.threes = 2 ∧ state.fours = 0) ∨
  (state.ones = 0 ∧ state.twos = 2 ∧ state.threes = 0 ∧ state.fours = 1) ∨
  (state.ones = 1 ∧ state.twos = 0 ∧ state.threes = 2 ∧ state.fours = 0) ∨
  (state.ones = 1 ∧ state.twos = 2 ∧ state.threes = 0 ∧ state.fours = 0) ∨
  (state.ones = 2 ∧ state.twos = 0 ∧ state.threes = 1 ∧ state.fours = 0) ∨
  (state.ones = 2 ∧ state.twos = 1 ∧ state.threes = 0 ∧ state.fours = 0)

/-- Calculates the product of the last three remaining numbers -/
def final_product (state : BoardState) : ℕ :=
  if state.ones > 0 then state.ones else 1 *
  if state.twos > 0 then state.twos else 1 *
  if state.threes > 0 then state.threes else 1 *
  if state.fours > 0 then state.fours else 1

/-- The main theorem to prove -/
theorem blackboard_final_product :
  ∀ (operations : List Operation),
  let initial_state : BoardState := ⟨11, 22, 33, 44⟩
  let final_state := operations.foldl apply_operation initial_state
  is_final_state final_state → final_product final_state = 12 :=
sorry

end blackboard_final_product_l2827_282726


namespace arithmetic_contains_geometric_l2827_282798

/-- Given positive integers a and d, there exist positive integers b and q such that 
    the geometric progression b, bq, bq^2, ... is a subset of the arithmetic progression a, a+d, a+2d, ... -/
theorem arithmetic_contains_geometric (a d : ℕ+) : 
  ∃ (b q : ℕ+), ∀ (n : ℕ), ∃ (k : ℕ), b * q ^ n = a + k * d := by
  sorry

end arithmetic_contains_geometric_l2827_282798


namespace francisFamily_violins_l2827_282768

theorem francisFamily_violins :
  let ukuleles : ℕ := 2
  let guitars : ℕ := 4
  let ukuleleStrings : ℕ := 4
  let guitarStrings : ℕ := 6
  let violinStrings : ℕ := 4
  let totalStrings : ℕ := 40
  
  ∃ violins : ℕ,
    violins * violinStrings + ukuleles * ukuleleStrings + guitars * guitarStrings = totalStrings ∧
    violins = 2 :=
by sorry

end francisFamily_violins_l2827_282768


namespace apple_pear_box_difference_l2827_282721

theorem apple_pear_box_difference :
  ∀ (initial_apples initial_pears additional : ℕ),
    initial_apples = 25 →
    initial_pears = 12 →
    additional = 8 →
    (initial_apples + additional) - (initial_pears + additional) = 13 :=
by
  sorry

end apple_pear_box_difference_l2827_282721


namespace unique_solution_l2827_282784

-- Define the equation
def equation (x : ℝ) : Prop := Real.rpow (5 - x) (1/3) + Real.sqrt (x + 2) = 3

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 4 := by sorry

end unique_solution_l2827_282784


namespace pentagon_sum_problem_l2827_282785

theorem pentagon_sum_problem : ∃ (a b c d e : ℝ),
  a + b = 1 ∧
  b + c = 2 ∧
  c + d = 3 ∧
  d + e = 4 ∧
  e + a = 5 := by
  sorry

end pentagon_sum_problem_l2827_282785


namespace smallest_prime_dividing_sum_l2827_282747

theorem smallest_prime_dividing_sum : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^11 + 5^13) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^11 + 5^13) → p ≤ q :=
sorry

end smallest_prime_dividing_sum_l2827_282747


namespace honey_distribution_l2827_282739

theorem honey_distribution (bottles : ℕ) (weight_per_bottle : ℚ) (share_per_neighbor : ℚ) :
  bottles = 4 →
  weight_per_bottle = 3 →
  share_per_neighbor = 3/4 →
  (bottles * weight_per_bottle) / share_per_neighbor = 16 := by
sorry

end honey_distribution_l2827_282739


namespace rectangular_plot_breadth_l2827_282780

/-- A rectangular plot with length thrice its breadth and area 675 sq m has a breadth of 15 m -/
theorem rectangular_plot_breadth : 
  ∀ (length breadth : ℝ),
  length = 3 * breadth →
  length * breadth = 675 →
  breadth = 15 :=
by
  sorry


end rectangular_plot_breadth_l2827_282780


namespace chocolate_bars_count_l2827_282751

theorem chocolate_bars_count (bar_price : ℕ) (remaining_bars : ℕ) (total_sales : ℕ) : 
  bar_price = 6 →
  remaining_bars = 6 →
  total_sales = 42 →
  ∃ (total_bars : ℕ), total_bars = 13 ∧ bar_price * (total_bars - remaining_bars) = total_sales :=
by sorry

end chocolate_bars_count_l2827_282751


namespace no_positive_integer_solution_l2827_282774

theorem no_positive_integer_solution :
  ¬ ∃ (a b c : ℕ+), a^2 + b^2 = 4 * c + 3 := by
  sorry

end no_positive_integer_solution_l2827_282774


namespace place_value_ratio_in_53687_4921_l2827_282715

/-- The place value of a digit in a decimal number -/
def place_value (digit_position : Int) : ℚ :=
  10 ^ digit_position

/-- The position of a digit in a decimal number, counting from right to left,
    with the decimal point at position 0 -/
def digit_position (n : ℚ) (d : ℕ) : Int :=
  sorry

theorem place_value_ratio_in_53687_4921 :
  let n : ℚ := 53687.4921
  let pos_8 := digit_position n 8
  let pos_2 := digit_position n 2
  place_value pos_8 / place_value pos_2 = 1000 := by sorry

end place_value_ratio_in_53687_4921_l2827_282715


namespace solution_exists_l2827_282776

theorem solution_exists (N : ℝ) : ∃ x₁ x₂ x₃ x₄ : ℤ, 
  (x₁ > ⌊N⌋) ∧ (x₂ > ⌊N⌋) ∧ (x₃ > ⌊N⌋) ∧ (x₄ > ⌊N⌋) ∧
  (x₁^2 + x₂^2 + x₃^2 + x₄^2 : ℤ) = x₁*x₂*x₃ + x₁*x₂*x₄ + x₁*x₃*x₄ + x₂*x₃*x₄ :=
by
  sorry

end solution_exists_l2827_282776


namespace candy_distribution_l2827_282786

theorem candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 10)
  (h2 : additional_candies = 4)
  (h3 : num_friends = 7)
  (h4 : num_friends > 0) :
  (initial_candies + additional_candies) / num_friends = 2 :=
by
  sorry

end candy_distribution_l2827_282786


namespace hyperbola_equation_l2827_282769

/-- The equation of a hyperbola sharing foci with an ellipse and passing through a point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    (x - 2)^2 / a^2 - (y - 1)^2 / b^2 = 1) →
  x^2 / 2 - y^2 = 1 :=
by sorry

end hyperbola_equation_l2827_282769


namespace gcf_78_104_l2827_282757

theorem gcf_78_104 : Nat.gcd 78 104 = 26 := by
  sorry

end gcf_78_104_l2827_282757


namespace budget_allocation_l2827_282750

def budget : ℝ := 1000

def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def clothes_percentage : ℝ := 0.05

def coursework_percentage : ℝ :=
  1 - (food_percentage + accommodation_percentage + entertainment_percentage + transportation_percentage + clothes_percentage)

def combined_percentage : ℝ :=
  entertainment_percentage + transportation_percentage + coursework_percentage

def combined_amount : ℝ :=
  combined_percentage * budget

theorem budget_allocation :
  combined_percentage = 0.50 ∧ combined_amount = 500 := by
  sorry

end budget_allocation_l2827_282750


namespace predictor_variable_is_fertilizer_l2827_282741

/-- Represents a variable in the study -/
inductive StudyVariable
  | YieldOfCrops
  | AmountOfFertilizer
  | Experimenter
  | OtherVariables

/-- Defines the characteristics of the study -/
structure CropStudy where
  predictedVariable : StudyVariable
  predictorVariable : StudyVariable
  aim : String

/-- Theorem stating that the predictor variable in the crop yield study is the amount of fertilizer -/
theorem predictor_variable_is_fertilizer (study : CropStudy) :
  study.aim = "determine whether the yield of crops can be predicted based on the amount of fertilizer applied" →
  study.predictedVariable = StudyVariable.YieldOfCrops →
  study.predictorVariable = StudyVariable.AmountOfFertilizer :=
by sorry

end predictor_variable_is_fertilizer_l2827_282741


namespace regular_polygon_sides_l2827_282722

theorem regular_polygon_sides (n : ℕ) (central_angle : ℝ) : 
  n > 0 ∧ central_angle = 72 → (360 : ℝ) / n = central_angle → n = 5 := by
  sorry

end regular_polygon_sides_l2827_282722


namespace smallest_set_size_existence_of_set_smallest_set_size_is_eight_l2827_282701

theorem smallest_set_size (n : ℕ) (h : n ≥ 5) :
  (∃ (S : Finset (ℕ × ℕ)),
    S.card = n ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2))) →
  n ≥ 8 := by sorry

theorem existence_of_set :
  ∃ (S : Finset (ℕ × ℕ)),
    S.card = 8 ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2)) := by sorry

theorem smallest_set_size_is_eight :
  (∃ (n : ℕ), n ≥ 5 ∧
    (∃ (S : Finset (ℕ × ℕ)),
      S.card = n ∧
      (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
      (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
        ∃ (r : ℕ × ℕ), r ∈ S ∧
          4 ∣ (p.1 + q.1 - r.1) ∧
          4 ∣ (p.2 + q.2 - r.2)))) →
  (∀ (m : ℕ), m ≥ 5 →
    (∃ (S : Finset (ℕ × ℕ)),
      S.card = m ∧
      (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
      (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
        ∃ (r : ℕ × ℕ), r ∈ S ∧
          4 ∣ (p.1 + q.1 - r.1) ∧
          4 ∣ (p.2 + q.2 - r.2))) →
    m ≥ 8) ∧
  (∃ (S : Finset (ℕ × ℕ)),
    S.card = 8 ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2))) := by sorry

end smallest_set_size_existence_of_set_smallest_set_size_is_eight_l2827_282701


namespace park_to_restaurant_time_l2827_282794

/-- Represents the time in minutes for various segments of Dante's walk -/
structure WalkTimes where
  parkToHiddenLake : ℕ
  hiddenLakeToPark : ℕ
  parkToRestaurant : ℕ
  totalTime : ℕ

/-- Proves that the time to walk from Park Office to Lake Park restaurant is 10 minutes -/
theorem park_to_restaurant_time (w : WalkTimes) 
  (h1 : w.parkToHiddenLake = 15)
  (h2 : w.hiddenLakeToPark = 7)
  (h3 : w.totalTime = 32)
  (h4 : w.totalTime = w.parkToHiddenLake + w.hiddenLakeToPark + w.parkToRestaurant) :
  w.parkToRestaurant = 10 := by
  sorry

end park_to_restaurant_time_l2827_282794


namespace triangle_ABC_properties_l2827_282764

/-- Triangle ABC with given side lengths and angle -/
structure TriangleABC where
  AB : ℝ
  BC : ℝ
  cosC : ℝ
  h_AB : AB = Real.sqrt 2
  h_BC : BC = 1
  h_cosC : cosC = 3/4

/-- The main theorem about TriangleABC -/
theorem triangle_ABC_properties (t : TriangleABC) :
  let sinA := Real.sqrt (14) / 8
  let dot_product := -(3/2 : ℝ)
  (∃ (CA : ℝ), sinA = Real.sqrt (1 - t.cosC^2) * t.BC / t.AB) ∧
  (∃ (CA : ℝ), dot_product = t.BC * CA * (-t.cosC)) := by
  sorry

end triangle_ABC_properties_l2827_282764


namespace f_properties_l2827_282724

noncomputable def f (x : ℝ) := 6 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)

theorem f_properties :
  (∃ (max_value : ℝ), ∀ (x : ℝ), f x ≤ max_value ∧ max_value = 2 * Real.sqrt 3 + 3) ∧
  (∃ (period : ℝ), period > 0 ∧ ∀ (x : ℝ), f (x + period) = f x ∧ 
    ∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 → 
    f α = 3 - 2 * Real.sqrt 3 → Real.tan (4 * α / 5) = Real.sqrt 3) :=
by sorry

end f_properties_l2827_282724


namespace factors_of_48_l2827_282702

/-- The number of distinct positive factors of 48 -/
def num_factors_48 : ℕ := (Nat.factors 48).card

/-- Theorem stating that the number of distinct positive factors of 48 is 10 -/
theorem factors_of_48 : num_factors_48 = 10 := by
  sorry

end factors_of_48_l2827_282702


namespace line_inclination_angle_l2827_282767

theorem line_inclination_angle (x y : ℝ) :
  let line_equation := (2 * x - 2 * y - 1 = 0)
  let slope := (2 : ℝ) / 2
  let angle_of_inclination := Real.arctan slope
  line_equation → angle_of_inclination = π / 4 := by
  sorry

end line_inclination_angle_l2827_282767


namespace log_sin_cos_theorem_l2827_282762

theorem log_sin_cos_theorem (x n : ℝ) 
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = (Real.log n - 2) / 2) : 
  n = Real.exp 2 + 2 := by
  sorry

end log_sin_cos_theorem_l2827_282762


namespace sum_of_nine_and_number_l2827_282713

theorem sum_of_nine_and_number (x : ℝ) : 
  (9 - x = 1) → (x < 10) → (9 + x = 17) := by
  sorry

end sum_of_nine_and_number_l2827_282713


namespace salary_calculation_l2827_282707

theorem salary_calculation (salary : ℝ) 
  (h1 : salary / 5 + salary / 10 + 3 * salary / 5 + 14000 = salary) : 
  salary = 140000 := by
sorry

end salary_calculation_l2827_282707


namespace production_line_uses_systematic_sampling_l2827_282734

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Systematic
  | Random
  | Stratified
  | Cluster

/-- Represents a production line with its characteristics --/
structure ProductionLine where
  daily_production : ℕ
  sampling_frequency : ℕ  -- days per week
  samples_per_day : ℕ
  sampling_start_time : ℕ  -- in minutes past midnight
  sampling_end_time : ℕ    -- in minutes past midnight

/-- Determines the sampling method based on production line characteristics --/
def determine_sampling_method (pl : ProductionLine) : SamplingMethod :=
  sorry  -- Proof to be implemented

/-- Theorem stating that the given production line uses systematic sampling --/
theorem production_line_uses_systematic_sampling (pl : ProductionLine) 
  (h1 : pl.daily_production = 128)
  (h2 : pl.sampling_frequency = 7)  -- weekly
  (h3 : pl.samples_per_day = 8)
  (h4 : pl.sampling_start_time = 14 * 60)  -- 2:00 PM
  (h5 : pl.sampling_end_time = 14 * 60 + 30)  -- 2:30 PM
  : determine_sampling_method pl = SamplingMethod.Systematic :=
by
  sorry  -- Proof to be implemented

end production_line_uses_systematic_sampling_l2827_282734


namespace points_in_segment_l2827_282731

theorem points_in_segment (n : ℕ) : 
  1 < (n^4 + n^2 + 2) / (n^4 + n^2 + 1) ∧ (n^4 + n^2 + 2) / (n^4 + n^2 + 1) ≤ 4/3 := by
  sorry

end points_in_segment_l2827_282731


namespace f_minimum_value_l2827_282729

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 5/2) ∧ (∃ x > 0, f x = 5/2) :=
by sorry

end f_minimum_value_l2827_282729


namespace least_prime_factor_of_5_4_minus_5_2_l2827_282773

theorem least_prime_factor_of_5_4_minus_5_2 :
  Nat.minFac (5^4 - 5^2) = 2 := by
  sorry

end least_prime_factor_of_5_4_minus_5_2_l2827_282773


namespace jeans_jail_sentence_l2827_282793

/-- Calculates the total jail sentence for Jean based on various charges --/
def total_jail_sentence (arson_counts : ℕ) (burglary_charges : ℕ) (arson_sentence : ℕ) (burglary_sentence : ℕ) : ℕ :=
  let petty_larceny_charges := 6 * burglary_charges
  let petty_larceny_sentence := burglary_sentence / 3
  arson_counts * arson_sentence +
  burglary_charges * burglary_sentence +
  petty_larceny_charges * petty_larceny_sentence

/-- Theorem stating that Jean's total jail sentence is 216 months --/
theorem jeans_jail_sentence :
  total_jail_sentence 3 2 36 18 = 216 := by
  sorry


end jeans_jail_sentence_l2827_282793


namespace unique_solution_complex_magnitude_and_inequality_l2827_282755

theorem unique_solution_complex_magnitude_and_inequality :
  ∃! (n : ℝ), n > 0 ∧ Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 ∧ n^2 + 5*n > 50 :=
by sorry

end unique_solution_complex_magnitude_and_inequality_l2827_282755


namespace algebraic_expression_value_l2827_282789

theorem algebraic_expression_value (x : ℝ) :
  12 * x - 8 * x^2 = -1 → 4 * x^2 - 6 * x + 5 = 5.5 := by
  sorry

end algebraic_expression_value_l2827_282789


namespace quadratic_equal_roots_l2827_282797

theorem quadratic_equal_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) → 
  a = 2 ∨ a = -2 := by
sorry

end quadratic_equal_roots_l2827_282797


namespace slope_from_angle_l2827_282736

theorem slope_from_angle (θ : Real) (h : θ = 5 * Real.pi / 6) :
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end slope_from_angle_l2827_282736


namespace largest_odd_sum_288_largest_odd_sum_288_is_43_l2827_282779

/-- Sum of first n consecutive odd integers -/
def sum_n_odd (n : ℕ) : ℕ := n^2

/-- Sum of odd integers from a to b inclusive -/
def sum_odd_range (a b : ℕ) : ℕ := 
  (sum_n_odd ((b - a) / 2 + 1)) - (sum_n_odd ((a - 1) / 2))

/-- The largest odd integer x such that the sum of all odd integers 
    from 13 to x inclusive is 288 -/
theorem largest_odd_sum_288 : 
  ∃ x : ℕ, x % 2 = 1 ∧ sum_odd_range 13 x = 288 ∧ 
  ∀ y : ℕ, y > x → y % 2 = 1 → sum_odd_range 13 y > 288 :=
sorry
 
/-- The largest odd integer x such that the sum of all odd integers 
    from 13 to x inclusive is 288 is equal to 43 -/
theorem largest_odd_sum_288_is_43 : 
  ∃! x : ℕ, x % 2 = 1 ∧ sum_odd_range 13 x = 288 ∧ 
  ∀ y : ℕ, y > x → y % 2 = 1 → sum_odd_range 13 y > 288 ∧ x = 43 :=
sorry

end largest_odd_sum_288_largest_odd_sum_288_is_43_l2827_282779


namespace sin_cos_sum_equals_negative_one_l2827_282766

open Real

theorem sin_cos_sum_equals_negative_one :
  sin (200 * π / 180) * cos (110 * π / 180) + cos (160 * π / 180) * sin (70 * π / 180) = -1 := by
  sorry

end sin_cos_sum_equals_negative_one_l2827_282766


namespace greg_original_seat_l2827_282778

/-- Represents a seat in the theater --/
inductive Seat
| one
| two
| three
| four
| five

/-- Represents a friend --/
inductive Friend
| Greg
| Iris
| Jamal
| Kim
| Leo

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Represents a movement of a friend --/
def Movement := Friend → Int

theorem greg_original_seat 
  (initial_arrangement : Arrangement)
  (final_arrangement : Arrangement)
  (movements : Movement) :
  (movements Friend.Iris = 1) →
  (movements Friend.Jamal = -2) →
  (movements Friend.Kim + movements Friend.Leo = 0) →
  (final_arrangement Friend.Greg = Seat.one) →
  (initial_arrangement Friend.Greg = Seat.two) :=
sorry

end greg_original_seat_l2827_282778


namespace cube_root_equation_solution_l2827_282708

theorem cube_root_equation_solution :
  ∀ x : ℝ, (7 - 3 / (3 + x))^(1/3) = -2 → x = -14/5 := by
  sorry

end cube_root_equation_solution_l2827_282708


namespace units_digit_17_pow_2023_l2827_282709

theorem units_digit_17_pow_2023 : ∃ k : ℕ, 17^2023 ≡ 3 [ZMOD 10] :=
by sorry

end units_digit_17_pow_2023_l2827_282709


namespace square_side_ratio_l2827_282788

theorem square_side_ratio (area_ratio : ℚ) (h : area_ratio = 72 / 98) :
  ∃ (a b c : ℕ), 
    (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) ∧ 
    a = 6 ∧ 
    b = 1 ∧ 
    c = 7 ∧ 
    a + b + c = 14 := by
  sorry

end square_side_ratio_l2827_282788


namespace books_left_over_l2827_282733

/-- Given a repacking scenario, proves the number of books left over -/
theorem books_left_over 
  (initial_boxes : ℕ) 
  (books_per_initial_box : ℕ) 
  (book_weight : ℕ) 
  (books_per_new_box : ℕ) 
  (max_new_box_weight : ℕ) 
  (h1 : initial_boxes = 1430)
  (h2 : books_per_initial_box = 42)
  (h3 : book_weight = 200)
  (h4 : books_per_new_box = 45)
  (h5 : max_new_box_weight = 9000)
  (h6 : books_per_new_box * book_weight ≤ max_new_box_weight) :
  (initial_boxes * books_per_initial_box) % books_per_new_box = 30 := by
  sorry

#check books_left_over

end books_left_over_l2827_282733


namespace legs_fraction_of_height_l2827_282728

/-- Represents the height measurements of a person --/
structure PersonHeight where
  total : ℝ
  head : ℝ
  restOfBody : ℝ

/-- Theorem stating the fraction of total height occupied by legs --/
theorem legs_fraction_of_height (p : PersonHeight) 
  (h_total : p.total = 60)
  (h_head : p.head = 1/4 * p.total)
  (h_rest : p.restOfBody = 25) :
  (p.total - p.head - p.restOfBody) / p.total = 1/3 := by
  sorry

#check legs_fraction_of_height

end legs_fraction_of_height_l2827_282728


namespace non_perfect_power_probability_l2827_282761

/-- A function that determines if a natural number is a perfect power --/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are not perfect powers --/
def nonPerfectPowerCount : ℕ := 178

/-- The total count of numbers from 1 to 200 --/
def totalCount : ℕ := 200

/-- The probability of selecting a non-perfect power from 1 to 200 --/
def probabilityNonPerfectPower : ℚ := 89 / 100

theorem non_perfect_power_probability :
  (nonPerfectPowerCount : ℚ) / (totalCount : ℚ) = probabilityNonPerfectPower :=
sorry

end non_perfect_power_probability_l2827_282761


namespace abc_fraction_l2827_282756

theorem abc_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b / (a + b) = 3)
  (hbc : b * c / (b + c) = 4)
  (hca : c * a / (c + a) = 5) :
  a * b * c / (a * b + b * c + c * a) = 120 / 47 := by
sorry

end abc_fraction_l2827_282756


namespace divisibility_equivalence_l2827_282753

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + sequence_a n

theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ sequence_a n ↔ 2^k ∣ n := by sorry

end divisibility_equivalence_l2827_282753


namespace find_A_l2827_282738

theorem find_A (A B C : ℚ) 
  (h1 : A = (1 / 2) * B) 
  (h2 : B = (3 / 4) * C) 
  (h3 : A + C = 55) : 
  A = 15 := by
sorry

end find_A_l2827_282738


namespace map_width_l2827_282712

/-- The width of a rectangular map given its length and area -/
theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) :
  area / length = 10 := by
  sorry

end map_width_l2827_282712


namespace min_distance_hyperbola_circle_l2827_282727

theorem min_distance_hyperbola_circle (a b c d : ℝ) 
  (h1 : a * b = 1) (h2 : c^2 + d^2 = 1) : 
  ∃ (min : ℝ), min = 3 - 2 * Real.sqrt 2 ∧ 
  ∀ (x y z w : ℝ), x * y = 1 → z^2 + w^2 = 1 → 
  (x - z)^2 + (y - w)^2 ≥ min := by
  sorry

end min_distance_hyperbola_circle_l2827_282727


namespace double_series_convergence_l2827_282742

/-- The double series ∑_{m=1}^∞ ∑_{n=1}^∞ 1/(mn(m+n+2)) converges to 3/2. -/
theorem double_series_convergence :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = 3/2 := by
  sorry

end double_series_convergence_l2827_282742


namespace dot_product_range_l2827_282770

/-- The range of the dot product OP · BA -/
theorem dot_product_range (O A B P : ℝ × ℝ) : 
  O = (0, 0) →
  A = (2, 0) →
  B = (1, -2 * Real.sqrt 3) →
  (∃ (x : ℝ), P.1 = x ∧ P.2 = Real.sqrt (1 - x^2 / 4)) →
  -2 ≤ (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) ∧
  (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) ≤ 4 :=
by sorry

end dot_product_range_l2827_282770


namespace function_value_order_l2827_282752

-- Define the function f
def f (a x : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

-- State the theorem
theorem function_value_order (a : ℝ) : 
  f a (Real.sqrt 2) < f a 4 ∧ f a 4 < f a 3 := by
  sorry

end function_value_order_l2827_282752


namespace garden_area_increase_l2827_282759

theorem garden_area_increase : 
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)
  let square_side : ℝ := rectangle_perimeter / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let square_area : ℝ := square_side * square_side
  square_area - rectangle_area = 400 := by
sorry

end garden_area_increase_l2827_282759


namespace cards_distribution_l2827_282772

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 48) (h2 : num_people = 7) :
  let cards_per_person := total_cards / num_people
  let remaining_cards := total_cards % num_people
  num_people - remaining_cards = 1 := by
  sorry

end cards_distribution_l2827_282772


namespace distance_from_blast_site_l2827_282700

/-- Proves the distance a man is from a blast site when he hears a second blast -/
theorem distance_from_blast_site (speed_of_sound : ℝ) (time_between_blasts : ℝ) (time_heard_second_blast : ℝ) : 
  speed_of_sound = 330 →
  time_between_blasts = 30 →
  time_heard_second_blast = 30 + 12 / 60 →
  speed_of_sound * (time_heard_second_blast - time_between_blasts) = 3960 := by
  sorry

end distance_from_blast_site_l2827_282700


namespace simplify_and_rationalize_l2827_282765

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 7) * (Real.sqrt 6 / Real.sqrt 14) * (Real.sqrt 9 / Real.sqrt 21) = Real.sqrt 2058 / 114 := by
  sorry

end simplify_and_rationalize_l2827_282765


namespace fraction_simplification_l2827_282748

theorem fraction_simplification (a b m : ℝ) (h : a + b ≠ 0) :
  (m * a) / (a + b) + (m * b) / (a + b) = m := by
  sorry

end fraction_simplification_l2827_282748


namespace letters_theorem_l2827_282737

def total_letters (brother_letters : ℕ) : ℕ :=
  let greta_letters := brother_letters + 10
  let mother_letters := 2 * (brother_letters + greta_letters)
  brother_letters + greta_letters + mother_letters

theorem letters_theorem : total_letters 40 = 270 := by
  sorry

end letters_theorem_l2827_282737


namespace prescription_final_cost_l2827_282758

/-- Calculates the final cost of a prescription after cashback and rebate --/
theorem prescription_final_cost 
  (original_price : ℝ) 
  (cashback_percent : ℝ) 
  (rebate : ℝ) 
  (h1 : original_price = 150)
  (h2 : cashback_percent = 0.1)
  (h3 : rebate = 25) :
  original_price - (cashback_percent * original_price) - rebate = 110 := by
  sorry

#check prescription_final_cost

end prescription_final_cost_l2827_282758


namespace solution_set_f_range_of_m_l2827_282704

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for part I
theorem solution_set_f (x : ℝ) : f x < 8 ↔ -5/2 < x ∧ x < 3/2 :=
sorry

-- Theorem for part II
theorem range_of_m (m : ℝ) : (∃ x, f x ≤ |3*m + 1|) → (m ≤ -5/3 ∨ m ≥ 1) :=
sorry

end solution_set_f_range_of_m_l2827_282704


namespace triangle_theorem_l2827_282718

noncomputable def triangle_problem (a b c A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  let m_x := m.1
  let m_y := m.2
  let n_x := n.1
  let n_y := n.2
  (m_x = Real.sin A ∧ m_y = 1) ∧ 
  (n_x = Real.cos A ∧ n_y = Real.sqrt 3) ∧
  (m_x / n_x = m_y / n_y) ∧  -- parallel vectors condition
  (a = 2) ∧ 
  (b = 2 * Real.sqrt 2) ∧
  (A = Real.pi / 6) ∧
  ((1 / 2 * a * b * Real.sin C = 1 + Real.sqrt 3) ∨ 
   (1 / 2 * a * b * Real.sin C = Real.sqrt 3 - 1))

theorem triangle_theorem (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  triangle_problem a b c A B C m n := by sorry

end triangle_theorem_l2827_282718


namespace negation_of_proposition_l2827_282796

theorem negation_of_proposition (a b : ℝ) :
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) := by sorry

end negation_of_proposition_l2827_282796


namespace find_decrease_rate_village_x_decrease_rate_l2827_282782

/-- Represents the population change in two villages over time -/
def village_population_equality (x_initial : ℕ) (y_initial : ℕ) (y_growth_rate : ℕ) (years : ℕ) (x_decrease_rate : ℕ) : Prop :=
  x_initial - years * x_decrease_rate = y_initial + years * y_growth_rate

/-- Theorem stating the condition for equal populations after a given time -/
theorem find_decrease_rate (x_initial y_initial y_growth_rate years : ℕ) :
  ∃ (x_decrease_rate : ℕ),
    village_population_equality x_initial y_initial y_growth_rate years x_decrease_rate ∧
    x_decrease_rate = (x_initial - y_initial - years * y_growth_rate) / years :=
by
  sorry

/-- Application of the theorem to the specific problem -/
theorem village_x_decrease_rate :
  ∃ (x_decrease_rate : ℕ),
    village_population_equality 76000 42000 800 17 x_decrease_rate ∧
    x_decrease_rate = 1200 :=
by
  sorry

end find_decrease_rate_village_x_decrease_rate_l2827_282782


namespace square_of_binomial_constant_l2827_282714

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end square_of_binomial_constant_l2827_282714


namespace cookie_calorie_count_l2827_282777

/-- The number of calories in each cracker -/
def cracker_calories : ℕ := 15

/-- The number of cookies Jimmy eats -/
def cookies_eaten : ℕ := 7

/-- The number of crackers Jimmy eats -/
def crackers_eaten : ℕ := 10

/-- The total number of calories Jimmy consumes -/
def total_calories : ℕ := 500

/-- The number of calories in each cookie -/
def cookie_calories : ℕ := 50

theorem cookie_calorie_count :
  cookie_calories * cookies_eaten + cracker_calories * crackers_eaten = total_calories :=
by sorry

end cookie_calorie_count_l2827_282777


namespace polynomial_equality_l2827_282720

theorem polynomial_equality (x : ℝ) : 
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = (2*x)^5 := by
  sorry

end polynomial_equality_l2827_282720


namespace no_root_greater_than_four_l2827_282717

-- Define the three equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 15 = 35
def equation2 (x : ℝ) : Prop := (3*x - 2)^2 = (2*x - 3)^2
def equation3 (x : ℝ) : Prop := (x^2 - 16 : ℝ) = 2*x - 4

-- Theorem stating that no root is greater than 4 for all equations
theorem no_root_greater_than_four :
  (∀ x > 4, ¬ equation1 x) ∧
  (∀ x > 4, ¬ equation2 x) ∧
  (∀ x > 4, ¬ equation3 x) :=
by sorry

end no_root_greater_than_four_l2827_282717


namespace tylers_puppies_l2827_282732

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) (h1 : num_dogs = 25) (h2 : puppies_per_dog = 8) :
  num_dogs * puppies_per_dog = 200 := by
  sorry

end tylers_puppies_l2827_282732


namespace function_count_l2827_282719

theorem function_count (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x * y) + f x + f y - f x * f y ≥ 2) ↔ 
  (∀ x : ℝ, f x = 1 ∨ f x = 2) :=
sorry

end function_count_l2827_282719


namespace trigonometric_identities_l2827_282763

theorem trigonometric_identities (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 2) : 
  (Real.sin θ * Real.cos θ = 1/2) ∧ 
  ((Real.sin θ + Real.cos θ)^2 = 2) := by
  sorry

end trigonometric_identities_l2827_282763


namespace quadratic_equation_roots_l2827_282790

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -3 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, x^2 + 4*x + 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_roots_l2827_282790


namespace distinct_d_values_l2827_282705

theorem distinct_d_values (a b c : ℂ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃! (s : Finset ℂ), s.card = 6 ∧ 
  (∀ d : ℂ, d ∈ s ↔ 
    (∀ z : ℂ, (z - a) * (z - b) * (z - c) = (z - d^2 * a) * (z - d^2 * b) * (z - d^2 * c))) :=
by sorry

end distinct_d_values_l2827_282705


namespace sphere_in_dihedral_angle_l2827_282783

/-- Given a sphere of unit radius with its center on the edge of a dihedral angle α,
    the radius r of a new sphere whose volume equals the volume of the part of the given sphere
    that lies inside the dihedral angle is r = ∛(α / (2π)). -/
theorem sphere_in_dihedral_angle (α : Real) (h : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (r : Real), r = (α / (2 * Real.pi)) ^ (1/3) ∧
  (4/3 * Real.pi * r^3) = (α / (2 * Real.pi)) * (4/3 * Real.pi) := by
  sorry

end sphere_in_dihedral_angle_l2827_282783


namespace wall_length_proof_l2827_282735

/-- Given a wall with specified dimensions and number of bricks, prove its length --/
theorem wall_length_proof (wall_height : ℝ) (wall_thickness : ℝ) 
  (brick_count : ℝ) (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) :
  wall_height = 100 →
  wall_thickness = 5 →
  brick_count = 242.42424242424244 →
  brick_length = 25 →
  brick_width = 11 →
  brick_height = 6 →
  (brick_length * brick_width * brick_height * brick_count) / (wall_height * wall_thickness) = 800 := by
  sorry

#check wall_length_proof

end wall_length_proof_l2827_282735


namespace system_is_linear_l2827_282744

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- A system of two equations is linear if both equations are linear and they involve exactly two variables. -/
def is_linear_system (f g : ℝ → ℝ → ℝ) : Prop :=
  is_linear_equation f ∧ is_linear_equation g

/-- The given system of equations -/
def equation1 (x y : ℝ) : ℝ := x - y - 11
def equation2 (x y : ℝ) : ℝ := 4 * x - y - 1

theorem system_is_linear : is_linear_system equation1 equation2 := by
  sorry

end system_is_linear_l2827_282744


namespace log_one_over_twenty_five_base_five_l2827_282746

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_twenty_five_base_five : log 5 (1/25) = -2 := by
  sorry

end log_one_over_twenty_five_base_five_l2827_282746


namespace equation_roots_l2827_282743

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (18 / (x^2 - 9)) - (3 / (x - 3)) - 2
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -4.5 := by
  sorry

end equation_roots_l2827_282743


namespace chen_trigonometric_problem_l2827_282771

theorem chen_trigonometric_problem :
  ∃ (N : ℕ) (α β γ θ : ℝ),
    0.1 = Real.sin γ * Real.cos θ * Real.sin α ∧
    0.2 = Real.sin γ * Real.sin θ * Real.cos α ∧
    0.3 = Real.cos γ * Real.cos θ * Real.sin β ∧
    0.4 = Real.cos γ * Real.sin θ * Real.cos β ∧
    0.5 ≥ |N - 100 * Real.cos (2 * θ)| ∧
    N = 79 := by
  sorry

end chen_trigonometric_problem_l2827_282771


namespace special_ellipse_eccentricity_l2827_282787

/-- An ellipse with a special point P -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  P : ℝ × ℝ
  h_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1
  h_PF1_perpendicular : P.1 = -((a^2 - b^2).sqrt)
  h_PF2_parallel : P.2 / (P.1 + ((a^2 - b^2).sqrt)) = -b / a

/-- The eccentricity of an ellipse with a special point P is √5/5 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) :
  ((E.a^2 - E.b^2) / E.a^2).sqrt = (5 : ℝ).sqrt / 5 := by
  sorry

end special_ellipse_eccentricity_l2827_282787


namespace triangle_area_inequalities_l2827_282710

/-- Given two triangles and a third triangle constructed from their sides, 
    the area of the third triangle is greater than or equal to 
    both the geometric and arithmetic means of the areas of the original triangles. -/
theorem triangle_area_inequalities 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h₂ : 0 < a₂ ∧ 0 < b₂ ∧ 0 < c₂)
  (h₃ : a₁ + b₁ > c₁ ∧ a₁ + c₁ > b₁ ∧ b₁ + c₁ > a₁)
  (h₄ : a₂ + b₂ > c₂ ∧ a₂ + c₂ > b₂ ∧ b₂ + c₂ > a₂)
  (a : ℝ) (ha : a = Real.sqrt ((a₁^2 + a₂^2) / 2))
  (b : ℝ) (hb : b = Real.sqrt ((b₁^2 + b₂^2) / 2))
  (c : ℝ) (hc : c = Real.sqrt ((c₁^2 + c₂^2) / 2))
  (h₅ : a + b > c ∧ a + c > b ∧ b + c > a)
  (S₁ : ℝ) (hS₁ : S₁ = Real.sqrt (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)))
  (S₂ : ℝ) (hS₂ : S₂ = Real.sqrt (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)))
  (S : ℝ)  (hS : S = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (s₁ : ℝ) (hs₁ : s₁ = (a₁ + b₁ + c₁) / 2)
  (s₂ : ℝ) (hs₂ : s₂ = (a₂ + b₂ + c₂) / 2)
  (s : ℝ)  (hs : s = (a + b + c) / 2) :
  S ≥ Real.sqrt (S₁ * S₂) ∧ S ≥ (S₁ + S₂) / 2 := by
  sorry

end triangle_area_inequalities_l2827_282710


namespace boat_production_l2827_282716

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boat_production : geometric_sum 5 3 4 = 200 := by
  sorry

end boat_production_l2827_282716


namespace percent_relation_l2827_282760

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.30 * a) 
  (h2 : c = 0.25 * b) : 
  b = 1.2 * a := by
sorry

end percent_relation_l2827_282760


namespace fraction_equality_l2827_282740

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 2) 
  (h2 : s / u = 7 / 11) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := by
  sorry

end fraction_equality_l2827_282740


namespace equation_solutions_l2827_282703

theorem equation_solutions : 
  let solutions := {x : ℝ | (x - 1)^2 = 4}
  solutions = {3, -1} := by sorry

end equation_solutions_l2827_282703


namespace polynomial_subtraction_l2827_282725

theorem polynomial_subtraction (x : ℝ) :
  (4*x - 3) * (x + 6) - (2*x + 1) * (x - 4) = 2*x^2 + 28*x - 14 := by
  sorry

end polynomial_subtraction_l2827_282725


namespace simplification_problems_l2827_282730

theorem simplification_problems :
  ((-1/2 + 2/3 - 1/4) / (-1/24) = 2) ∧
  (7/2 * (-5/7) - (-5/7) * 5/2 - 5/7 * (-1/2) = -5/14) := by
  sorry

end simplification_problems_l2827_282730


namespace class_test_problem_l2827_282799

theorem class_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.45 := by
sorry

end class_test_problem_l2827_282799


namespace third_vertex_coordinates_l2827_282749

/-- Given a triangle with vertices at (7, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 24 square units, then x = -48/√58. -/
theorem third_vertex_coordinates (x : ℝ) :
  x < 0 →
  (1/2 : ℝ) * |x| * 3 * Real.sqrt 58 = 24 →
  x = -48 / Real.sqrt 58 := by
sorry

end third_vertex_coordinates_l2827_282749


namespace sqrt_mixed_number_simplification_l2827_282791

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 :=
by
  sorry

end sqrt_mixed_number_simplification_l2827_282791
