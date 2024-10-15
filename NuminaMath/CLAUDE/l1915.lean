import Mathlib

namespace NUMINAMATH_CALUDE_correct_sample_size_l1915_191577

-- Define the population size
def population_size : ℕ := 5000

-- Define the number of sampled students
def sampled_students : ℕ := 450

-- Define what sample size means in this context
def sample_size (n : ℕ) : Prop := n = sampled_students

-- Theorem stating that the sample size is 450
theorem correct_sample_size : sample_size 450 := by sorry

end NUMINAMATH_CALUDE_correct_sample_size_l1915_191577


namespace NUMINAMATH_CALUDE_x_power_twelve_equals_one_l1915_191560

theorem x_power_twelve_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twelve_equals_one_l1915_191560


namespace NUMINAMATH_CALUDE_abc_inequality_l1915_191570

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1915_191570


namespace NUMINAMATH_CALUDE_album_jumps_l1915_191558

/-- Calculates the total number of jumps a person can make while listening to an album. -/
theorem album_jumps (jumps_per_second : ℕ) (song_length : ℚ) (num_songs : ℕ) :
  jumps_per_second = 1 →
  song_length = 3.5 →
  num_songs = 10 →
  (jumps_per_second * 60 : ℚ) * (song_length * num_songs) = 2100 := by
  sorry

end NUMINAMATH_CALUDE_album_jumps_l1915_191558


namespace NUMINAMATH_CALUDE_periodic_function_l1915_191549

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f 1 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_l1915_191549


namespace NUMINAMATH_CALUDE_length_breadth_difference_l1915_191507

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_24_times_breadth : area = 24 * breadth
  breadth_is_14 : breadth = 14
  area_def : area = length * breadth

/-- The difference between length and breadth is 10 meters -/
theorem length_breadth_difference (plot : RectangularPlot) : 
  plot.length - plot.breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_breadth_difference_l1915_191507


namespace NUMINAMATH_CALUDE_investment_principal_calculation_l1915_191537

/-- Proves that given a monthly interest payment of $234 and a simple annual interest rate of 9%,
    the principal amount of the investment is $31,200. -/
theorem investment_principal_calculation (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 234 →
  annual_rate = 0.09 →
  (monthly_interest * 12) / annual_rate = 31200 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_calculation_l1915_191537


namespace NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l1915_191503

/-- Represents the state of the water tank -/
structure TankState where
  water : ℚ
  antifreeze : ℚ

/-- Performs one replacement operation on the tank -/
def replace (state : TankState) : TankState :=
  let removed := state.water * (5 / 20) + state.antifreeze * (5 / 20)
  { water := state.water - removed + 2.5,
    antifreeze := state.antifreeze - removed + 2.5 }

/-- The initial state of the tank -/
def initialState : TankState :=
  { water := 20, antifreeze := 0 }

/-- Performs n replacements on the tank -/
def nReplacements (n : ℕ) : TankState :=
  match n with
  | 0 => initialState
  | n + 1 => replace (nReplacements n)

theorem water_fraction_after_four_replacements :
  (nReplacements 4).water / ((nReplacements 4).water + (nReplacements 4).antifreeze) = 21 / 32 :=
by sorry

end NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l1915_191503


namespace NUMINAMATH_CALUDE_max_homework_time_l1915_191536

def homework_time (biology_time : ℕ) : ℕ :=
  let history_time := 2 * biology_time
  let geography_time := 3 * history_time
  biology_time + history_time + geography_time

theorem max_homework_time :
  homework_time 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_max_homework_time_l1915_191536


namespace NUMINAMATH_CALUDE_apple_boxes_bought_l1915_191523

-- Define the variables
variable (cherry_price : ℝ) -- Price of one cherry
variable (apple_price : ℝ) -- Price of one apple
variable (cherry_size : ℝ) -- Size of one cherry
variable (apple_size : ℝ) -- Size of one apple
variable (cherries_per_box : ℕ) -- Number of cherries in a box

-- Define the conditions
axiom price_relation : 2 * cherry_price = 3 * apple_price
axiom size_relation : apple_size = 12 * cherry_size
axiom box_size_equality : cherries_per_box * cherry_size = cherries_per_box * apple_size

-- Define the theorem
theorem apple_boxes_bought (h : cherries_per_box > 0) :
  (cherries_per_box * cherry_price) / apple_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_bought_l1915_191523


namespace NUMINAMATH_CALUDE_two_alarms_parallel_reliability_l1915_191544

/-- The reliability of a single alarm -/
def single_alarm_reliability : ℝ := 0.90

/-- The reliability of two independent alarms connected in parallel -/
def parallel_reliability (p : ℝ) : ℝ := 1 - (1 - p) * (1 - p)

theorem two_alarms_parallel_reliability :
  parallel_reliability single_alarm_reliability = 0.99 := by
  sorry

end NUMINAMATH_CALUDE_two_alarms_parallel_reliability_l1915_191544


namespace NUMINAMATH_CALUDE_train_speed_problem_l1915_191543

theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let total_distance := 3 * x
  let first_distance := x
  let second_distance := 2 * x
  let second_speed := 20
  let average_speed := 26
  let time_first := first_distance / V
  let time_second := second_distance / second_speed
  let total_time := time_first + time_second
  average_speed = total_distance / total_time →
  V = 65 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1915_191543


namespace NUMINAMATH_CALUDE_program_flowchart_components_l1915_191597

-- Define a program flowchart
structure ProgramFlowchart where
  is_diagram : Bool
  represents_algorithm : Bool
  uses_specified_shapes : Bool
  uses_directional_lines : Bool
  uses_textual_explanations : Bool

-- Define the components of a program flowchart
structure FlowchartComponents where
  has_operation_boxes : Bool
  has_flow_lines_with_arrows : Bool
  has_textual_explanations : Bool

-- Theorem statement
theorem program_flowchart_components 
  (pf : ProgramFlowchart) 
  (h1 : pf.is_diagram = true)
  (h2 : pf.represents_algorithm = true)
  (h3 : pf.uses_specified_shapes = true)
  (h4 : pf.uses_directional_lines = true)
  (h5 : pf.uses_textual_explanations = true) :
  ∃ (fc : FlowchartComponents), 
    fc.has_operation_boxes = true ∧ 
    fc.has_flow_lines_with_arrows = true ∧ 
    fc.has_textual_explanations = true :=
  sorry

end NUMINAMATH_CALUDE_program_flowchart_components_l1915_191597


namespace NUMINAMATH_CALUDE_abs_neg_2022_l1915_191566

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2022_l1915_191566


namespace NUMINAMATH_CALUDE_other_divisor_proof_l1915_191579

theorem other_divisor_proof (x : ℕ) (h : x > 0) : 
  (261 % 37 = 2 ∧ 261 % x = 2) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_other_divisor_proof_l1915_191579


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1915_191509

/-- The longest segment in a cylinder with radius 5 and height 12 is 2√61 -/
theorem longest_segment_in_cylinder : 
  let r : ℝ := 5
  let h : ℝ := 12
  let longest_segment := Real.sqrt ((2 * r) ^ 2 + h ^ 2)
  longest_segment = 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1915_191509


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1915_191541

theorem contrapositive_equivalence (a b : ℝ) :
  (ab = 0 → a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0 → ab ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1915_191541


namespace NUMINAMATH_CALUDE_adjacent_pair_with_distinct_roots_l1915_191524

/-- Represents a 6x6 grid containing integers from 1 to 36 --/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- Checks if two numbers are adjacent in a row --/
def areAdjacent (grid : Grid) (i j : Fin 6) (k : Fin 5) : Prop :=
  grid i k = j ∧ grid i (k + 1) = j + 1 ∨ grid i k = j + 1 ∧ grid i (k + 1) = j

/-- Checks if a quadratic equation has two distinct real roots --/
def hasTwoDistinctRealRoots (p q : ℕ) : Prop :=
  p * p > 4 * q

theorem adjacent_pair_with_distinct_roots (grid : Grid) :
  ∃ (i : Fin 6) (j : Fin 36) (k : Fin 5),
    areAdjacent grid j (j + 1) k ∧
    hasTwoDistinctRealRoots j (j + 1) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_pair_with_distinct_roots_l1915_191524


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1915_191599

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1915_191599


namespace NUMINAMATH_CALUDE_ball_probabilities_l1915_191533

theorem ball_probabilities (total_balls : ℕ) (red_prob black_prob white_prob green_prob : ℚ)
  (h_total : total_balls = 12)
  (h_red : red_prob = 5 / 12)
  (h_black : black_prob = 1 / 3)
  (h_white : white_prob = 1 / 6)
  (h_green : green_prob = 1 / 12)
  (h_sum : red_prob + black_prob + white_prob + green_prob = 1) :
  (red_prob + black_prob = 3 / 4) ∧ (red_prob + black_prob + white_prob = 11 / 12) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1915_191533


namespace NUMINAMATH_CALUDE_fifth_pile_magazines_l1915_191519

def magazine_sequence : ℕ → ℕ
  | 0 => 3
  | 1 => 4
  | 2 => 6
  | 3 => 9
  | n + 4 => magazine_sequence n + (n + 1)

theorem fifth_pile_magazines : magazine_sequence 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fifth_pile_magazines_l1915_191519


namespace NUMINAMATH_CALUDE_seashells_remaining_l1915_191522

theorem seashells_remaining (initial_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : given_seashells = 43) : 
  initial_seashells - given_seashells = 27 := by
  sorry

end NUMINAMATH_CALUDE_seashells_remaining_l1915_191522


namespace NUMINAMATH_CALUDE_parabola_directrix_l1915_191587

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -1/16

/-- Theorem: The directrix of the parabola y = 4x^2 is y = -1/16 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1915_191587


namespace NUMINAMATH_CALUDE_yield_difference_l1915_191508

-- Define the initial yields and growth rates
def tomatoes_initial : ℝ := 2073
def corn_initial : ℝ := 4112
def onions_initial : ℝ := 985
def carrots_initial : ℝ := 6250

def tomatoes_growth_rate : ℝ := 0.12
def corn_growth_rate : ℝ := 0.15
def onions_growth_rate : ℝ := 0.08
def carrots_growth_rate : ℝ := 0.10

-- Calculate the yields after growth
def tomatoes_yield : ℝ := tomatoes_initial * (1 + tomatoes_growth_rate)
def corn_yield : ℝ := corn_initial * (1 + corn_growth_rate)
def onions_yield : ℝ := onions_initial * (1 + onions_growth_rate)
def carrots_yield : ℝ := carrots_initial * (1 + carrots_growth_rate)

-- Define the theorem
theorem yield_difference : 
  (max tomatoes_yield (max corn_yield (max onions_yield carrots_yield))) - 
  (min tomatoes_yield (min corn_yield (min onions_yield carrots_yield))) = 5811.2 := by
  sorry

end NUMINAMATH_CALUDE_yield_difference_l1915_191508


namespace NUMINAMATH_CALUDE_total_cement_is_15_1_l1915_191593

/-- The amount of cement used for Lexi's street in tons -/
def lexis_street_cement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tesss_street_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def total_cement : ℝ := lexis_street_cement + tesss_street_cement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_1 : total_cement = 15.1 := by
  sorry

end NUMINAMATH_CALUDE_total_cement_is_15_1_l1915_191593


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l1915_191564

theorem quadratic_inequalities :
  (∀ x : ℝ, 2 * x^2 + x + 1 > 0) ∧
  (∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x + 2 > 0 ↔ -1/2 < x ∧ x < 2) ∧ a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l1915_191564


namespace NUMINAMATH_CALUDE_sum_of_first_10_odd_numbers_l1915_191548

def sum_of_odd_numbers (n : ℕ) : ℕ := n^2

theorem sum_of_first_10_odd_numbers : sum_of_odd_numbers 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_10_odd_numbers_l1915_191548


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l1915_191584

theorem sum_of_square_areas (side1 side2 : ℝ) (h1 : side1 = 11) (h2 : side2 = 5) :
  side1 * side1 + side2 * side2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l1915_191584


namespace NUMINAMATH_CALUDE_product_equals_one_l1915_191525

theorem product_equals_one (x₁ x₂ x₃ : ℝ) 
  (h_nonneg₁ : x₁ ≥ 0) (h_nonneg₂ : x₂ ≥ 0) (h_nonneg₃ : x₃ ≥ 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1915_191525


namespace NUMINAMATH_CALUDE_childrens_cookbook_cost_l1915_191555

theorem childrens_cookbook_cost (dictionary_cost dinosaur_book_cost savings needed_more total_cost : ℕ) :
  dictionary_cost = 11 →
  dinosaur_book_cost = 19 →
  savings = 8 →
  needed_more = 29 →
  total_cost = savings + needed_more →
  total_cost - (dictionary_cost + dinosaur_book_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_childrens_cookbook_cost_l1915_191555


namespace NUMINAMATH_CALUDE_solution_set_implies_m_equals_one_l1915_191532

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*m*x - 4

-- State the theorem
theorem solution_set_implies_m_equals_one :
  (∀ x : ℝ, f m x < 0 ↔ -4 < x ∧ x < 1) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_equals_one_l1915_191532


namespace NUMINAMATH_CALUDE_max_apartment_size_l1915_191553

/-- Given:
  * The rental rate in Greenview is $1.20 per square foot.
  * Max's monthly budget for rent is $720.
  Prove that the largest apartment size Max can afford is 600 square feet. -/
theorem max_apartment_size (rental_rate : ℝ) (max_budget : ℝ) (max_size : ℝ) : 
  rental_rate = 1.20 →
  max_budget = 720 →
  max_size * rental_rate = max_budget →
  max_size = 600 := by
  sorry

#check max_apartment_size

end NUMINAMATH_CALUDE_max_apartment_size_l1915_191553


namespace NUMINAMATH_CALUDE_max_x_for_perfect_square_l1915_191592

theorem max_x_for_perfect_square : 
  ∀ x : ℕ, x > 1972 → ¬(∃ y : ℕ, 4^27 + 4^1000 + 4^x = y^2) ∧ 
  ∃ y : ℕ, 4^27 + 4^1000 + 4^1972 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_max_x_for_perfect_square_l1915_191592


namespace NUMINAMATH_CALUDE_school_teachers_count_l1915_191595

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sampled_students : ℕ) : 
  total = 2400 →
  sample_size = 160 →
  sampled_students = 150 →
  (total : ℚ) / sample_size = 15 →
  total - (sampled_students * ((total : ℚ) / sample_size).floor) = 150 :=
by sorry

end NUMINAMATH_CALUDE_school_teachers_count_l1915_191595


namespace NUMINAMATH_CALUDE_computer_price_reduction_l1915_191528

/-- Given a computer with original price x, after reducing it by m yuan and then by 20%,
    resulting in a final price of n yuan, prove that the original price x is equal to (5/4)n + m. -/
theorem computer_price_reduction (x m n : ℝ) (h : (x - m) * (1 - 0.2) = n) :
  x = (5/4) * n + m := by
  sorry

end NUMINAMATH_CALUDE_computer_price_reduction_l1915_191528


namespace NUMINAMATH_CALUDE_odd_even_functions_inequality_l1915_191500

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_functions_inequality (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h_diff : ∀ x, f x - g x = (1/2)^x) :
  g 1 < f 0 ∧ f 0 < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_odd_even_functions_inequality_l1915_191500


namespace NUMINAMATH_CALUDE_function_shift_l1915_191506

/-- Given a function f with the specified properties, prove that g can be obtained
    by shifting f to the left by π/8 units. -/
theorem function_shift (ω : ℝ) (h1 : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 4)
  let g : ℝ → ℝ := λ x ↦ Real.cos (ω * x)
  (∀ x, f (x + π / ω) = f x) →  -- minimum positive period is π
  ∀ x, g x = f (x + π / 8) := by
sorry

end NUMINAMATH_CALUDE_function_shift_l1915_191506


namespace NUMINAMATH_CALUDE_crocodile_count_correct_l1915_191573

/-- Represents the number of crocodiles in the pond -/
def num_crocodiles : ℕ := 10

/-- Represents the number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- Represents the number of eyes each animal (frog or crocodile) has -/
def eyes_per_animal : ℕ := 2

/-- Represents the total number of animal eyes in the pond -/
def total_eyes : ℕ := 60

/-- Theorem stating that the number of crocodiles is correct given the conditions -/
theorem crocodile_count_correct :
  num_crocodiles * eyes_per_animal + num_frogs * eyes_per_animal = total_eyes :=
sorry

end NUMINAMATH_CALUDE_crocodile_count_correct_l1915_191573


namespace NUMINAMATH_CALUDE_fair_expenses_correct_l1915_191547

/-- Calculates the total amount spent at a fair given the following conditions:
  - Entrance fee for persons under 18: $5
  - Entrance fee for persons 18 and older: 20% more than $5
  - Cost per ride: $0.50
  - One adult (Joe) and two children (6-year-old twin brothers)
  - Each person took 3 rides
-/
def fairExpenses (childEntranceFee adultEntranceFeeIncrease ridePrice : ℚ) 
                 (numChildren numAdults numRidesPerPerson : ℕ) : ℚ :=
  let childrenEntranceFees := childEntranceFee * numChildren
  let adultEntranceFee := childEntranceFee * (1 + adultEntranceFeeIncrease)
  let adultEntranceFees := adultEntranceFee * numAdults
  let totalEntranceFees := childrenEntranceFees + adultEntranceFees
  let totalRideCost := ridePrice * numRidesPerPerson * (numChildren + numAdults)
  totalEntranceFees + totalRideCost

/-- Theorem stating that the total amount spent at the fair under the given conditions is $20.50 -/
theorem fair_expenses_correct : 
  fairExpenses 5 0.2 0.5 2 1 3 = 41/2 := by
  sorry

end NUMINAMATH_CALUDE_fair_expenses_correct_l1915_191547


namespace NUMINAMATH_CALUDE_four_sharp_40_l1915_191529

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem four_sharp_40 : sharp (sharp (sharp (sharp 40))) = 9.536 := by
  sorry

end NUMINAMATH_CALUDE_four_sharp_40_l1915_191529


namespace NUMINAMATH_CALUDE_quadratic_root_interval_l1915_191514

theorem quadratic_root_interval (a b : ℝ) (hb : b > 0) 
  (h_distinct : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0)
  (h_one_in_unit : ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x ∈ Set.Icc (-1) 1) :
  ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x ∈ Set.Ioo (-b) b :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_interval_l1915_191514


namespace NUMINAMATH_CALUDE_boys_present_age_boys_present_age_proof_l1915_191518

theorem boys_present_age : ℕ → Prop :=
  fun x => (x + 4 = 2 * (x - 6)) → x = 16

-- The proof is omitted
theorem boys_present_age_proof : ∃ x : ℕ, boys_present_age x :=
  sorry

end NUMINAMATH_CALUDE_boys_present_age_boys_present_age_proof_l1915_191518


namespace NUMINAMATH_CALUDE_sine_matrix_determinant_zero_l1915_191571

theorem sine_matrix_determinant_zero :
  let A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    match i, j with
    | 0, 0 => Real.sin 3
    | 0, 1 => Real.sin 4
    | 0, 2 => Real.sin 5
    | 1, 0 => Real.sin 6
    | 1, 1 => Real.sin 7
    | 1, 2 => Real.sin 8
    | 2, 0 => Real.sin 9
    | 2, 1 => Real.sin 10
    | 2, 2 => Real.sin 11
  Matrix.det A = 0 := by
  sorry

-- Sine angle addition formula
axiom sine_angle_addition (x y : ℝ) :
  Real.sin (x + y) = Real.sin x * Real.cos y + Real.cos x * Real.sin y

end NUMINAMATH_CALUDE_sine_matrix_determinant_zero_l1915_191571


namespace NUMINAMATH_CALUDE_half_and_neg_third_are_like_terms_l1915_191521

/-- Definition of like terms -/
def are_like_terms (a b : ℚ) : Prop :=
  (∀ x, a.num * x = 0 ↔ b.num * x = 0) ∧ (a ≠ 0 ∨ b ≠ 0)

/-- Theorem: 1/2 and -1/3 are like terms -/
theorem half_and_neg_third_are_like_terms :
  are_like_terms (1/2 : ℚ) (-1/3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_half_and_neg_third_are_like_terms_l1915_191521


namespace NUMINAMATH_CALUDE_andrew_work_hours_l1915_191582

/-- Calculates the total hours worked given the number of days and hours per day -/
def total_hours (days : ℕ) (hours_per_day : ℝ) : ℝ :=
  days * hours_per_day

/-- Proves that Andrew worked for 7.5 hours given the conditions -/
theorem andrew_work_hours :
  let days : ℕ := 3
  let hours_per_day : ℝ := 2.5
  total_hours days hours_per_day = 7.5 := by
sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l1915_191582


namespace NUMINAMATH_CALUDE_jessica_watermelons_l1915_191588

/-- Given that Jessica grew some watermelons and 30 carrots,
    rabbits ate 27 watermelons, and Jessica has 8 watermelons left,
    prove that Jessica originally grew 35 watermelons. -/
theorem jessica_watermelons :
  ∀ (original_watermelons : ℕ) (carrots : ℕ),
    carrots = 30 →
    original_watermelons - 27 = 8 →
    original_watermelons = 35 := by
  sorry

end NUMINAMATH_CALUDE_jessica_watermelons_l1915_191588


namespace NUMINAMATH_CALUDE_anthony_pencils_l1915_191575

theorem anthony_pencils (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 245 → received = 758 → total = initial + received → total = 1003 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_l1915_191575


namespace NUMINAMATH_CALUDE_final_distance_after_two_hours_l1915_191517

/-- The distance between Jay and Paul after walking for a given time -/
def distance_after_time (initial_distance : ℝ) (jay_speed : ℝ) (paul_speed : ℝ) (time : ℝ) : ℝ :=
  initial_distance + jay_speed * time + paul_speed * time

/-- Theorem stating the final distance between Jay and Paul after 2 hours -/
theorem final_distance_after_two_hours :
  let initial_distance : ℝ := 3
  let jay_speed : ℝ := 1 / (20 / 60) -- miles per hour
  let paul_speed : ℝ := 3 / (40 / 60) -- miles per hour
  let time : ℝ := 2 -- hours
  distance_after_time initial_distance jay_speed paul_speed time = 18 := by
  sorry


end NUMINAMATH_CALUDE_final_distance_after_two_hours_l1915_191517


namespace NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l1915_191557

theorem min_value_product (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

theorem min_value_product_achieved (x : ℝ) : 
  ∃ y : ℝ, (15 - y) * (13 - y) * (15 + y) * (13 + y) = -784 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l1915_191557


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l1915_191504

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l1915_191504


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l1915_191515

def num_candidates : ℕ := 4

theorem president_vice_president_selection :
  (num_candidates * (num_candidates - 1) = 12) := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l1915_191515


namespace NUMINAMATH_CALUDE_linear_function_implies_m_equals_negative_one_l1915_191511

theorem linear_function_implies_m_equals_negative_one (m : ℝ) :
  (∃ a b : ℝ, ∀ x y : ℝ, y = (m^2 - m) * x / (m^2 + 1) ↔ y = a * x + b) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_implies_m_equals_negative_one_l1915_191511


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l1915_191539

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 0; 5, -3]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![8, -2; 1, 1]
  A * B = !![16, -4; 37, -13] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l1915_191539


namespace NUMINAMATH_CALUDE_equation_solution_l1915_191545

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1915_191545


namespace NUMINAMATH_CALUDE_bridget_bakery_profit_l1915_191510

/-- Bridget's bakery problem -/
theorem bridget_bakery_profit :
  let total_loaves : ℕ := 60
  let morning_price : ℚ := 3
  let afternoon_discount : ℚ := 1
  let late_afternoon_price : ℚ := 3/2
  let production_cost : ℚ := 4/5
  let morning_sales : ℕ := total_loaves / 2
  let afternoon_sales : ℕ := ((total_loaves - morning_sales) * 3 + 2) / 4 -- Rounding up
  let late_afternoon_sales : ℕ := total_loaves - morning_sales - afternoon_sales
  let total_revenue : ℚ := 
    morning_sales * morning_price + 
    afternoon_sales * (morning_price - afternoon_discount) + 
    late_afternoon_sales * late_afternoon_price
  let total_cost : ℚ := total_loaves * production_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 197/2 := by sorry

end NUMINAMATH_CALUDE_bridget_bakery_profit_l1915_191510


namespace NUMINAMATH_CALUDE_inequality_proof_l1915_191590

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x*y + y*z + z*x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1915_191590


namespace NUMINAMATH_CALUDE_rachel_hourly_wage_l1915_191534

/-- Rachel's earnings as a waitress in a coffee shop -/
def rachel_earnings (people_served : ℕ) (tip_per_person : ℚ) (total_earnings : ℚ) : Prop :=
  let total_tips : ℚ := people_served * tip_per_person
  let hourly_wage_without_tips : ℚ := total_earnings - total_tips
  hourly_wage_without_tips = 12

theorem rachel_hourly_wage :
  rachel_earnings 20 (25/20) 37 := by
  sorry

end NUMINAMATH_CALUDE_rachel_hourly_wage_l1915_191534


namespace NUMINAMATH_CALUDE_double_root_equation_example_double_root_equation_condition_double_root_equation_m_value_l1915_191516

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) ∧ (y = 2*x ∨ x = 2*y)

/-- Theorem 1: x^2 - 3x + 2 = 0 is a double root equation -/
theorem double_root_equation_example : is_double_root_equation 1 (-3) 2 := sorry

/-- Theorem 2: For (x-2)(x-m) = 0 to be a double root equation, m^2 + 2m + 2 = 26 or 5 -/
theorem double_root_equation_condition (m : ℝ) :
  is_double_root_equation 1 (-(2+m)) (2*m) →
  m^2 + 2*m + 2 = 26 ∨ m^2 + 2*m + 2 = 5 := sorry

/-- Theorem 3: For x^2 - (m-1)x + 32 = 0 to be a double root equation, m = 13 or -11 -/
theorem double_root_equation_m_value (m : ℝ) :
  is_double_root_equation 1 (-(m-1)) 32 →
  m = 13 ∨ m = -11 := sorry

end NUMINAMATH_CALUDE_double_root_equation_example_double_root_equation_condition_double_root_equation_m_value_l1915_191516


namespace NUMINAMATH_CALUDE_probability_two_non_red_marbles_l1915_191589

/-- Given a bag of marbles, calculate the probability of drawing two non-red marbles in succession with replacement after the first draw. -/
theorem probability_two_non_red_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (h1 : total_marbles = 84) 
  (h2 : red_marbles = 12) :
  (total_marbles - red_marbles : ℚ) / total_marbles * 
  ((total_marbles - red_marbles : ℚ) / total_marbles) = 36/49 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_red_marbles_l1915_191589


namespace NUMINAMATH_CALUDE_triangle_side_length_l1915_191505

/-- In a triangle ABC, if a = 1, c = 2, and B = 60°, then b = √3 -/
theorem triangle_side_length (a c b : ℝ) (B : ℝ) : 
  a = 1 → c = 2 → B = π / 3 → b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) → b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1915_191505


namespace NUMINAMATH_CALUDE_banana_apple_sales_l1915_191542

/-- Proves that if the revenue from selling apples and bananas with reversed prices
    is $1 more than the revenue with original prices, then the number of bananas
    sold is 10 more than the number of apples sold. -/
theorem banana_apple_sales
  (apple_price : ℚ)
  (banana_price : ℚ)
  (apple_count : ℕ)
  (banana_count : ℕ)
  (h1 : apple_price = 0.5)
  (h2 : banana_price = 0.4)
  (h3 : banana_price * apple_count + apple_price * banana_count =
        apple_price * apple_count + banana_price * banana_count + 1) :
  banana_count = apple_count + 10 := by
sorry

end NUMINAMATH_CALUDE_banana_apple_sales_l1915_191542


namespace NUMINAMATH_CALUDE_buckets_taken_away_is_three_l1915_191550

/-- Calculates the number of buckets taken away to reach the bath level -/
def buckets_taken_away (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (weekly_usage : ℕ) (baths_per_week : ℕ) : ℕ :=
  let full_tub := bucket_capacity * buckets_to_fill
  let bath_level := weekly_usage / baths_per_week
  let difference := full_tub - bath_level
  difference / bucket_capacity

/-- Proves that the number of buckets taken away is 3 given the problem conditions -/
theorem buckets_taken_away_is_three :
  buckets_taken_away 120 14 9240 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_buckets_taken_away_is_three_l1915_191550


namespace NUMINAMATH_CALUDE_joint_completion_time_l1915_191574

/-- Given two people A and B who can complete a task in x and y hours respectively,
    the time it takes for them to complete the task together is xy/(x+y) hours. -/
theorem joint_completion_time (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x⁻¹ + y⁻¹)⁻¹ = x * y / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_joint_completion_time_l1915_191574


namespace NUMINAMATH_CALUDE_exists_triangle_altitudes_form_triangle_but_not_bisectors_l1915_191530

/-- A triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The altitude triangle formed by the altitudes of the original triangle. -/
def AltitudeTriangle (t : Triangle) : Triangle := sorry

/-- The angle bisectors of a triangle. -/
def AngleBisectors (t : Triangle) : Fin 3 → ℝ := sorry

/-- Predicate to check if three lengths can form a triangle. -/
def CanFormTriangle (l₁ l₂ l₃ : ℝ) : Prop :=
  l₁ + l₂ > l₃ ∧ l₂ + l₃ > l₁ ∧ l₃ + l₁ > l₂

theorem exists_triangle_altitudes_form_triangle_but_not_bisectors :
  ∃ t : Triangle,
    CanFormTriangle (AltitudeTriangle t).a (AltitudeTriangle t).b (AltitudeTriangle t).c ∧
    ¬CanFormTriangle (AngleBisectors (AltitudeTriangle t) 0)
                     (AngleBisectors (AltitudeTriangle t) 1)
                     (AngleBisectors (AltitudeTriangle t) 2) :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_altitudes_form_triangle_but_not_bisectors_l1915_191530


namespace NUMINAMATH_CALUDE_greatest_number_of_sets_l1915_191531

theorem greatest_number_of_sets (t_shirts : ℕ) (buttons : ℕ) : 
  t_shirts = 4 → buttons = 20 → 
  (∃ (sets : ℕ), sets > 0 ∧ 
    t_shirts % sets = 0 ∧ 
    buttons % sets = 0 ∧
    ∀ (k : ℕ), k > 0 ∧ t_shirts % k = 0 ∧ buttons % k = 0 → k ≤ sets) →
  Nat.gcd t_shirts buttons = 4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_of_sets_l1915_191531


namespace NUMINAMATH_CALUDE_seed_germination_problem_l1915_191527

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.30 * x + 0.50 * 200) / (x + 200) = 0.35714285714285715 → 
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l1915_191527


namespace NUMINAMATH_CALUDE_number_equation_solution_l1915_191546

theorem number_equation_solution : ∃ x : ℝ, (0.75 * x + 2 = 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1915_191546


namespace NUMINAMATH_CALUDE_hearty_beads_count_l1915_191540

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each red package -/
def beads_per_red_package : ℕ := 40

/-- The number of beads in each blue package is twice the number in each red package -/
def beads_per_blue_package : ℕ := 2 * beads_per_red_package

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * beads_per_blue_package + red_packages * beads_per_red_package

theorem hearty_beads_count : total_beads = 440 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l1915_191540


namespace NUMINAMATH_CALUDE_second_team_cups_l1915_191580

def total_required : ℕ := 280
def first_team : ℕ := 90
def third_team : ℕ := 70

theorem second_team_cups : total_required - first_team - third_team = 120 := by
  sorry

end NUMINAMATH_CALUDE_second_team_cups_l1915_191580


namespace NUMINAMATH_CALUDE_orange_buckets_total_l1915_191567

theorem orange_buckets_total (bucket1 bucket2 bucket3 : ℕ) : 
  bucket1 = 22 →
  bucket2 = bucket1 + 17 →
  bucket3 = bucket2 - 11 →
  bucket1 + bucket2 + bucket3 = 89 := by
sorry

end NUMINAMATH_CALUDE_orange_buckets_total_l1915_191567


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1915_191559

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 - 4*I) = 1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1915_191559


namespace NUMINAMATH_CALUDE_function_properties_l1915_191512

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem function_properties (f g : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_odd : is_odd_function g) 
  (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  (f 1 + g 1 = 1) ∧ (∀ x, f x = x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1915_191512


namespace NUMINAMATH_CALUDE_egg_weight_probability_l1915_191561

theorem egg_weight_probability (p_less_than_30 : ℝ) (p_between_30_and_40 : ℝ) 
  (h1 : p_less_than_30 = 0.3)
  (h2 : p_between_30_and_40 = 0.5) :
  1 - p_less_than_30 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_egg_weight_probability_l1915_191561


namespace NUMINAMATH_CALUDE_unfollows_calculation_correct_l1915_191594

/-- Calculates the number of unfollows for an Instagram influencer over a year -/
def calculate_unfollows (initial_followers : ℕ) (daily_new_followers : ℕ) (final_followers : ℕ) : ℕ :=
  let potential_followers := initial_followers + daily_new_followers * 365
  potential_followers - final_followers

/-- Theorem: The number of unfollows is correct given the problem conditions -/
theorem unfollows_calculation_correct :
  calculate_unfollows 100000 1000 445000 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_unfollows_calculation_correct_l1915_191594


namespace NUMINAMATH_CALUDE_square_fence_perimeter_is_77_and_third_l1915_191568

/-- The outer perimeter of a square fence with given specifications -/
def squareFencePerimeter (totalPosts : ℕ) (postWidth : ℚ) (gapWidth : ℕ) : ℚ :=
  let postsPerSide : ℕ := totalPosts / 4 + 1
  let gapsPerSide : ℕ := postsPerSide - 1
  let sideLength : ℚ := gapsPerSide * gapWidth + postsPerSide * postWidth
  4 * sideLength

/-- Theorem stating the perimeter of the square fence with given specifications -/
theorem square_fence_perimeter_is_77_and_third :
  squareFencePerimeter 16 (1/3) 6 = 77 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_is_77_and_third_l1915_191568


namespace NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l1915_191556

/-- The percentage of copper in the first alloy -/
def first_alloy_copper_percentage : ℝ := 25

/-- The percentage of copper in the second alloy -/
def second_alloy_copper_percentage : ℝ := 50

/-- The weight of the first alloy used -/
def first_alloy_weight : ℝ := 200

/-- The weight of the second alloy used -/
def second_alloy_weight : ℝ := 800

/-- The total weight of the final alloy -/
def total_weight : ℝ := 1000

/-- The percentage of copper in the final alloy -/
def final_alloy_copper_percentage : ℝ := 45

theorem copper_percentage_in_first_alloy :
  (first_alloy_weight * first_alloy_copper_percentage / 100 +
   second_alloy_weight * second_alloy_copper_percentage / 100) / total_weight * 100 =
  final_alloy_copper_percentage :=
by sorry

end NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l1915_191556


namespace NUMINAMATH_CALUDE_white_balls_count_l1915_191583

theorem white_balls_count (total green blue yellow white : ℕ) : 
  total = green + blue + yellow + white →
  4 * green = total →
  8 * blue = total →
  12 * yellow = total →
  blue = 6 →
  white = 26 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1915_191583


namespace NUMINAMATH_CALUDE_car_trip_speed_l1915_191578

theorem car_trip_speed (initial_speed initial_time total_speed total_time : ℝ) 
  (h1 : initial_speed = 45)
  (h2 : initial_time = 4)
  (h3 : total_speed = 65)
  (h4 : total_time = 12) :
  let remaining_time := total_time - initial_time
  let initial_distance := initial_speed * initial_time
  let total_distance := total_speed * total_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 75 := by sorry

end NUMINAMATH_CALUDE_car_trip_speed_l1915_191578


namespace NUMINAMATH_CALUDE_root_implies_ab_leq_one_l1915_191551

theorem root_implies_ab_leq_one (a b : ℝ) : 
  ((a + b + a) * (a + b + b) = 9) → ab ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_ab_leq_one_l1915_191551


namespace NUMINAMATH_CALUDE_count_squares_below_line_l1915_191569

/-- The number of 1x1 squares in the first quadrant with interiors lying entirely below the line 7x + 268y = 1876 -/
def squares_below_line : ℕ := 801

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 7 * x + 268 * y = 1876

theorem count_squares_below_line :
  squares_below_line = 801 :=
sorry

end NUMINAMATH_CALUDE_count_squares_below_line_l1915_191569


namespace NUMINAMATH_CALUDE_exists_k_l1915_191552

/-- A game configuration with two players and blank squares. -/
structure GameConfig where
  s₁ : ℕ  -- Steps for player 1
  s₂ : ℕ  -- Steps for player 2
  board_size : ℕ  -- Total number of squares on the board

/-- Winning probability for a player given a game configuration and number of blank squares. -/
def winning_probability (config : GameConfig) (player : ℕ) (num_blanks : ℕ) : ℝ :=
  sorry

/-- The statement that proves the existence of k satisfying the given conditions. -/
theorem exists_k (config : GameConfig) : ∃ k : ℕ,
  (∀ n < k, winning_probability config 1 n > 1/2) ∧
  (∃ board_config : List ℕ, 
    board_config.length = k ∧ 
    winning_probability config 2 k > 1/2) :=
by
  -- Assume s₁ = 3 and s₂ = 2
  have h1 : config.s₁ = 3 := by sorry
  have h2 : config.s₂ = 2 := by sorry

  -- Prove that k = 3 satisfies the conditions
  use 3
  sorry


end NUMINAMATH_CALUDE_exists_k_l1915_191552


namespace NUMINAMATH_CALUDE_land_conversion_equation_l1915_191502

/-- Represents the land conversion scenario in a village --/
theorem land_conversion_equation (x : ℝ) : 
  (54 - x = (20 / 100) * (108 + x)) ↔ 
  (54 - x = 0.2 * (108 + x) ∧ 
   0 ≤ x ∧ 
   x ≤ 54 ∧
   108 + x > 0) := by
  sorry

end NUMINAMATH_CALUDE_land_conversion_equation_l1915_191502


namespace NUMINAMATH_CALUDE_gold_silver_coin_values_l1915_191563

theorem gold_silver_coin_values :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset ℕ, S.card = n ∧
    ∀ x ∈ S, x > 0 ∧
    ∃ y : ℕ, y > 0 ∧ y < 100 ∧
    (100 + x) * (100 - y) = 10000) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_gold_silver_coin_values_l1915_191563


namespace NUMINAMATH_CALUDE_original_number_is_two_l1915_191526

theorem original_number_is_two :
  ∃ (x : ℕ), 
    (∃ (y : ℕ), 
      (∀ (z : ℕ), z < y → ¬∃ (w : ℕ), x * z = w^3) ∧ 
      (∃ (w : ℕ), x * y = w^3) ∧
      x * y = 4 * x) →
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_two_l1915_191526


namespace NUMINAMATH_CALUDE_tangent_line_equations_l1915_191576

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 2)*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 2)

-- Theorem statement
theorem tangent_line_equations (a : ℝ) 
  (h1 : ∀ x, f' a x = f' a (-x)) -- f' is an even function
  : (∃ x₀ y₀, x₀ ≠ 1 ∧ f a x₀ = y₀ ∧ 
    (y₀ - (-2)) / (x₀ - 1) = f' a x₀ ∧
    (2 * x + y = 0 ∨ 19 * x - 4 * y - 27 = 0)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l1915_191576


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l1915_191591

theorem quadratic_equation_transformation :
  ∀ x : ℝ, (2 * x^2 = -3 * x + 1) ↔ (2 * x^2 + 3 * x - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l1915_191591


namespace NUMINAMATH_CALUDE_unattainable_value_l1915_191581

theorem unattainable_value (x : ℝ) (y : ℝ) (h : x ≠ -4/3) :
  y = (2 - x) / (3 * x + 4) → y ≠ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_value_l1915_191581


namespace NUMINAMATH_CALUDE_simplify_expression_l1915_191572

theorem simplify_expression (x : ℝ) : (2*x)^4 + (3*x)*(x^3) = 19*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1915_191572


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1915_191513

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - (2 + a) * x + 2 * a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) :
  (a < 2 → solution_set a = {x | a < x ∧ x < 2}) ∧
  (a = 2 → solution_set a = ∅) ∧
  (a > 2 → solution_set a = {x | 2 < x ∧ x < a}) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1915_191513


namespace NUMINAMATH_CALUDE_incorrect_operation_l1915_191596

theorem incorrect_operation : 
  (∀ a : ℝ, (-a)^4 = a^4) ∧ 
  (∀ a : ℝ, -a + 3*a = 2*a) ∧ 
  (¬ ∀ a : ℝ, (2*a^2)^3 = 6*a^5) ∧ 
  (∀ a : ℝ, a^6 / a^2 = a^4) := by sorry

end NUMINAMATH_CALUDE_incorrect_operation_l1915_191596


namespace NUMINAMATH_CALUDE_penultimate_digit_of_quotient_l1915_191535

theorem penultimate_digit_of_quotient : ∃ k : ℕ, 
  (4^1994 + 7^1994) / 10 = k * 10 + 1 := by
  sorry

end NUMINAMATH_CALUDE_penultimate_digit_of_quotient_l1915_191535


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1915_191538

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + 7/8)))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1915_191538


namespace NUMINAMATH_CALUDE_milk_price_calculation_l1915_191585

/-- Proves that given the initial volume of milk, volume of water added, and final price of the mixture,
    the original price of milk per litre can be calculated. -/
theorem milk_price_calculation (initial_milk_volume : ℝ) (water_added : ℝ) (final_mixture_price : ℝ) :
  initial_milk_volume = 60 →
  water_added = 15 →
  final_mixture_price = 32 / 3 →
  ∃ (original_milk_price : ℝ), original_milk_price = 800 / 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_calculation_l1915_191585


namespace NUMINAMATH_CALUDE_mark_chocolates_proof_l1915_191598

/-- The number of chocolates Mark started with --/
def initial_chocolates : ℕ := 104

/-- The number of chocolates Mark's sister took --/
def sister_chocolates : ℕ → Prop := λ x => 5 ≤ x ∧ x ≤ 10

theorem mark_chocolates_proof :
  ∃ (sister_took : ℕ),
    sister_chocolates sister_took ∧
    (initial_chocolates / 4 : ℚ) * 3 / 3 * 2 - 40 - sister_took = 4 ∧
    initial_chocolates % 4 = 0 ∧
    initial_chocolates % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_mark_chocolates_proof_l1915_191598


namespace NUMINAMATH_CALUDE_max_distance_for_A_l1915_191565

/-- Represents a member of the expedition team -/
structure Member where
  name : String
  supplies : Nat

/-- Represents the expedition team -/
structure Team where
  members : List Member
  daily_distance : Nat

/-- Calculates the maximum distance a member can travel -/
def max_distance (team : Team) : Nat :=
  sorry

/-- Main theorem: The maximum distance A can travel is 900 kilometers -/
theorem max_distance_for_A (team : Team) :
  team.members.length = 3 ∧
  team.members.all (λ m => m.supplies = 36) ∧
  team.daily_distance = 30 →
  max_distance team = 900 :=
sorry

end NUMINAMATH_CALUDE_max_distance_for_A_l1915_191565


namespace NUMINAMATH_CALUDE_range_of_b_l1915_191554

/-- A region in the xy-plane defined by y ≤ 3x + b -/
def region (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ 3 * p.1 + b}

/-- The theorem stating the range of b given the conditions -/
theorem range_of_b :
  ∀ b : ℝ,
  (¬ ((3, 4) ∈ region b) ∧ ((4, 4) ∈ region b)) ↔
  (-8 ≤ b ∧ b < -5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1915_191554


namespace NUMINAMATH_CALUDE_smallest_sum_of_leftmost_three_digits_l1915_191586

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def contains_zero (n : ℕ) : Prop := ∃ (a c : ℕ), n = 100 * a + c ∧ a < 10 ∧ c < 100

def all_digits_different (x y : ℕ) : Prop :=
  ∀ (d : ℕ), d < 10 → (
    (∃ (i : ℕ), i < 3 ∧ (x / 10^i) % 10 = d) ↔
    ¬(∃ (j : ℕ), j < 3 ∧ (y / 10^j) % 10 = d)
  )

def sum_of_leftmost_three_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10)

theorem smallest_sum_of_leftmost_three_digits
  (x y : ℕ)
  (hx : is_three_digit x)
  (hy : is_three_digit y)
  (hx0 : contains_zero x)
  (hdiff : all_digits_different x y)
  (hsum : 1000 ≤ x + y ∧ x + y ≤ 9999) :
  ∀ (z : ℕ), is_three_digit z → contains_zero z → all_digits_different z (x + y - z) →
    sum_of_leftmost_three_digits (x + y) ≤ sum_of_leftmost_three_digits (z + (x + y - z)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_leftmost_three_digits_l1915_191586


namespace NUMINAMATH_CALUDE_potato_ratio_l1915_191520

theorem potato_ratio (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) 
  (h1 : total_potatoes = 24)
  (h2 : num_people = 3)
  (h3 : potatoes_per_person = 8)
  (h4 : total_potatoes = num_people * potatoes_per_person) :
  ∃ (r : ℕ), r > 0 ∧ 
    (potatoes_per_person, potatoes_per_person, potatoes_per_person) = (r, r, r) := by
  sorry

end NUMINAMATH_CALUDE_potato_ratio_l1915_191520


namespace NUMINAMATH_CALUDE_count_common_divisors_84_90_l1915_191501

def common_divisors (a b : ℕ) : Finset ℕ :=
  (Finset.range (min a b + 1)).filter (fun d => d > 1 ∧ a % d = 0 ∧ b % d = 0)

theorem count_common_divisors_84_90 :
  (common_divisors 84 90).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_84_90_l1915_191501


namespace NUMINAMATH_CALUDE_bus_car_ratio_l1915_191562

theorem bus_car_ratio (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 85 →
  num_buses = num_cars - 80 →
  (num_buses : ℚ) / (num_cars : ℚ) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_bus_car_ratio_l1915_191562
