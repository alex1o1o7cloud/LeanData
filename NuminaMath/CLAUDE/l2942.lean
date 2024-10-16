import Mathlib

namespace NUMINAMATH_CALUDE_f_equals_g_l2942_294219

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l2942_294219


namespace NUMINAMATH_CALUDE_class_average_score_l2942_294214

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percentage : ℚ) (makeup_day_percentage : ℚ)
  (assigned_day_average : ℚ) (makeup_day_average : ℚ) :
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_day_percentage = 30 / 100 →
  assigned_day_average = 65 / 100 →
  makeup_day_average = 95 / 100 →
  (assigned_day_percentage * assigned_day_average + 
   makeup_day_percentage * makeup_day_average) = 74 / 100 := by
sorry

end NUMINAMATH_CALUDE_class_average_score_l2942_294214


namespace NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_remaining_fuel_formula_l2942_294256

/-- Represents the fuel consumption model of a car -/
structure CarFuelModel where
  initial_fuel : ℝ
  consumption_rate : ℝ

/-- Calculates the remaining fuel after a given time -/
def remaining_fuel (model : CarFuelModel) (t : ℝ) : ℝ :=
  model.initial_fuel - model.consumption_rate * t

/-- Theorem stating the remaining fuel after 3 hours for a specific car model -/
theorem remaining_fuel_after_three_hours
  (model : CarFuelModel)
  (h1 : model.initial_fuel = 100)
  (h2 : model.consumption_rate = 6) :
  remaining_fuel model 3 = 82 := by
  sorry

/-- Theorem proving the general formula for remaining fuel -/
theorem remaining_fuel_formula
  (model : CarFuelModel)
  (h1 : model.initial_fuel = 100)
  (h2 : model.consumption_rate = 6)
  (t : ℝ) :
  remaining_fuel model t = 100 - 6 * t := by
  sorry

end NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_remaining_fuel_formula_l2942_294256


namespace NUMINAMATH_CALUDE_fish_pond_estimate_l2942_294211

theorem fish_pond_estimate (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_marked = 40 →
  second_catch = 100 →
  marked_in_second = 5 →
  (second_catch : ℚ) / marked_in_second = (800 : ℚ) / initial_marked :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_estimate_l2942_294211


namespace NUMINAMATH_CALUDE_largest_sum_is_994_l2942_294291

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of the given configuration -/
def sum (x y : Digit) : ℕ := 113 * x.val + 10 * y.val

/-- The largest possible 3-digit sum for the given configuration -/
def largest_sum : ℕ := 994

theorem largest_sum_is_994 (x y z : Digit) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  sum x y ≤ largest_sum ∧
  ∃ (a b : Digit), sum a b = largest_sum ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_largest_sum_is_994_l2942_294291


namespace NUMINAMATH_CALUDE_lindsey_squat_weight_l2942_294270

/-- The weight Lindsey will squat given exercise bands and a dumbbell -/
theorem lindsey_squat_weight 
  (num_bands : ℕ) 
  (resistance_per_band : ℕ) 
  (dumbbell_weight : ℕ) 
  (h1 : num_bands = 2)
  (h2 : resistance_per_band = 5)
  (h3 : dumbbell_weight = 10) :
  num_bands * resistance_per_band + dumbbell_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_squat_weight_l2942_294270


namespace NUMINAMATH_CALUDE_entrance_exam_score_l2942_294206

theorem entrance_exam_score (total_questions : ℕ) 
  (correct_score incorrect_score unattempted_score : ℤ) 
  (total_score : ℤ) :
  total_questions = 70 ∧ 
  correct_score = 3 ∧ 
  incorrect_score = -1 ∧ 
  unattempted_score = -2 ∧
  total_score = 38 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct = 27 ∧
    incorrect = 43 := by
  sorry

end NUMINAMATH_CALUDE_entrance_exam_score_l2942_294206


namespace NUMINAMATH_CALUDE_complex_product_real_l2942_294232

theorem complex_product_real (m : ℝ) :
  (Complex.I + 1) * (Complex.I * m + 1) ∈ Set.range Complex.ofReal → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l2942_294232


namespace NUMINAMATH_CALUDE_almonds_problem_l2942_294295

theorem almonds_problem (lily_almonds jack_almonds : ℕ) : 
  lily_almonds = jack_almonds + 8 →
  jack_almonds = lily_almonds / 3 →
  lily_almonds = 12 := by
sorry

end NUMINAMATH_CALUDE_almonds_problem_l2942_294295


namespace NUMINAMATH_CALUDE_boxes_with_neither_l2942_294243

/-- Given a set of boxes with markers and stickers, calculate the number of boxes
    containing neither markers nor stickers. -/
theorem boxes_with_neither (total : ℕ) (markers : ℕ) (stickers : ℕ) (both : ℕ)
    (h_total : total = 15)
    (h_markers : markers = 9)
    (h_stickers : stickers = 5)
    (h_both : both = 4) :
    total - (markers + stickers - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l2942_294243


namespace NUMINAMATH_CALUDE_min_people_like_both_l2942_294209

/-- Represents the number of people who like both Vivaldi and Chopin -/
def both_like (v c b : ℕ) : Prop := b = v + c - 150

/-- The minimum number of people who like both Vivaldi and Chopin -/
def min_both_like (v c : ℕ) : ℕ := max 0 (v + c - 150)

theorem min_people_like_both (total v c : ℕ) 
  (h_total : total = 150) 
  (h_v : v = 120) 
  (h_c : c = 90) : 
  min_both_like v c = 60 := by
  sorry

#eval min_both_like 120 90

end NUMINAMATH_CALUDE_min_people_like_both_l2942_294209


namespace NUMINAMATH_CALUDE_builder_nuts_boxes_l2942_294240

/-- Represents the number of boxes of nuts purchased by the builder. -/
def boxes_of_nuts : ℕ := sorry

/-- Represents the number of boxes of bolts purchased by the builder. -/
def boxes_of_bolts : ℕ := 7

/-- Represents the number of bolts in each box. -/
def bolts_per_box : ℕ := 11

/-- Represents the number of nuts in each box. -/
def nuts_per_box : ℕ := 15

/-- Represents the number of bolts left over after the project. -/
def bolts_leftover : ℕ := 3

/-- Represents the number of nuts left over after the project. -/
def nuts_leftover : ℕ := 6

/-- Represents the total number of bolts and nuts used in the project. -/
def total_used : ℕ := 113

theorem builder_nuts_boxes : 
  boxes_of_nuts = 3 ∧
  boxes_of_bolts * bolts_per_box - bolts_leftover + 
  boxes_of_nuts * nuts_per_box - nuts_leftover = total_used :=
sorry

end NUMINAMATH_CALUDE_builder_nuts_boxes_l2942_294240


namespace NUMINAMATH_CALUDE_min_box_value_l2942_294210

/-- Given that (ax+b)(bx+a) = 30x^2 + ⬜x + 30, where a, b, and ⬜ are distinct integers,
    prove that the minimum possible value of ⬜ is 61. -/
theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a*x + b)*(b*x + a) = 30*x^2 + box*x + 30) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  a * b = 30 →
  box = a^2 + b^2 →
  (∀ a' b' box' : ℤ, 
    (∀ x, (a'*x + b')*(b'*x + a') = 30*x^2 + box'*x + 30) →
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' →
    a' * b' = 30 →
    box' = a'^2 + b'^2 →
    box ≤ box') →
  box = 61 := by
sorry

end NUMINAMATH_CALUDE_min_box_value_l2942_294210


namespace NUMINAMATH_CALUDE_num_small_squares_seven_l2942_294220

/-- The number of small squares formed when a square is divided into n equal parts on each side and the points are joined -/
def num_small_squares (n : ℕ) : ℕ := 4 * (n * (n - 1) / 2)

/-- Theorem stating that the number of small squares is 84 when n = 7 -/
theorem num_small_squares_seven : num_small_squares 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_num_small_squares_seven_l2942_294220


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2942_294255

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 509) :
  ∃ (k : ℕ), k = 14 ∧
  (∀ m : ℕ, m < k → ¬((n - m) % 9 = 0 ∧ (n - m) % 15 = 0)) ∧
  (n - k) % 9 = 0 ∧ (n - k) % 15 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2942_294255


namespace NUMINAMATH_CALUDE_initial_percent_problem_l2942_294267

theorem initial_percent_problem (x : ℝ) :
  (3 : ℝ) / 100 = (60 : ℝ) / 100 * x → x = (5 : ℝ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_percent_problem_l2942_294267


namespace NUMINAMATH_CALUDE_total_earned_is_144_l2942_294221

/-- Calculates the total money earned from selling milk and butter --/
def total_money_earned (milk_price : ℚ) (butter_conversion : ℚ) (butter_price : ℚ) 
  (num_cows : ℕ) (milk_per_cow : ℚ) (num_customers : ℕ) (milk_per_customer : ℚ) : ℚ :=
  let total_milk := num_cows * milk_per_cow
  let sold_milk := min total_milk (num_customers * milk_per_customer)
  let remaining_milk := total_milk - sold_milk
  let butter_sticks := remaining_milk * butter_conversion
  milk_price * sold_milk + butter_price * butter_sticks

/-- Theorem stating that the total money earned is $144 given the problem conditions --/
theorem total_earned_is_144 :
  total_money_earned 3 2 (3/2) 12 4 6 6 = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_earned_is_144_l2942_294221


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_not_l2942_294254

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |a*x + 1|

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem negation_of_existence_is_universal_not :
  (¬ ∃ a : ℝ, is_even_function (f a)) ↔ (∀ a : ℝ, ¬ is_even_function (f a)) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_not_l2942_294254


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l2942_294239

/-- Proves that Maxwell walks for 10 hours before meeting Brad -/
theorem maxwell_brad_meeting_time :
  ∀ (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (head_start : ℝ),
    distance = 94 →
    maxwell_speed = 4 →
    brad_speed = 6 →
    head_start = 1 →
    ∃ (t : ℝ),
      t > 0 ∧
      maxwell_speed * (t + head_start) + brad_speed * t = distance ∧
      t + head_start = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l2942_294239


namespace NUMINAMATH_CALUDE_only_C_is_lying_l2942_294265

-- Define the possible scores
def possible_scores : List ℕ := [1, 3, 5, 7, 9]

-- Define a structure for a person's statement
structure Statement where
  shots : ℕ
  hits : ℕ
  total_score : ℕ

-- Define the statements of A, B, C, and D
def statement_A : Statement := ⟨5, 5, 35⟩
def statement_B : Statement := ⟨6, 6, 36⟩
def statement_C : Statement := ⟨3, 3, 24⟩
def statement_D : Statement := ⟨4, 3, 21⟩

-- Define a function to check if a statement is valid
def is_valid_statement (s : Statement) (scores : List ℕ) : Prop :=
  ∃ (score_list : List ℕ),
    score_list.length = s.hits ∧
    score_list.sum = s.total_score ∧
    ∀ x ∈ score_list, x ∈ scores

-- Theorem stating that C's statement is false while others are true
theorem only_C_is_lying :
  is_valid_statement statement_A possible_scores ∧
  is_valid_statement statement_B possible_scores ∧
  ¬is_valid_statement statement_C possible_scores ∧
  is_valid_statement statement_D possible_scores :=
sorry

end NUMINAMATH_CALUDE_only_C_is_lying_l2942_294265


namespace NUMINAMATH_CALUDE_james_annual_training_hours_l2942_294230

/-- Represents James' training schedule and calculates his total training hours in a year --/
def jamesTrainingHours : ℕ :=
  let weeklyHours : ℕ := 3 * 2 * 4 + 2 * (3 + 5)  -- Weekly training hours
  let totalWeeks : ℕ := 52  -- Weeks in a year
  let holidayWeeks : ℕ := 1  -- Week off for holidays
  let missedDays : ℕ := 10  -- Additional missed days
  let trainingDaysPerWeek : ℕ := 5  -- Number of training days per week
  let effectiveTrainingWeeks : ℕ := totalWeeks - holidayWeeks - (missedDays / trainingDaysPerWeek)
  weeklyHours * effectiveTrainingWeeks

/-- Theorem stating that James trains for 1960 hours in a year --/
theorem james_annual_training_hours :
  jamesTrainingHours = 1960 := by
  sorry

end NUMINAMATH_CALUDE_james_annual_training_hours_l2942_294230


namespace NUMINAMATH_CALUDE_difference_of_squares_l2942_294227

theorem difference_of_squares (x y : ℝ) : (y + x) * (y - x) = y^2 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2942_294227


namespace NUMINAMATH_CALUDE_distance_from_Q_to_l_l2942_294263

/-- Given points A and B, a line l, and a point Q on the x-axis equidistant from A and B,
    prove that the distance from Q to l is 18/5. -/
theorem distance_from_Q_to_l (A B Q : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : 
  A = (4, -3) →
  B = (2, -1) →
  (∀ x y, l (x, y) = 4*x + 3*y - 2) →
  Q.2 = 0 →
  (Q.1 - A.1)^2 + (Q.2 - A.2)^2 = (Q.1 - B.1)^2 + (Q.2 - B.2)^2 →
  (|4 * Q.1 + 3 * Q.2 - 2| / Real.sqrt (4^2 + 3^2) : ℝ) = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_Q_to_l_l2942_294263


namespace NUMINAMATH_CALUDE_tennis_cost_calculation_l2942_294282

/-- Represents the cost of tennis equipment under different purchasing options -/
def TennisCost (x : ℕ) : Prop :=
  let racketPrice : ℕ := 200
  let ballPrice : ℕ := 40
  let racketQuantity : ℕ := 20
  let option1Cost : ℕ := racketPrice * racketQuantity + ballPrice * (x - racketQuantity)
  let option2Cost : ℕ := (racketPrice * racketQuantity + ballPrice * x) * 9 / 10
  x > 20 ∧ option1Cost = 40 * x + 3200 ∧ option2Cost = 3600 + 36 * x

theorem tennis_cost_calculation (x : ℕ) : TennisCost x := by
  sorry

end NUMINAMATH_CALUDE_tennis_cost_calculation_l2942_294282


namespace NUMINAMATH_CALUDE_min_moves_to_equalize_l2942_294287

/-- Represents the state of coin stacks -/
structure CoinStacks :=
  (stack1 : ℕ)
  (stack2 : ℕ)
  (stack3 : ℕ)
  (stack4 : ℕ)

/-- Represents a move in the coin stacking game -/
def move (s : CoinStacks) : CoinStacks := sorry

/-- Checks if all stacks have equal coins -/
def is_equal (s : CoinStacks) : Prop := 
  s.stack1 = s.stack2 ∧ s.stack2 = s.stack3 ∧ s.stack3 = s.stack4

/-- The initial state of coin stacks -/
def initial_state : CoinStacks := ⟨9, 7, 5, 10⟩

/-- Applies n moves to a given state -/
def apply_moves (s : CoinStacks) (n : ℕ) : CoinStacks := sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_to_equalize : 
  ∃ (n : ℕ), n = 11 ∧ is_equal (apply_moves initial_state n) ∧ 
  ∀ (m : ℕ), m < n → ¬is_equal (apply_moves initial_state m) :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_equalize_l2942_294287


namespace NUMINAMATH_CALUDE_remaining_money_l2942_294269

def initial_amount : ℕ := 400
def dress_count : ℕ := 5
def dress_price : ℕ := 20
def pants_count : ℕ := 3
def pants_price : ℕ := 12
def jacket_count : ℕ := 4
def jacket_price : ℕ := 30
def transportation_cost : ℕ := 5

def total_expense : ℕ := 
  dress_count * dress_price + 
  pants_count * pants_price + 
  jacket_count * jacket_price + 
  transportation_cost

theorem remaining_money : 
  initial_amount - total_expense = 139 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2942_294269


namespace NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l2942_294204

/-- Given a point with rectangular coordinates (8, 6) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r³, 3π/2 * θ) has rectangular
    coordinates (-600, -800). -/
theorem polar_to_rectangular_transformation (r θ : ℝ) :
  r * Real.cos θ = 8 ∧ r * Real.sin θ = 6 →
  (r^3 * Real.cos ((3 * Real.pi / 2) * θ) = -600) ∧
  (r^3 * Real.sin ((3 * Real.pi / 2) * θ) = -800) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l2942_294204


namespace NUMINAMATH_CALUDE_euler_identity_complex_power_exp_sum_bound_l2942_294296

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- Theorems to prove
theorem euler_identity : cexp π + 1 = 0 := by sorry

theorem complex_power : (1/2 + Complex.I * (Real.sqrt 3)/2) ^ 2022 = 1 := by sorry

theorem exp_sum_bound (x : ℝ) : Complex.abs (cexp x + cexp (-x)) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_euler_identity_complex_power_exp_sum_bound_l2942_294296


namespace NUMINAMATH_CALUDE_no_cubic_four_primes_pm3_l2942_294274

theorem no_cubic_four_primes_pm3 : 
  ¬∃ (f : ℤ → ℤ) (p q r s : ℕ), 
    (∀ x : ℤ, ∃ a b c d : ℤ, f x = a*x^3 + b*x^2 + c*x + d) ∧ 
    Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    ((f p = 3 ∨ f p = -3) ∧ 
     (f q = 3 ∨ f q = -3) ∧ 
     (f r = 3 ∨ f r = -3) ∧ 
     (f s = 3 ∨ f s = -3)) :=
by sorry

end NUMINAMATH_CALUDE_no_cubic_four_primes_pm3_l2942_294274


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2942_294238

/-- The complex number i such that i² = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The given complex number z -/
noncomputable def z : ℂ := (1 - i) / (1 + i) + 4 - 2*i

/-- Theorem stating that the magnitude of z is 5 -/
theorem magnitude_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2942_294238


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2942_294212

/-- A geometric sequence of positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_positive : ∀ n, a n > 0)
  (h_sum : a 2 * a 8 + a 3 * a 7 = 32) : 
  a 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2942_294212


namespace NUMINAMATH_CALUDE_balloons_left_after_sharing_l2942_294244

def blue_balloons : ℕ := 303
def purple_balloons : ℕ := 453

theorem balloons_left_after_sharing :
  (blue_balloons + purple_balloons) / 2 = 378 := by
  sorry

end NUMINAMATH_CALUDE_balloons_left_after_sharing_l2942_294244


namespace NUMINAMATH_CALUDE_correct_equation_l2942_294217

theorem correct_equation (x y : ℝ) : x * y - 2 * (x * y) = -(x * y) := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2942_294217


namespace NUMINAMATH_CALUDE_mod_pow_98_50_100_l2942_294272

theorem mod_pow_98_50_100 : 98^50 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mod_pow_98_50_100_l2942_294272


namespace NUMINAMATH_CALUDE_aluminum_carbonate_weight_l2942_294292

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The molecular formula of aluminum carbonate -/
structure AluminumCarbonate where
  Al : Fin 2
  CO3 : Fin 3

/-- Calculate the molecular weight of aluminum carbonate -/
def molecular_weight (ac : AluminumCarbonate) : ℝ :=
  2 * Al_weight + 3 * C_weight + 9 * O_weight

/-- Theorem: The molecular weight of aluminum carbonate is 233.99 g/mol -/
theorem aluminum_carbonate_weight :
  ∀ ac : AluminumCarbonate, molecular_weight ac = 233.99 := by
  sorry

end NUMINAMATH_CALUDE_aluminum_carbonate_weight_l2942_294292


namespace NUMINAMATH_CALUDE_total_guests_served_l2942_294271

theorem total_guests_served (adults : ℕ) (children : ℕ) (seniors : ℕ) : 
  adults = 58 →
  children = adults - 35 →
  seniors = 2 * children →
  adults + children + seniors = 127 := by
  sorry

end NUMINAMATH_CALUDE_total_guests_served_l2942_294271


namespace NUMINAMATH_CALUDE_max_popsicles_for_eight_dollars_l2942_294241

/-- Represents the number of popsicles in a box -/
inductive BoxSize
  | Single : BoxSize
  | Three : BoxSize
  | Five : BoxSize

/-- Returns the cost of a box given its size -/
def boxCost (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 2
  | BoxSize.Five => 3

/-- Returns the number of popsicles in a box given its size -/
def boxCount (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 3
  | BoxSize.Five => 5

/-- Represents a purchase of popsicle boxes -/
structure Purchase where
  singles : ℕ
  threes : ℕ
  fives : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * boxCost BoxSize.Single +
  p.threes * boxCost BoxSize.Three +
  p.fives * boxCost BoxSize.Five

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.singles * boxCount BoxSize.Single +
  p.threes * boxCount BoxSize.Three +
  p.fives * boxCount BoxSize.Five

/-- Theorem: The maximum number of popsicles that can be purchased with $8 is 13 -/
theorem max_popsicles_for_eight_dollars :
  (∃ p : Purchase, totalCost p = 8 ∧ totalPopsicles p = 13) ∧
  (∀ p : Purchase, totalCost p ≤ 8 → totalPopsicles p ≤ 13) := by
  sorry

end NUMINAMATH_CALUDE_max_popsicles_for_eight_dollars_l2942_294241


namespace NUMINAMATH_CALUDE_problem_solution_l2942_294224

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2*(a-1) + b^2*(b-1) + c^2*(c-1) = a*(a-1) + b*(b-1) + c*(c-1)) :
  1956*a^2 + 1986*b^2 + 2016*c^2 = 5958 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2942_294224


namespace NUMINAMATH_CALUDE_present_cost_l2942_294226

/-- Proves that the total amount paid for a present by 4 friends is $60, given specific conditions. -/
theorem present_cost (initial_contribution : ℝ) : 
  (4 : ℝ) > 0 → 
  0 < initial_contribution → 
  0.75 * (4 * initial_contribution) = 4 * (initial_contribution - 5) → 
  0.75 * (4 * initial_contribution) = 60 := by
  sorry

end NUMINAMATH_CALUDE_present_cost_l2942_294226


namespace NUMINAMATH_CALUDE_problem_solution_l2942_294289

-- Define the conditions
def conditions (a b t : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ t = a * b

-- Theorem statement
theorem problem_solution (a b t : ℝ) (h : conditions a b t) :
  (0 < a ∧ a < 1) ∧
  (0 < t ∧ t ≤ 1/4) ∧
  ((a + 1/a) * (b + 1/b) ≥ 25/4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2942_294289


namespace NUMINAMATH_CALUDE_carlos_has_largest_answer_l2942_294261

def alice_calculation (x : ℕ) : ℕ := ((x - 3) * 3) + 5

def bob_calculation (x : ℕ) : ℕ := (x^2 - 4) + 5

def carlos_calculation (x : ℕ) : ℕ := (x - 2 + 3)^2

theorem carlos_has_largest_answer :
  let initial_number := 12
  carlos_calculation initial_number > alice_calculation initial_number ∧
  carlos_calculation initial_number > bob_calculation initial_number :=
by sorry

end NUMINAMATH_CALUDE_carlos_has_largest_answer_l2942_294261


namespace NUMINAMATH_CALUDE_min_electricity_price_l2942_294294

/-- The minimum electricity price that ensures at least a 20% revenue increase -/
theorem min_electricity_price (a : ℝ) (h : a > 0) : 
  let f := fun (x : ℝ) ↦ (a + 0.2 * a / (x - 0.4)) * (x - 0.3) - 1.2 * a * 0.5
  ∃ (x_min : ℝ), x_min = 0.6 ∧ 
    (∀ x, 0.55 ≤ x ∧ x ≤ 0.75 ∧ f x ≥ 0 → x_min ≤ x) ∧
    (0.55 ≤ x_min ∧ x_min ≤ 0.75 ∧ f x_min ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_electricity_price_l2942_294294


namespace NUMINAMATH_CALUDE_function_properties_l2942_294237

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- State the theorem
theorem function_properties (a b : ℝ) :
  a ≠ 0 →
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (a = -1 ∧ b = 4) ∧
  (f a b 1 = 2 → a > 0 → b > 0 → 
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 4/b' ≥ 9) ∧
    (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 4/b' = 9)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2942_294237


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2942_294285

-- Problem 1
theorem problem_1 : |(-2023 : ℤ)| + π^(0 : ℝ) - (1/6)⁻¹ + Real.sqrt 16 = 2022 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (hm : m ≠ 1) :
  (1 + 1/m) / ((m^2 - 1) / m) = 1 / (m - 1) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2942_294285


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2942_294201

theorem chocolate_box_problem (day1 day2 day3 day4 remaining : ℕ) :
  day1 = 4 →
  day2 = 2 * day1 - 3 →
  day3 = day1 - 2 →
  day4 = day3 - 1 →
  remaining = 12 →
  day1 + day2 + day3 + day4 + remaining = 24 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2942_294201


namespace NUMINAMATH_CALUDE_max_vertical_distance_is_sqrt2_over_2_l2942_294242

/-- Represents a square with side length 1 inch -/
structure UnitSquare where
  center : ℝ × ℝ

/-- Represents the configuration of four squares -/
structure SquareConfiguration where
  squares : List UnitSquare
  rotated_square : UnitSquare

/-- The maximum vertical distance from the original line to any point on the rotated square -/
def max_vertical_distance (config : SquareConfiguration) : ℝ :=
  sorry

/-- Theorem stating the maximum vertical distance is √2/2 -/
theorem max_vertical_distance_is_sqrt2_over_2 (config : SquareConfiguration) :
  max_vertical_distance config = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_vertical_distance_is_sqrt2_over_2_l2942_294242


namespace NUMINAMATH_CALUDE_distance_AB_bounds_l2942_294281

/-- Given six points in space with specific distance relationships, 
    prove that the distance between two of the points lies within a certain range. -/
theorem distance_AB_bounds 
  (A B C D E F : EuclideanSpace ℝ (Fin 3)) 
  (h1 : dist A C = 10 ∧ dist A D = 10 ∧ dist B E = 10 ∧ dist B F = 10)
  (h2 : dist A E = 12 ∧ dist A F = 12 ∧ dist B C = 12 ∧ dist B D = 12)
  (h3 : dist C D = 11 ∧ dist E F = 11)
  (h4 : dist C E = 5 ∧ dist D F = 5) : 
  8.8 < dist A B ∧ dist A B < 19.2 := by
  sorry


end NUMINAMATH_CALUDE_distance_AB_bounds_l2942_294281


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2942_294279

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2942_294279


namespace NUMINAMATH_CALUDE_min_value_fraction_l2942_294297

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → 1/a + a/(8*b) ≤ 1/x + x/(8*y)) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ 1/x + x/(8*y) = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2942_294297


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l2942_294273

-- Define a quadratic polynomial with integer coefficients
def QuadraticPolynomial (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

theorem quadratic_polynomial_property (a b c : ℤ) :
  let f := QuadraticPolynomial a b c
  (f (Real.sqrt 3) - f (Real.sqrt 2) = 4) →
  (f (Real.sqrt 10) - f (Real.sqrt 7) = 12) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l2942_294273


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2942_294257

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^10 + 3*x^8 - 5*x^6 + 15*x^4 + 25) ≤ 1/17 ∧
  ∃ y : ℝ, y^6 / (y^10 + 3*y^8 - 5*y^6 + 15*y^4 + 25) = 1/17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2942_294257


namespace NUMINAMATH_CALUDE_apples_in_refrigerator_l2942_294298

def initial_apples : ℕ := 62
def pie_apples : ℕ := initial_apples / 2
def muffin_apples : ℕ := 6

def refrigerator_apples : ℕ := initial_apples - pie_apples - muffin_apples

theorem apples_in_refrigerator : refrigerator_apples = 25 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_refrigerator_l2942_294298


namespace NUMINAMATH_CALUDE_min_value_x_over_y_l2942_294290

theorem min_value_x_over_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) :
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 1/x' + y' = 2 → x/y ≤ x'/y' ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + y₀ = 2 ∧ x₀/y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_over_y_l2942_294290


namespace NUMINAMATH_CALUDE_square_inscribed_in_circle_l2942_294216

theorem square_inscribed_in_circle (D : ℝ) (A : ℝ) :
  D = 10 →
  A = (D / (2 * Real.sqrt 2))^2 →
  A = 50 := by sorry

end NUMINAMATH_CALUDE_square_inscribed_in_circle_l2942_294216


namespace NUMINAMATH_CALUDE_fraction_of_students_with_As_l2942_294286

theorem fraction_of_students_with_As (fraction_B : ℝ) (fraction_A_or_B : ℝ) 
  (h1 : fraction_B = 0.2) 
  (h2 : fraction_A_or_B = 0.9) : 
  ∃ fraction_A : ℝ, fraction_A + fraction_B = fraction_A_or_B ∧ fraction_A = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_students_with_As_l2942_294286


namespace NUMINAMATH_CALUDE_no_harmonic_point_on_reciprocal_unique_harmonic_point_range_of_m_l2942_294236

-- Definition of a harmonic point
def is_harmonic_point (x y : ℝ) : Prop := x = y

-- Part 1: No harmonic point on y = -4/x
theorem no_harmonic_point_on_reciprocal : ¬∃ x : ℝ, is_harmonic_point x (-4/x) := by sorry

-- Part 2: Quadratic function with one harmonic point
def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + 6 * x + c

theorem unique_harmonic_point :
  ∃! (a c : ℝ), a ≠ 0 ∧ 
  (∃! x : ℝ, is_harmonic_point x (quadratic_function a c x)) ∧
  is_harmonic_point (5/2) (quadratic_function a c (5/2)) := by sorry

-- Part 3: Range of m for the modified quadratic function
def modified_quadratic (x : ℝ) : ℝ := -x^2 + 6*x - 6

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, 1 ≤ x → x ≤ m → -1 ≤ modified_quadratic x ∧ modified_quadratic x ≤ 3) ↔
  (3 ≤ m ∧ m ≤ 5) := by sorry

end NUMINAMATH_CALUDE_no_harmonic_point_on_reciprocal_unique_harmonic_point_range_of_m_l2942_294236


namespace NUMINAMATH_CALUDE_additional_deductible_calculation_l2942_294208

/-- Calculates the additional deductible amount for an average family --/
def additional_deductible_amount (
  current_deductible : ℝ)
  (plan_a_increase : ℝ)
  (plan_b_increase : ℝ)
  (plan_c_increase : ℝ)
  (plan_a_percentage : ℝ)
  (plan_b_percentage : ℝ)
  (plan_c_percentage : ℝ)
  (inflation_rate : ℝ) : ℝ :=
  let plan_a_additional := current_deductible * plan_a_increase
  let plan_b_additional := current_deductible * plan_b_increase
  let plan_c_additional := current_deductible * plan_c_increase
  let weighted_additional := plan_a_additional * plan_a_percentage +
                             plan_b_additional * plan_b_percentage +
                             plan_c_additional * plan_c_percentage
  weighted_additional * (1 + inflation_rate)

/-- Theorem stating the additional deductible amount for an average family --/
theorem additional_deductible_calculation :
  additional_deductible_amount 3000 (2/3) (1/2) (3/5) 0.4 0.3 0.3 0.03 = 1843.70 := by
  sorry

end NUMINAMATH_CALUDE_additional_deductible_calculation_l2942_294208


namespace NUMINAMATH_CALUDE_max_tan_A_in_triangle_l2942_294277

/-- Given a triangle ABC where sin A + 2sin B cos C = 0, the maximum value of tan A is 1/√3 -/
theorem max_tan_A_in_triangle (A B C : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.sin A + 2 * Real.sin B * Real.cos C = 0) : 
  (∀ A' B' C' : Real, 0 < A' ∧ A' < π → 0 < B' ∧ B' < π → 0 < C' ∧ C' < π → 
   A' + B' + C' = π → Real.sin A' + 2 * Real.sin B' * Real.cos C' = 0 → 
   Real.tan A' ≤ Real.tan A) → Real.tan A = 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_tan_A_in_triangle_l2942_294277


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_diff_eq_sum_of_roots_eq_fourteen_l2942_294231

theorem sum_of_roots_squared_diff_eq (a c : ℝ) : 
  (∀ x : ℝ, (x - a)^2 = c) → (∃ x₁ x₂ : ℝ, (x₁ - a)^2 = c ∧ (x₂ - a)^2 = c ∧ x₁ + x₂ = 2 * a) :=
by sorry

theorem sum_of_roots_eq_fourteen : 
  (∃ x₁ x₂ : ℝ, (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 ∧ x₁ + x₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_diff_eq_sum_of_roots_eq_fourteen_l2942_294231


namespace NUMINAMATH_CALUDE_minibus_boys_count_l2942_294245

theorem minibus_boys_count : 
  ∀ (total boys girls : ℕ),
  total = 18 →
  boys + girls = total →
  boys = girls - 2 →
  boys = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_minibus_boys_count_l2942_294245


namespace NUMINAMATH_CALUDE_alcohol_in_mixture_l2942_294200

/-- Proves that the amount of alcohol in a mixture is 7.5 liters given specific conditions -/
theorem alcohol_in_mixture :
  ∀ (A W : ℝ), 
    (A / W = 4 / 3) →  -- Initial ratio of alcohol to water
    (A / (W + 5) = 4 / 5) →  -- Ratio after adding 5 liters of water
    A = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_in_mixture_l2942_294200


namespace NUMINAMATH_CALUDE_divisor_problem_l2942_294228

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 149 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2942_294228


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2942_294235

/-- A regular polygon with exterior angles measuring 18 degrees -/
structure RegularPolygon where
  -- The number of sides
  sides : ℕ
  -- The measure of each exterior angle in degrees
  exterior_angle : ℝ
  -- The measure of each interior angle in degrees
  interior_angle : ℝ
  -- Condition: The polygon is regular and each exterior angle measures 18 degrees
  h_exterior : exterior_angle = 18
  -- Relationship between number of sides and exterior angle
  h_sides : sides = (360 : ℝ) / exterior_angle
  -- Relationship between interior and exterior angles
  h_interior : interior_angle = 180 - exterior_angle

/-- Theorem about the properties of the specific regular polygon -/
theorem regular_polygon_properties (p : RegularPolygon) : 
  p.sides = 20 ∧ p.interior_angle = 162 := by
  sorry


end NUMINAMATH_CALUDE_regular_polygon_properties_l2942_294235


namespace NUMINAMATH_CALUDE_gunther_free_time_l2942_294246

/-- Represents the time required for cleaning tasks and available free time -/
structure CleaningTime where
  vacuum : ℕ
  dust : ℕ
  mop : ℕ
  brush_per_cat : ℕ
  num_cats : ℕ
  free_time : ℕ

/-- Calculates the remaining free time after cleaning -/
def remaining_free_time (ct : CleaningTime) : ℕ :=
  ct.free_time - (ct.vacuum + ct.dust + ct.mop + ct.brush_per_cat * ct.num_cats)

/-- Theorem: Given Gunther's cleaning tasks and available time, he will have 30 minutes left -/
theorem gunther_free_time :
  ∀ (ct : CleaningTime),
    ct.vacuum = 45 →
    ct.dust = 60 →
    ct.mop = 30 →
    ct.brush_per_cat = 5 →
    ct.num_cats = 3 →
    ct.free_time = 180 →
    remaining_free_time ct = 30 := by
  sorry

end NUMINAMATH_CALUDE_gunther_free_time_l2942_294246


namespace NUMINAMATH_CALUDE_f_local_min_at_neg_one_f_two_extrema_iff_l2942_294229

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x - a * Real.exp x)

-- Theorem 1: When a = 0, f has a local minimum at x = -1
theorem f_local_min_at_neg_one :
  ∃ δ > 0, ∀ x, |x - (-1)| < δ ∧ x ≠ -1 → f 0 x > f 0 (-1) :=
sorry

-- Theorem 2: f has two different extremum points iff 0 < a < 1/2
theorem f_two_extrema_iff (a : ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ 
    (∀ h, 0 < h → f a (x₁ - h) > f a x₁ ∧ f a (x₁ + h) > f a x₁) ∧
    (∀ h, 0 < h → f a (x₂ - h) < f a x₂ ∧ f a (x₂ + h) < f a x₂))
  ↔ 0 < a ∧ a < 1/2 :=
sorry

end

end NUMINAMATH_CALUDE_f_local_min_at_neg_one_f_two_extrema_iff_l2942_294229


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l2942_294284

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ+),
    (Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = 
     (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val) ∧
    (∀ (a' b' c' : ℕ+),
      (Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = 
       (a'.val * Real.sqrt 6 + b'.val * Real.sqrt 8) / c'.val) →
      c'.val ≥ c.val) ∧
    (a.val + b.val + c.val = 19) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l2942_294284


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2942_294215

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2942_294215


namespace NUMINAMATH_CALUDE_keith_receives_144_messages_l2942_294275

/-- Represents the number of messages sent between people in a day -/
structure MessageCount where
  juan_to_laurence : ℕ
  juan_to_keith : ℕ
  laurence_to_missy : ℕ

/-- The conditions of the messaging problem -/
def messaging_problem (m : MessageCount) : Prop :=
  m.juan_to_keith = 8 * m.juan_to_laurence ∧
  m.laurence_to_missy = m.juan_to_laurence ∧
  m.laurence_to_missy = 18

/-- The theorem stating that Keith receives 144 messages from Juan -/
theorem keith_receives_144_messages (m : MessageCount) 
  (h : messaging_problem m) : m.juan_to_keith = 144 := by
  sorry

end NUMINAMATH_CALUDE_keith_receives_144_messages_l2942_294275


namespace NUMINAMATH_CALUDE_quadratic_trinomial_negative_l2942_294203

theorem quadratic_trinomial_negative (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_negative_l2942_294203


namespace NUMINAMATH_CALUDE_simplify_expression_solve_cubic_equation_l2942_294283

-- Problem 1
theorem simplify_expression (a b : ℝ) : 2*a*(a-2*b) - (2*a-b)^2 = -2*a^2 - b^2 := by
  sorry

-- Problem 2
theorem solve_cubic_equation : ∃ x : ℝ, (x-1)^3 - 3 = 3/8 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_cubic_equation_l2942_294283


namespace NUMINAMATH_CALUDE_det_special_matrix_l2942_294268

/-- The determinant of the matrix [[2x + 2, 2x, 2x], [2x, 2x + 2, 2x], [2x, 2x, 2x + 2]] is equal to 20x + 8 -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![2*x + 2, 2*x, 2*x; 
                2*x, 2*x + 2, 2*x; 
                2*x, 2*x, 2*x + 2] = 20*x + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2942_294268


namespace NUMINAMATH_CALUDE_comparison_of_special_angles_l2942_294258

open Real

theorem comparison_of_special_angles (a b c : ℝ) 
  (ha : 0 < a ∧ a < π/2)
  (hb : 0 < b ∧ b < π/2)
  (hc : 0 < c ∧ c < π/2)
  (eq_a : cos a = a)
  (eq_b : sin (cos b) = b)
  (eq_c : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end NUMINAMATH_CALUDE_comparison_of_special_angles_l2942_294258


namespace NUMINAMATH_CALUDE_power_equality_implies_x_equals_two_l2942_294293

theorem power_equality_implies_x_equals_two :
  ∀ x : ℝ, (2 : ℝ)^10 = 32^x → x = 2 := by
sorry

end NUMINAMATH_CALUDE_power_equality_implies_x_equals_two_l2942_294293


namespace NUMINAMATH_CALUDE_vector_sum_norm_equality_implies_parallel_l2942_294276

/-- Given two non-zero vectors a and b, if |a + b| = |a| - |b|, then a and b are parallel -/
theorem vector_sum_norm_equality_implies_parallel
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ‖a + b‖ = ‖a‖ - ‖b‖) :
  ∃ (k : ℝ), a = k • b :=
sorry

end NUMINAMATH_CALUDE_vector_sum_norm_equality_implies_parallel_l2942_294276


namespace NUMINAMATH_CALUDE_arithmetic_mean_function_is_constant_l2942_294202

/-- A function from ℤ × ℤ to ℤ⁺ satisfying the arithmetic mean property -/
def ArithmeticMeanFunction (f : ℤ × ℤ → ℤ) : Prop :=
  (∀ i j : ℤ, 0 < f (i, j)) ∧ 
  (∀ i j : ℤ, 4 * f (i, j) = f (i-1, j) + f (i+1, j) + f (i, j-1) + f (i, j+1))

/-- Theorem stating that any function satisfying the arithmetic mean property is constant -/
theorem arithmetic_mean_function_is_constant (f : ℤ × ℤ → ℤ) 
  (h : ArithmeticMeanFunction f) : 
  ∃ c : ℤ, ∀ i j : ℤ, f (i, j) = c :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_function_is_constant_l2942_294202


namespace NUMINAMATH_CALUDE_populations_equal_after_16_years_l2942_294233

def village_x_initial_population : ℕ := 74000
def village_x_decrease_rate : ℕ := 1200
def village_y_initial_population : ℕ := 42000
def village_y_increase_rate : ℕ := 800

def population_equal_time : ℕ := 16

theorem populations_equal_after_16_years :
  village_x_initial_population - population_equal_time * village_x_decrease_rate =
  village_y_initial_population + population_equal_time * village_y_increase_rate :=
by sorry

end NUMINAMATH_CALUDE_populations_equal_after_16_years_l2942_294233


namespace NUMINAMATH_CALUDE_doughnut_costs_9_l2942_294222

/-- The price of a cake in Kč -/
def cake_price : ℕ := sorry

/-- The price of a doughnut in Kč -/
def doughnut_price : ℕ := sorry

/-- The amount of pocket money Honzík has in Kč -/
def pocket_money : ℕ := sorry

/-- Theorem stating the price of one doughnut is 9 Kč -/
theorem doughnut_costs_9 
  (h1 : pocket_money - 4 * cake_price = 5)
  (h2 : 5 * cake_price - pocket_money = 6)
  (h3 : 2 * cake_price + 3 * doughnut_price = pocket_money) :
  doughnut_price = 9 := by sorry

end NUMINAMATH_CALUDE_doughnut_costs_9_l2942_294222


namespace NUMINAMATH_CALUDE_least_x_1894x_divisible_by_3_l2942_294253

theorem least_x_1894x_divisible_by_3 :
  ∃ x : ℕ, x < 10 ∧ (1894 * x) % 3 = 0 ∧ ∀ y : ℕ, y < x → y < 10 → (1894 * y) % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_x_1894x_divisible_by_3_l2942_294253


namespace NUMINAMATH_CALUDE_sum_integers_between_two_and_eleven_l2942_294260

theorem sum_integers_between_two_and_eleven : 
  (Finset.range 8).sum (fun i => i + 3) = 52 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_between_two_and_eleven_l2942_294260


namespace NUMINAMATH_CALUDE_triangle_angle_measures_l2942_294259

theorem triangle_angle_measures :
  ∀ (A B C : ℝ),
  (A + B + C = 180) →
  (B = 2 * A) →
  (C + A + B = 180) →
  ∃ (x : ℝ),
    A = x ∧
    B = 2 * x ∧
    C = 180 - 3 * x :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measures_l2942_294259


namespace NUMINAMATH_CALUDE_jason_toys_count_l2942_294264

theorem jason_toys_count :
  ∀ (rachel_toys john_toys jason_toys : ℕ),
    rachel_toys = 1 →
    john_toys = rachel_toys + 6 →
    jason_toys = 3 * john_toys →
    jason_toys = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_jason_toys_count_l2942_294264


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2942_294225

theorem gcd_lcm_product (a b : ℕ) (h : a = 140 ∧ b = 175) : 
  (Nat.gcd a b) * (Nat.lcm a b) = 24500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2942_294225


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l2942_294266

def four_digit_numbers_with_1_and_2 : ℕ :=
  let one_one := 4  -- 1 occurrence of 1, 3 occurrences of 2
  let two_ones := 6 -- 2 occurrences of 1, 2 occurrences of 2
  let three_ones := 4 -- 3 occurrences of 1, 1 occurrence of 2
  one_one + two_ones + three_ones

theorem count_four_digit_numbers : four_digit_numbers_with_1_and_2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l2942_294266


namespace NUMINAMATH_CALUDE_sqrt_four_twentyfifths_equals_two_fifths_l2942_294250

theorem sqrt_four_twentyfifths_equals_two_fifths : 
  Real.sqrt (4 / 25) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_twentyfifths_equals_two_fifths_l2942_294250


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l2942_294288

theorem least_three_digit_multiple_of_11 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n → 110 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l2942_294288


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2942_294249

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 1} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2942_294249


namespace NUMINAMATH_CALUDE_roots_properties_l2942_294252

-- Define the coefficients of the quadratic equation
def a : ℝ := 24
def b : ℝ := 60
def c : ℝ := -600

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem statement
theorem roots_properties :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y) →
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y ∧ x * y = -25 ∧ x + y = -2.5) :=
sorry

end NUMINAMATH_CALUDE_roots_properties_l2942_294252


namespace NUMINAMATH_CALUDE_multiple_of_five_last_digit_l2942_294278

def is_multiple_of_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def last_digit (n : ℕ) : ℕ := n % 10

def five_digit_number (d : ℕ) : ℕ := 45670 + d

theorem multiple_of_five_last_digit (d : ℕ) (h : d < 10) : 
  is_multiple_of_five (five_digit_number d) ↔ (d = 0 ∨ d = 5) :=
sorry

end NUMINAMATH_CALUDE_multiple_of_five_last_digit_l2942_294278


namespace NUMINAMATH_CALUDE_custom_op_example_l2942_294280

-- Define the custom operation
def custom_op (a b : Int) : Int := a * (b + 1) + a * b

-- State the theorem
theorem custom_op_example : custom_op (-3) 4 = -27 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2942_294280


namespace NUMINAMATH_CALUDE_series_sum_equals_first_term_l2942_294213

def decreasing_to_zero (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a n ≥ a (n + 1)) ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, a n < ε)

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a n - 2 * a (n + 1) + a (n + 2)

theorem series_sum_equals_first_term (a : ℕ → ℝ) :
  decreasing_to_zero a →
  (∀ n, b a n ≥ 0) →
  (∑' n, n * b a n) = a 1 :=
sorry

end NUMINAMATH_CALUDE_series_sum_equals_first_term_l2942_294213


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_2000_mod_13_l2942_294251

theorem remainder_of_3_pow_2000_mod_13 : (3^2000 : ℕ) % 13 = 9 := by sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_2000_mod_13_l2942_294251


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2942_294205

theorem condition_sufficient_not_necessary (a b : ℝ) :
  ((1 < b) ∧ (b < a)) → (a - 1 > |b - 1|) ∧
  ¬(∀ a b : ℝ, (a - 1 > |b - 1|) → ((1 < b) ∧ (b < a))) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2942_294205


namespace NUMINAMATH_CALUDE_sam_distance_l2942_294218

theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) : 
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l2942_294218


namespace NUMINAMATH_CALUDE_gambler_final_amount_l2942_294248

def gamble (initial : ℚ) (rounds : ℕ) (wins : ℕ) (losses : ℕ) : ℚ :=
  let bet_fraction : ℚ := 1/3
  let win_multiplier : ℚ := 2
  let loss_multiplier : ℚ := 1
  sorry

theorem gambler_final_amount :
  let initial_amount : ℚ := 100
  let total_rounds : ℕ := 4
  let wins : ℕ := 2
  let losses : ℕ := 2
  gamble initial_amount total_rounds wins losses = 8000/81 := by sorry

end NUMINAMATH_CALUDE_gambler_final_amount_l2942_294248


namespace NUMINAMATH_CALUDE_cafeteria_pie_problem_l2942_294234

/-- Given a cafeteria with initial apples, apples handed out, and number of pies made,
    calculate the number of apples used for each pie. -/
def apples_per_pie (initial_apples : ℕ) (apples_handed_out : ℕ) (num_pies : ℕ) : ℕ :=
  (initial_apples - apples_handed_out) / num_pies

/-- Theorem stating that given 47 initial apples, 27 apples handed out, and 5 pies made,
    the number of apples used for each pie is 4. -/
theorem cafeteria_pie_problem :
  apples_per_pie 47 27 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pie_problem_l2942_294234


namespace NUMINAMATH_CALUDE_largest_non_representable_l2942_294262

/-- Coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (λ i => 2^(n - i) * 3^i)

/-- A number is representable if it can be expressed as a sum of coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The largest non-representable amount in Limonia's currency system -/
theorem largest_non_representable (n : ℕ) :
  ¬ is_representable (3^(n+1) - 2^(n+2)) n ∧
  ∀ s, s > 3^(n+1) - 2^(n+2) → is_representable s n :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_l2942_294262


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2942_294247

/-- The line y = kx + 2 intersects the circle (x-3)^2 + (y-2)^2 = 4 at points M and N.
    If |MN| ≥ 2√3, then k ∈ [-√2/4, √2/4]. -/
theorem intersection_chord_length (k : ℝ) : 
  let line (x : ℝ) := k * x + 2
  let circle (x y : ℝ) := (x - 3)^2 + (y - 2)^2 = 4
  let M : ℝ × ℝ := sorry
  let N : ℝ × ℝ := sorry
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    M = (x₁, y₁) ∧ N = (x₂, y₂) ∧ 
    line x₁ = y₁ ∧ line x₂ = y₂ ∧ 
    circle x₁ y₁ ∧ circle x₂ y₂) →
  (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ≥ 2 * Real.sqrt 3) →
  k ∈ Set.Icc (-Real.sqrt 2 / 4) (Real.sqrt 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2942_294247


namespace NUMINAMATH_CALUDE_cube_divisibility_l2942_294223

theorem cube_divisibility (k : ℕ) (n : ℕ) : 
  (k ≥ 30) → 
  (∀ m : ℕ, m ≥ 30 → m < k → ¬(∃ p : ℕ, m^3 = p * n)) → 
  (∃ q : ℕ, k^3 = q * n) → 
  n = 27000 := by
sorry

end NUMINAMATH_CALUDE_cube_divisibility_l2942_294223


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2942_294207

/-- A quadratic function of the form y = x^2 + mx + m^2 - 3 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + m*x + m^2 - 3

theorem quadratic_function_properties :
  ∀ m : ℝ, m > 0 →
  quadratic_function m 2 = 4 →
  (m = 1 ∧ ∃ x y : ℝ, x ≠ y ∧ quadratic_function m x = 0 ∧ quadratic_function m y = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2942_294207


namespace NUMINAMATH_CALUDE_chips_and_juice_weight_l2942_294299

/-- Given the weight of chips and juice bottles, calculate the total weight of a specific quantity -/
theorem chips_and_juice_weight
  (chip_weight : ℝ) -- Weight of a bag of chips
  (juice_weight : ℝ) -- Weight of a bottle of juice
  (h1 : 2 * chip_weight = 800) -- Weight of 2 bags of chips is 800 g
  (h2 : chip_weight = juice_weight + 350) -- A bag of chips is 350 g heavier than a bottle of juice
  : 5 * chip_weight + 4 * juice_weight = 2200 := by
  sorry

end NUMINAMATH_CALUDE_chips_and_juice_weight_l2942_294299
