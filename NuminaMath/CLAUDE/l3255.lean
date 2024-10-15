import Mathlib

namespace NUMINAMATH_CALUDE_estimate_rabbit_population_l3255_325568

/-- Estimate the number of rabbits in a forest using the capture-recapture method. -/
theorem estimate_rabbit_population (initial_marked : ℕ) (second_capture : ℕ) (marked_in_second : ℕ) :
  initial_marked = 50 →
  second_capture = 42 →
  marked_in_second = 5 →
  (initial_marked * second_capture) / marked_in_second = 420 :=
by
  sorry

#check estimate_rabbit_population

end NUMINAMATH_CALUDE_estimate_rabbit_population_l3255_325568


namespace NUMINAMATH_CALUDE_viewer_increase_l3255_325571

/-- The number of people who watched the second baseball game -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first baseball game -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The number of people who watched the third baseball game -/
def third_game_viewers : ℕ := second_game_viewers + 15

/-- The total number of people who watched the games last week -/
def last_week_viewers : ℕ := 200

/-- The total number of people who watched the games this week -/
def this_week_viewers : ℕ := first_game_viewers + second_game_viewers + third_game_viewers

theorem viewer_increase :
  this_week_viewers - last_week_viewers = 35 := by
  sorry

end NUMINAMATH_CALUDE_viewer_increase_l3255_325571


namespace NUMINAMATH_CALUDE_total_fish_count_l3255_325577

/-- The number of fish in three tanks given specific conditions -/
def total_fish (goldfish1 guppies1 : ℕ) : ℕ :=
  let tank1 := goldfish1 + guppies1
  let tank2 := 2 * goldfish1 + 3 * guppies1
  let tank3 := 3 * goldfish1 + 2 * guppies1
  tank1 + tank2 + tank3

/-- Theorem stating that the total number of fish is 162 given the specific conditions -/
theorem total_fish_count : total_fish 15 12 = 162 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3255_325577


namespace NUMINAMATH_CALUDE_diamond_calculation_l3255_325567

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 3 4) 5
  let y := diamond 3 (diamond 4 5)
  x - y = -71 / 380 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3255_325567


namespace NUMINAMATH_CALUDE_digit_2003_is_4_l3255_325553

/-- Calculates the digit at a given position in the sequence of natural numbers written consecutively -/
def digitAtPosition (n : ℕ) : ℕ :=
  sorry

/-- The 2003rd digit in the sequence of natural numbers written consecutively is 4 -/
theorem digit_2003_is_4 : digitAtPosition 2003 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_2003_is_4_l3255_325553


namespace NUMINAMATH_CALUDE_product_equals_result_l3255_325545

theorem product_equals_result : ∃ x : ℝ, 469158 * x = 4691110842 ∧ x = 10000.2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_result_l3255_325545


namespace NUMINAMATH_CALUDE_negation_equivalence_l3255_325547

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Adult : U → Prop)
variable (GoodCook : U → Prop)

-- Define the statements
def AllAdultsAreGoodCooks : Prop := ∀ x, Adult x → GoodCook x
def AtLeastOneAdultIsBadCook : Prop := ∃ x, Adult x ∧ ¬GoodCook x

-- Theorem statement
theorem negation_equivalence : 
  AtLeastOneAdultIsBadCook U Adult GoodCook ↔ ¬(AllAdultsAreGoodCooks U Adult GoodCook) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3255_325547


namespace NUMINAMATH_CALUDE_ralphs_initial_cards_l3255_325542

theorem ralphs_initial_cards (cards_from_father cards_after : ℕ) :
  cards_from_father = 8 →
  cards_after = 12 →
  cards_after - cards_from_father = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ralphs_initial_cards_l3255_325542


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3255_325526

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.75 * P) : 
  M / N = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3255_325526


namespace NUMINAMATH_CALUDE_beth_marbles_l3255_325508

/-- The number of marbles Beth has initially -/
def initial_marbles : ℕ := 72

/-- The number of colors of marbles -/
def num_colors : ℕ := 3

/-- The number of red marbles Beth loses -/
def lost_red : ℕ := 5

/-- Calculates the number of marbles Beth has left after losing some -/
def marbles_left (initial : ℕ) (colors : ℕ) (lost_red : ℕ) : ℕ :=
  initial - (lost_red + 2 * lost_red + 3 * lost_red)

theorem beth_marbles :
  marbles_left initial_marbles num_colors lost_red = 42 := by
  sorry

end NUMINAMATH_CALUDE_beth_marbles_l3255_325508


namespace NUMINAMATH_CALUDE_quadratic_function_form_l3255_325582

/-- A quadratic function with two equal real roots and a specific derivative -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧ 
  (∃! r : ℝ, f r = 0) ∧
  (∀ x, deriv f x = 2 * x + 2)

/-- The theorem stating the specific form of the quadratic function -/
theorem quadratic_function_form (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∀ x, f x = x^2 + 2*x + 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_form_l3255_325582


namespace NUMINAMATH_CALUDE_givenPoint_on_y_axis_l3255_325559

/-- A point in the Cartesian coordinate system. -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on the y-axis. -/
def OnYAxis (p : CartesianPoint) : Prop :=
  p.x = 0

/-- The given point (0, -1) in the Cartesian coordinate system. -/
def givenPoint : CartesianPoint :=
  ⟨0, -1⟩

/-- Theorem stating that the given point lies on the y-axis. -/
theorem givenPoint_on_y_axis : OnYAxis givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPoint_on_y_axis_l3255_325559


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3255_325523

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b : V)

-- State the theorem
theorem vector_difference_magnitude 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = 1) : 
  ‖a - b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3255_325523


namespace NUMINAMATH_CALUDE_train_speed_problem_l3255_325583

theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  faster_speed = 45 →
  passing_time = 9.599232061435085 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    (faster_speed + slower_speed) * (passing_time / 3600) = 2 * (train_length / 1000) ∧
    slower_speed = 30 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3255_325583


namespace NUMINAMATH_CALUDE_function_machine_output_l3255_325558

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 25 then
    step1 - 7
  else
    (step1 + 3) * 2

theorem function_machine_output : function_machine 12 = 78 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l3255_325558


namespace NUMINAMATH_CALUDE_sixth_term_before_three_l3255_325562

def fibonacci_like_sequence (a : ℤ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem sixth_term_before_three (a : ℤ → ℤ) :
  fibonacci_like_sequence a →
  a 0 = 3 ∧ a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 13 ∧ a 4 = 21 →
  a (-6) = -1 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_before_three_l3255_325562


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3255_325516

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x => Real.sqrt x
  let h : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  ∃ x₀ y₀ : ℝ,
    x₀ ≥ 0 ∧
    y₀ = f x₀ ∧
    h x₀ y₀ ∧
    (λ x => (f x₀ - 0) / (x₀ - (-1)) * (x - x₀) + f x₀) (-1) = 0 →
    (Real.sqrt (a^2 + b^2)) / a = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3255_325516


namespace NUMINAMATH_CALUDE_ice_cream_cost_l3255_325557

/-- Proves that the cost of a scoop of ice cream is $5 given the problem conditions -/
theorem ice_cream_cost (people : ℕ) (meal_cost : ℕ) (total_money : ℕ) 
  (h1 : people = 3)
  (h2 : meal_cost = 10)
  (h3 : total_money = 45)
  (h4 : ∃ (ice_cream_cost : ℕ), total_money = people * meal_cost + people * ice_cream_cost) :
  ∃ (ice_cream_cost : ℕ), ice_cream_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l3255_325557


namespace NUMINAMATH_CALUDE_tangent_circles_area_sum_l3255_325566

/-- A right triangle with sides 6, 8, and 10, where each vertex is the center of a circle
    and each circle is externally tangent to the other two. -/
structure TangentCirclesTriangle where
  -- The sides of the triangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- The radii of the circles
  radius1 : ℝ
  radius2 : ℝ
  radius3 : ℝ
  -- Conditions
  is_right_triangle : side1^2 + side2^2 = side3^2
  side_lengths : side1 = 6 ∧ side2 = 8 ∧ side3 = 10
  tangency1 : radius1 + radius2 = side3
  tangency2 : radius2 + radius3 = side1
  tangency3 : radius1 + radius3 = side2

/-- The sum of the areas of the circles in a TangentCirclesTriangle is 56π. -/
theorem tangent_circles_area_sum (t : TangentCirclesTriangle) :
  π * (t.radius1^2 + t.radius2^2 + t.radius3^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_area_sum_l3255_325566


namespace NUMINAMATH_CALUDE_pencils_left_l3255_325552

/-- Calculates the number of pencils Steve has left after giving some to Matt and Lauren -/
theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra : ℕ) : 
  boxes * pencils_per_box - (lauren_pencils + (lauren_pencils + matt_extra)) = 9 :=
by
  sorry

#check pencils_left 2 12 6 3

end NUMINAMATH_CALUDE_pencils_left_l3255_325552


namespace NUMINAMATH_CALUDE_fourth_black_ball_probability_l3255_325570

/-- Represents a box of colored balls -/
structure ColoredBallBox where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball on any draw -/
def prob_black_ball (box : ColoredBallBox) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The box described in the problem -/
def problem_box : ColoredBallBox :=
  { red_balls := 3, black_balls := 4 }

theorem fourth_black_ball_probability :
  prob_black_ball problem_box = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_black_ball_probability_l3255_325570


namespace NUMINAMATH_CALUDE_function_inequality_l3255_325515

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) (a : ℝ) (ha : a > 0) : 
  f a < Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3255_325515


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3255_325504

theorem base_conversion_problem :
  ∃! (x y z b : ℕ),
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x < b ∧ y < b ∧ z < b ∧
    b > 10 ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3255_325504


namespace NUMINAMATH_CALUDE_expansion_terms_product_l3255_325518

/-- The number of terms in the expansion of a product of two polynomials -/
def expansion_terms (n m : ℕ) : ℕ := n * m

theorem expansion_terms_product (n m : ℕ) (h1 : n = 3) (h2 : m = 5) :
  expansion_terms n m = 15 := by
  sorry

#check expansion_terms_product

end NUMINAMATH_CALUDE_expansion_terms_product_l3255_325518


namespace NUMINAMATH_CALUDE_magnitude_of_BC_l3255_325589

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, -2)
def AC : ℝ × ℝ := (4, -1)

theorem magnitude_of_BC : 
  let C : ℝ × ℝ := (A.1 + AC.1, A.2 + AC.2)
  let BC : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)
  Real.sqrt ((BC.1)^2 + (BC.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_BC_l3255_325589


namespace NUMINAMATH_CALUDE_treasure_day_l3255_325544

/-- Pongpong's starting amount -/
def pongpong_start : ℕ := 8000

/-- Longlong's starting amount -/
def longlong_start : ℕ := 5000

/-- Pongpong's daily increase -/
def pongpong_daily : ℕ := 300

/-- Longlong's daily increase -/
def longlong_daily : ℕ := 500

/-- The number of days until Pongpong and Longlong have the same amount -/
def days_until_equal : ℕ := 15

theorem treasure_day :
  pongpong_start + pongpong_daily * days_until_equal =
  longlong_start + longlong_daily * days_until_equal :=
by sorry

end NUMINAMATH_CALUDE_treasure_day_l3255_325544


namespace NUMINAMATH_CALUDE_triangle_side_length_l3255_325555

theorem triangle_side_length (a : ℕ) : 
  (a % 2 = 1) → -- a is odd
  (2 + a > 3) ∧ (2 + 3 > a) ∧ (a + 3 > 2) → -- triangle inequality
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3255_325555


namespace NUMINAMATH_CALUDE_correct_multiple_choice_count_l3255_325505

/-- Represents the citizenship test with multiple-choice and fill-in-the-blank questions. -/
structure CitizenshipTest where
  totalQuestions : ℕ
  multipleChoiceTime : ℕ
  fillInBlankTime : ℕ
  totalStudyTime : ℕ

/-- Calculates the number of multiple-choice questions on the test. -/
def multipleChoiceCount (test : CitizenshipTest) : ℕ :=
  30

/-- Theorem stating that for the given test parameters, 
    the number of multiple-choice questions is 30. -/
theorem correct_multiple_choice_count 
  (test : CitizenshipTest)
  (h1 : test.totalQuestions = 60)
  (h2 : test.multipleChoiceTime = 15)
  (h3 : test.fillInBlankTime = 25)
  (h4 : test.totalStudyTime = 1200) :
  multipleChoiceCount test = 30 := by
  sorry

#eval multipleChoiceCount {
  totalQuestions := 60,
  multipleChoiceTime := 15,
  fillInBlankTime := 25,
  totalStudyTime := 1200
}

end NUMINAMATH_CALUDE_correct_multiple_choice_count_l3255_325505


namespace NUMINAMATH_CALUDE_smith_family_buffet_cost_l3255_325574

/-- Represents the cost calculation for a family at a seafood buffet. -/
def buffet_cost (adult_price : ℚ) (child_price : ℚ) (senior_discount : ℚ) 
  (num_adults num_seniors num_children : ℕ) : ℚ :=
  (num_adults * adult_price) + 
  (num_seniors * (adult_price * (1 - senior_discount))) + 
  (num_children * child_price)

/-- Theorem stating that the total cost for Mr. Smith's family at the seafood buffet is $159. -/
theorem smith_family_buffet_cost : 
  buffet_cost 30 15 (1/10) 2 2 3 = 159 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_buffet_cost_l3255_325574


namespace NUMINAMATH_CALUDE_quincy_peter_difference_l3255_325594

/-- The number of pictures Randy drew -/
def randy_pictures : ℕ := 5

/-- The number of additional pictures Peter drew compared to Randy -/
def peter_additional : ℕ := 3

/-- The total number of pictures drawn by all three -/
def total_pictures : ℕ := 41

/-- The number of pictures Peter drew -/
def peter_pictures : ℕ := randy_pictures + peter_additional

/-- The number of pictures Quincy drew -/
def quincy_pictures : ℕ := total_pictures - randy_pictures - peter_pictures

theorem quincy_peter_difference : quincy_pictures - peter_pictures = 20 := by
  sorry

end NUMINAMATH_CALUDE_quincy_peter_difference_l3255_325594


namespace NUMINAMATH_CALUDE_code_cracking_probability_l3255_325560

theorem code_cracking_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_code_cracking_probability_l3255_325560


namespace NUMINAMATH_CALUDE_factor_polynomial_l3255_325513

theorem factor_polynomial (x : ℝ) : 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3255_325513


namespace NUMINAMATH_CALUDE_inequality_proof_l3255_325541

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 3 / 2) :
  x + 4 * y + 9 * z ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3255_325541


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l3255_325564

/-- Represents the capital contribution and duration for a business partner -/
structure Partner where
  capital : ℕ
  duration : ℕ

/-- Calculates the effective capital contribution of a partner -/
def effectiveCapital (p : Partner) : ℕ := p.capital * p.duration

/-- Represents the business scenario with two partners -/
structure Business where
  partnerA : Partner
  partnerB : Partner

/-- The given business scenario -/
def givenBusiness : Business :=
  { partnerA := { capital := 3500, duration := 12 }
  , partnerB := { capital := 21000, duration := 3 }
  }

/-- Theorem stating that the profit sharing ratio is 2:3 for the given business -/
theorem profit_sharing_ratio (b : Business := givenBusiness) :
  (effectiveCapital b.partnerA) * 3 = (effectiveCapital b.partnerB) * 2 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l3255_325564


namespace NUMINAMATH_CALUDE_dress_design_count_l3255_325527

/-- The number of fabric colors available -/
def num_colors : ℕ := 3

/-- The number of fabric types available -/
def num_fabric_types : ℕ := 4

/-- The number of patterns available -/
def num_patterns : ℕ := 3

/-- Each dress design requires exactly one color, one fabric type, and one pattern -/
axiom dress_design_requirements : True

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_fabric_types * num_patterns

theorem dress_design_count : total_designs = 36 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_count_l3255_325527


namespace NUMINAMATH_CALUDE_paint_cost_theorem_l3255_325517

-- Define the paint properties
structure Paint where
  cost : Float
  coverage : Float

-- Define the cuboid dimensions
def cuboid_length : Float := 12
def cuboid_width : Float := 15
def cuboid_height : Float := 20

-- Define the paints
def paint_A : Paint := { cost := 3.20, coverage := 60 }
def paint_B : Paint := { cost := 5.50, coverage := 55 }
def paint_C : Paint := { cost := 4.00, coverage := 50 }

-- Calculate the areas of the faces
def largest_face_area : Float := 2 * cuboid_width * cuboid_height
def middle_face_area : Float := 2 * cuboid_length * cuboid_height
def smallest_face_area : Float := 2 * cuboid_length * cuboid_width

-- Calculate the number of quarts needed for each paint
def quarts_A : Float := Float.ceil (largest_face_area / paint_A.coverage)
def quarts_B : Float := Float.ceil (middle_face_area / paint_B.coverage)
def quarts_C : Float := Float.ceil (smallest_face_area / paint_C.coverage)

-- Calculate the total cost
def total_cost : Float := quarts_A * paint_A.cost + quarts_B * paint_B.cost + quarts_C * paint_C.cost

-- Theorem to prove
theorem paint_cost_theorem : total_cost = 113.50 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_theorem_l3255_325517


namespace NUMINAMATH_CALUDE_sequence_properties_l3255_325509

def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a (n + 1) = a n * q

theorem sequence_properties (a b : ℕ → ℕ) (k : ℝ) :
  geometric_sequence a →
  (a 1 = 3) →
  (2 * a 3 = a 2 + (3/4) * a 4) →
  (b 1 = 1) →
  (∀ n : ℕ, b (n + 1) = 2 * b n + 1) →
  (∀ n : ℕ, k * ((b n + 5) / 2) - a n ≥ 8 * n + 2 * k - 24) →
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, b n = 2^n - 1) ∧
  (k ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3255_325509


namespace NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l3255_325561

/-- Given a point P in a 3D Cartesian coordinate system, 
    return its symmetric point P' with respect to the y-axis -/
def symmetric_point_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, -z)

theorem symmetry_wrt_y_axis :
  let P : ℝ × ℝ × ℝ := (2, -4, 6)
  symmetric_point_y_axis P = (-2, -4, -6) := by
sorry

end NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l3255_325561


namespace NUMINAMATH_CALUDE_min_value_theorem_l3255_325528

theorem min_value_theorem (x : ℝ) (h : x > 6) :
  x^2 / (x - 6) ≥ 18 ∧ (x^2 / (x - 6) = 18 ↔ x = 12) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3255_325528


namespace NUMINAMATH_CALUDE_left_handed_classical_music_lovers_l3255_325538

theorem left_handed_classical_music_lovers (total : ℕ) (left_handed : ℕ) (classical_music_lovers : ℕ) (right_handed_non_lovers : ℕ) :
  total = 30 →
  left_handed = 12 →
  classical_music_lovers = 20 →
  right_handed_non_lovers = 3 →
  ∃ (x : ℕ), x = 5 ∧ 
    x + (left_handed - x) + (classical_music_lovers - x) + right_handed_non_lovers = total :=
by sorry

end NUMINAMATH_CALUDE_left_handed_classical_music_lovers_l3255_325538


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l3255_325546

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an additional inning -/
def newAverage (bp : BatsmanPerformance) (newRuns : Nat) : Rat :=
  (bp.totalRuns + newRuns) / (bp.innings + 1)

/-- Theorem: Given the conditions, prove that the batsman's average after the 17th inning is 39 -/
theorem batsman_average_after_17th_inning 
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 16)
  (h2 : newAverage bp 87 = bp.average + 3)
  : newAverage bp 87 = 39 := by
  sorry

#check batsman_average_after_17th_inning

end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l3255_325546


namespace NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l3255_325507

-- Define propositions p and q
def p (x : ℝ) : Prop := abs (x + 2) > 2
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬q is necessary but not sufficient for ¬p
theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  ¬(∀ x : ℝ, not_q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l3255_325507


namespace NUMINAMATH_CALUDE_village_population_equality_l3255_325592

/-- The number of years it takes for two village populations to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_rate + x_rate)

/-- Theorem stating that the populations of Village X and Village Y will be equal after 16 years -/
theorem village_population_equality :
  years_to_equal_population 74000 1200 42000 800 = 16 := by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l3255_325592


namespace NUMINAMATH_CALUDE_rich_book_pages_left_to_read_l3255_325540

/-- Given a book with a total number of pages, the number of pages already read,
    and the number of pages to be skipped, calculate the number of pages left to read. -/
def pages_left_to_read (total_pages read_pages skipped_pages : ℕ) : ℕ :=
  total_pages - (read_pages + skipped_pages)

/-- Theorem stating that for a 372-page book with 125 pages read and 16 pages skipped,
    there are 231 pages left to read. -/
theorem rich_book_pages_left_to_read :
  pages_left_to_read 372 125 16 = 231 := by
  sorry

end NUMINAMATH_CALUDE_rich_book_pages_left_to_read_l3255_325540


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3255_325576

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℝ) 
                                  (set2_count : ℕ) (set2_mean : ℝ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 27 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℝ) = 21.4 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3255_325576


namespace NUMINAMATH_CALUDE_jerrys_age_l3255_325590

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 22 → 
  mickey_age = 2 * jerry_age - 6 → 
  jerry_age = 14 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3255_325590


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l3255_325510

/-- The length of the tangent segment from the origin to a circle passing through three given points -/
theorem tangent_length_to_circle (A B C : ℝ × ℝ) : 
  A = (2, 3) → B = (4, 6) → C = (3, 9) → 
  ∃ (circle : Set (ℝ × ℝ)), 
    (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle) ∧
    (∃ (T : ℝ × ℝ), T ∈ circle ∧ 
      (∀ (P : ℝ × ℝ), P ∈ circle → dist (0, 0) P ≥ dist (0, 0) T) ∧
      dist (0, 0) T = Real.sqrt (10 + 3 * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l3255_325510


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3255_325512

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_sum : a 3 + a 4 = 9) : 
  a 1 * a 6 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3255_325512


namespace NUMINAMATH_CALUDE_fifteenth_number_with_digit_sum_14_l3255_325599

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 14 -/
def nth_number_with_digit_sum_14 (n : ℕ+) : ℕ+ := sorry

/-- The main theorem -/
theorem fifteenth_number_with_digit_sum_14 :
  nth_number_with_digit_sum_14 15 = 266 := by sorry

end NUMINAMATH_CALUDE_fifteenth_number_with_digit_sum_14_l3255_325599


namespace NUMINAMATH_CALUDE_train_crossing_time_l3255_325595

/-- Proves that a train 400 meters long, traveling at 36 km/h, takes 40 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 →
  train_speed_kmh = 36 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 40 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3255_325595


namespace NUMINAMATH_CALUDE_partitioned_triangle_area_l3255_325563

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  area_quad : ℝ

/-- The theorem to be proved -/
theorem partitioned_triangle_area 
  (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 12) : 
  t.area_quad = 16 := by
  sorry


end NUMINAMATH_CALUDE_partitioned_triangle_area_l3255_325563


namespace NUMINAMATH_CALUDE_fib_mod_5_periodic_fib_10_mod_5_fib_50_mod_5_l3255_325531

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_mod_5_periodic (n : ℕ) : fib n % 5 = fib (n % 20) % 5 := sorry

theorem fib_10_mod_5 : fib 10 % 5 = 0 := sorry

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_mod_5_periodic_fib_10_mod_5_fib_50_mod_5_l3255_325531


namespace NUMINAMATH_CALUDE_negation_equivalence_l3255_325551

theorem negation_equivalence (m : ℝ) :
  (¬ ∃ (x : ℤ), x^2 + x + m < 0) ↔ (∀ (x : ℝ), x^2 + x + m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3255_325551


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3255_325585

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n ∧ a n > 0

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16 →
  a 3 + a 6 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l3255_325585


namespace NUMINAMATH_CALUDE_course_selection_schemes_l3255_325598

/-- The number of elective courses in physical education -/
def pe_courses : ℕ := 4

/-- The number of elective courses in art -/
def art_courses : ℕ := 4

/-- The minimum number of courses a student can choose -/
def min_courses : ℕ := 2

/-- The maximum number of courses a student can choose -/
def max_courses : ℕ := 3

/-- The minimum number of courses a student must choose from each category -/
def min_per_category : ℕ := 1

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := 64

/-- Theorem stating that the total number of different course selection schemes is 64 -/
theorem course_selection_schemes :
  (pe_courses = 4) →
  (art_courses = 4) →
  (min_courses = 2) →
  (max_courses = 3) →
  (min_per_category = 1) →
  (total_schemes = 64) :=
by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l3255_325598


namespace NUMINAMATH_CALUDE_equivalence_of_equations_l3255_325511

theorem equivalence_of_equations (p : ℕ) (hp : Nat.Prime p) :
  (∃ (x s : ℤ), x^2 - x + 3 - p * s = 0) ↔
  (∃ (y t : ℤ), y^2 - y + 25 - p * t = 0) := by
sorry

end NUMINAMATH_CALUDE_equivalence_of_equations_l3255_325511


namespace NUMINAMATH_CALUDE_problem_statement_l3255_325543

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) → m ≥ 1/4) ∧
  (∀ x : ℝ, (1/a + 1/b ≥ |2*x - 1| - |x + 1|) ↔ -2 ≤ x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3255_325543


namespace NUMINAMATH_CALUDE_player1_wins_l3255_325578

/-- Represents the state of the game -/
structure GameState :=
  (coins : ℕ)

/-- Represents a player's move -/
structure Move :=
  (coins_taken : ℕ)

/-- Defines a valid move for Player 1 -/
def valid_move_player1 (m : Move) : Prop :=
  m.coins_taken % 2 = 1 ∧ 1 ≤ m.coins_taken ∧ m.coins_taken ≤ 99

/-- Defines a valid move for Player 2 -/
def valid_move_player2 (m : Move) : Prop :=
  m.coins_taken % 2 = 0 ∧ 2 ≤ m.coins_taken ∧ m.coins_taken ≤ 100

/-- Defines the game transition for a player's move -/
def make_move (s : GameState) (m : Move) : GameState :=
  ⟨s.coins - m.coins_taken⟩

/-- Defines a winning strategy for Player 1 -/
def winning_strategy (initial_coins : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    (∀ s : GameState, valid_move_player1 (strategy s)) ∧
    (∀ s : GameState, ∀ m : Move, 
      valid_move_player2 m → 
      ∃ (next_move : Move), 
        valid_move_player1 next_move ∧
        make_move (make_move s m) next_move = ⟨0⟩)

theorem player1_wins : winning_strategy 2001 := by
  sorry


end NUMINAMATH_CALUDE_player1_wins_l3255_325578


namespace NUMINAMATH_CALUDE_alligators_count_l3255_325521

/-- Given the number of alligators seen by Samara and her friends, prove the total number of alligators seen. -/
theorem alligators_count (samara_count : ℕ) (friend_count : ℕ) (friend_average : ℕ) : 
  samara_count = 20 → friend_count = 3 → friend_average = 10 →
  samara_count + friend_count * friend_average = 50 := by
  sorry


end NUMINAMATH_CALUDE_alligators_count_l3255_325521


namespace NUMINAMATH_CALUDE_missing_number_is_34_l3255_325550

theorem missing_number_is_34 : 
  ∃ x : ℝ, ((306 / x) * 15 + 270 = 405) ∧ (x = 34) :=
by sorry

end NUMINAMATH_CALUDE_missing_number_is_34_l3255_325550


namespace NUMINAMATH_CALUDE_four_inequalities_true_l3255_325525

theorem four_inequalities_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x < a) (hyb : y < b) 
  (hxneg : x < 0) (hyneg : y < 0)
  (hapos : a > 0) (hbpos : b > 0) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_four_inequalities_true_l3255_325525


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3255_325548

/-- The polynomial function f(x) = x^8 + 6x^7 + 13x^6 + 256x^5 - 684x^4 -/
def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 13*x^6 + 256*x^5 - 684*x^4

/-- The theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3255_325548


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l3255_325536

/-- Represents the number of rooms that can be painted with the original amount of paint -/
def original_rooms : ℕ := 40

/-- Represents the number of rooms that can be painted after losing some paint -/
def remaining_rooms : ℕ := 31

/-- Represents the number of cans lost -/
def lost_cans : ℕ := 3

/-- Calculates the number of cans used to paint a given number of rooms -/
def cans_used (rooms : ℕ) : ℕ :=
  (rooms + 2) / 3

theorem paint_cans_theorem : cans_used remaining_rooms = 11 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l3255_325536


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3255_325581

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (3 + 4 * x) = 7 ∧ x = 11.5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3255_325581


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_11_mod_14_l3255_325506

theorem smallest_five_digit_congruent_to_11_mod_14 :
  ∃ n : ℕ, 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    n % 14 = 11 ∧
    (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 14 = 11 → n ≤ m) ∧
    n = 10007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_11_mod_14_l3255_325506


namespace NUMINAMATH_CALUDE_even_cube_plus_20_divisible_by_48_l3255_325522

theorem even_cube_plus_20_divisible_by_48 (k : ℤ) : 
  ∃ (n : ℤ), 8 * k * (k^2 + 5) = 48 * n := by
  sorry

end NUMINAMATH_CALUDE_even_cube_plus_20_divisible_by_48_l3255_325522


namespace NUMINAMATH_CALUDE_ramon_age_l3255_325593

/-- Ramon's age problem -/
theorem ramon_age (loui_age : ℕ) (ramon_future_age : ℕ) : 
  loui_age = 23 →
  ramon_future_age = 2 * loui_age →
  ramon_future_age - 20 = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_ramon_age_l3255_325593


namespace NUMINAMATH_CALUDE_james_has_more_balloons_l3255_325572

/-- James has 1222 balloons -/
def james_balloons : ℕ := 1222

/-- Amy has 513 balloons -/
def amy_balloons : ℕ := 513

/-- The difference in balloon count between James and Amy -/
def balloon_difference : ℕ := james_balloons - amy_balloons

/-- Theorem stating that James has 709 more balloons than Amy -/
theorem james_has_more_balloons : balloon_difference = 709 := by
  sorry

end NUMINAMATH_CALUDE_james_has_more_balloons_l3255_325572


namespace NUMINAMATH_CALUDE_grandmas_salad_l3255_325539

/-- The number of mushrooms Grandma put on her salad -/
def mushrooms : ℕ := sorry

/-- The number of cherry tomatoes Grandma put on her salad -/
def cherry_tomatoes : ℕ := 2 * mushrooms

/-- The number of pickles Grandma put on her salad -/
def pickles : ℕ := 4 * cherry_tomatoes

/-- The total number of bacon bits Grandma put on her salad -/
def bacon_bits : ℕ := 4 * pickles

/-- The number of red bacon bits Grandma put on her salad -/
def red_bacon_bits : ℕ := 32

theorem grandmas_salad : mushrooms = 3 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_salad_l3255_325539


namespace NUMINAMATH_CALUDE_parabola_property_l3255_325597

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define the perpendicular condition
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  directrix A.1 ∧ (P.2 = A.2)

-- Define the slope condition for AF
def slope_AF_is_neg_one (A : ℝ × ℝ) : Prop :=
  (A.2 - focus.2) / (A.1 - focus.1) = -1

theorem parabola_property :
  ∀ P : ℝ × ℝ,
  point_on_parabola P →
  ∃ A : ℝ × ℝ,
  perpendicular_to_directrix P A ∧
  slope_AF_is_neg_one A →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_property_l3255_325597


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3255_325587

-- Define the properties of the polygon
def perimeter : ℝ := 150
def side_length : ℝ := 15

-- Theorem statement
theorem regular_polygon_sides : 
  perimeter / side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3255_325587


namespace NUMINAMATH_CALUDE_minimum_cost_is_74_l3255_325500

-- Define the box types
inductive BoxType
| A
| B

-- Define the problem parameters
def totalVolume : ℕ := 15
def boxCapacity : BoxType → ℕ
  | BoxType.A => 2
  | BoxType.B => 3
def boxPrice : BoxType → ℕ
  | BoxType.A => 13
  | BoxType.B => 15
def discountThreshold : ℕ := 3
def discountAmount : ℕ := 10

-- Define a purchase plan
def PurchasePlan := BoxType → ℕ

-- Calculate the total volume of a purchase plan
def totalVolumeOfPlan (plan : PurchasePlan) : ℕ :=
  (plan BoxType.A) * (boxCapacity BoxType.A) + (plan BoxType.B) * (boxCapacity BoxType.B)

-- Calculate the cost of a purchase plan
def costOfPlan (plan : PurchasePlan) : ℕ :=
  let basePrice := (plan BoxType.A) * (boxPrice BoxType.A) + (plan BoxType.B) * (boxPrice BoxType.B)
  if plan BoxType.A ≥ discountThreshold then basePrice - discountAmount else basePrice

-- Define a valid purchase plan
def isValidPlan (plan : PurchasePlan) : Prop :=
  totalVolumeOfPlan plan = totalVolume

-- Theorem to prove
theorem minimum_cost_is_74 :
  ∃ (plan : PurchasePlan), isValidPlan plan ∧
    ∀ (otherPlan : PurchasePlan), isValidPlan otherPlan → costOfPlan plan ≤ costOfPlan otherPlan ∧
    costOfPlan plan = 74 :=
  sorry

end NUMINAMATH_CALUDE_minimum_cost_is_74_l3255_325500


namespace NUMINAMATH_CALUDE_vacation_tents_l3255_325591

/-- Represents the sleeping arrangements for a family vacation --/
structure SleepingArrangements where
  indoor_capacity : ℕ
  max_per_tent : ℕ
  total_people : ℕ
  teenagers : ℕ
  young_children : ℕ
  infant_families : ℕ
  single_adults : ℕ
  dogs : ℕ

/-- Calculates the number of tents needed given the sleeping arrangements --/
def calculate_tents (arrangements : SleepingArrangements) : ℕ :=
  let outdoor_people := arrangements.total_people - arrangements.indoor_capacity
  let teen_tents := (arrangements.teenagers + 1) / 2
  let child_tents := (arrangements.young_children + 1) / 2
  let adult_tents := (outdoor_people - arrangements.teenagers - arrangements.young_children - arrangements.infant_families + 1) / 2
  teen_tents + child_tents + adult_tents + arrangements.dogs

/-- Theorem stating that the given sleeping arrangements require 7 tents --/
theorem vacation_tents (arrangements : SleepingArrangements) 
  (h1 : arrangements.indoor_capacity = 6)
  (h2 : arrangements.max_per_tent = 2)
  (h3 : arrangements.total_people = 20)
  (h4 : arrangements.teenagers = 2)
  (h5 : arrangements.young_children = 5)
  (h6 : arrangements.infant_families = 3)
  (h7 : arrangements.single_adults = 1)
  (h8 : arrangements.dogs = 1) :
  calculate_tents arrangements = 7 := by
  sorry


end NUMINAMATH_CALUDE_vacation_tents_l3255_325591


namespace NUMINAMATH_CALUDE_teacher_grading_problem_l3255_325573

def remaining_problems (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem teacher_grading_problem :
  let problems_per_worksheet : ℕ := 3
  let total_worksheets : ℕ := 15
  let graded_worksheets : ℕ := 7
  remaining_problems problems_per_worksheet total_worksheets graded_worksheets = 24 := by
sorry

end NUMINAMATH_CALUDE_teacher_grading_problem_l3255_325573


namespace NUMINAMATH_CALUDE_ab_value_l3255_325514

theorem ab_value (a b : ℕ+) (h : a^2 + 3*b = 33) : a*b = 24 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3255_325514


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3255_325588

theorem quadratic_roots_difference (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 12 = 0 → r₂^2 - 7*r₂ + 12 = 0 → |r₁ - r₂| = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3255_325588


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3255_325579

/-- The coefficient of x^3 in the expansion of (1+ax)^5 -/
def coefficient_x3 (a : ℝ) : ℝ := 10 * a^3

theorem binomial_expansion_coefficient (a : ℝ) :
  coefficient_x3 a = -80 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3255_325579


namespace NUMINAMATH_CALUDE_third_quiz_score_is_92_l3255_325529

/-- Given the average score of three quizzes and the average score of the first two quizzes,
    calculates the score of the third quiz. -/
def third_quiz_score (avg_three : ℚ) (avg_two : ℚ) : ℚ :=
  3 * avg_three - 2 * avg_two

/-- Theorem stating that given the specific average scores,
    the third quiz score is 92. -/
theorem third_quiz_score_is_92 :
  third_quiz_score 94 95 = 92 := by
  sorry

end NUMINAMATH_CALUDE_third_quiz_score_is_92_l3255_325529


namespace NUMINAMATH_CALUDE_fraction_simplification_complex_fraction_simplification_l3255_325586

-- Problem 1
theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

-- Problem 2
theorem complex_fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ((x - 2) / (x - 1)) / ((x^2 - 4*x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_complex_fraction_simplification_l3255_325586


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3255_325549

/-- The parabola y = x^2 - 4 intersects the y-axis at the point (0, -4) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := fun x ↦ x^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3255_325549


namespace NUMINAMATH_CALUDE_betty_blue_beads_l3255_325530

/-- Given a ratio of red to blue beads and a number of red beads, calculate the number of blue beads -/
def calculate_blue_beads (red_ratio : ℕ) (blue_ratio : ℕ) (total_red : ℕ) : ℕ :=
  (total_red / red_ratio) * blue_ratio

/-- Theorem: Given Betty's bead ratio and total red beads, prove she has 20 blue beads -/
theorem betty_blue_beads :
  let red_ratio : ℕ := 3
  let blue_ratio : ℕ := 2
  let total_red : ℕ := 30
  calculate_blue_beads red_ratio blue_ratio total_red = 20 := by
  sorry

#eval calculate_blue_beads 3 2 30

end NUMINAMATH_CALUDE_betty_blue_beads_l3255_325530


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l3255_325575

def number_of_players : ℕ := 16
def number_of_triplets : ℕ := 3
def number_of_twins : ℕ := 2
def number_of_starters : ℕ := 7

def remaining_players : ℕ := number_of_players - number_of_triplets - number_of_twins

theorem volleyball_team_starters :
  (number_of_triplets * number_of_twins * (Nat.choose remaining_players (number_of_starters - 2))) = 2772 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l3255_325575


namespace NUMINAMATH_CALUDE_intersection_distance_implies_k_range_l3255_325537

/-- Given a line y = kx and a circle (x-2)^2 + (y+1)^2 = 4,
    if the distance between their intersection points is at least 2√3,
    then -4/3 ≤ k ≤ 0 -/
theorem intersection_distance_implies_k_range (k : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.2 = k * A.1 ∧ B.2 = k * B.1) ∧
    ((A.1 - 2)^2 + (A.2 + 1)^2 = 4 ∧ (B.1 - 2)^2 + (B.2 + 1)^2 = 4) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ 12)) →
  -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_k_range_l3255_325537


namespace NUMINAMATH_CALUDE_max_area_is_35_l3255_325535

/-- Represents the cost constraint for the rectangular frame -/
def cost_constraint (l w : ℕ) : Prop := 3 * l + 5 * w ≤ 50

/-- Represents the area of the rectangular frame -/
def area (l w : ℕ) : ℕ := l * w

/-- Theorem stating that the maximum area of the rectangular frame is 35 m² -/
theorem max_area_is_35 :
  ∃ (l w : ℕ), cost_constraint l w ∧ area l w = 35 ∧
  ∀ (l' w' : ℕ), cost_constraint l' w' → area l' w' ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_max_area_is_35_l3255_325535


namespace NUMINAMATH_CALUDE_smallest_percentage_both_drinks_l3255_325519

/-- The percentage of adults who drink coffee -/
def coffee_drinkers : ℝ := 90

/-- The percentage of adults who drink tea -/
def tea_drinkers : ℝ := 85

/-- The smallest possible percentage of adults who drink both coffee and tea -/
def both_drinkers : ℝ := 75

theorem smallest_percentage_both_drinks (coffee_drinkers tea_drinkers both_drinkers : ℝ) 
  (h1 : coffee_drinkers = 90) 
  (h2 : tea_drinkers = 85) : 
  both_drinkers ≥ 75 ∧ ∃ (x : ℝ), x ≥ 75 ∧ 
  coffee_drinkers + tea_drinkers - x ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_percentage_both_drinks_l3255_325519


namespace NUMINAMATH_CALUDE_point_A_on_curve_l3255_325501

/-- The equation of the curve C is x^2 - xy + y - 5 = 0 -/
def curve_equation (x y : ℝ) : Prop := x^2 - x*y + y - 5 = 0

/-- Point A lies on curve C -/
theorem point_A_on_curve : curve_equation (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_on_curve_l3255_325501


namespace NUMINAMATH_CALUDE_total_paintable_area_l3255_325580

/-- Represents a rectangular surface with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a wall with its dimensions and optional window or door -/
structure Wall where
  dimensions : Rectangle
  opening : Option Rectangle

/-- Calculates the paintable area of a wall -/
def Wall.paintableArea (w : Wall) : ℝ :=
  w.dimensions.area - (match w.opening with
    | some o => o.area
    | none => 0)

/-- The four walls of the room -/
def walls : List Wall := [
  { dimensions := { width := 4, height := 8 },
    opening := some { width := 2, height := 3 } },
  { dimensions := { width := 6, height := 8 },
    opening := some { width := 3, height := 6.5 } },
  { dimensions := { width := 4, height := 8 },
    opening := some { width := 3, height := 4 } },
  { dimensions := { width := 6, height := 8 },
    opening := none }
]

theorem total_paintable_area :
  (walls.map Wall.paintableArea).sum = 122.5 := by sorry

end NUMINAMATH_CALUDE_total_paintable_area_l3255_325580


namespace NUMINAMATH_CALUDE_logarithm_properties_l3255_325565

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define lg as log base 10
noncomputable def lg (x : ℝ) : ℝ := log 10 x

theorem logarithm_properties :
  (lg 2 + lg 5 = 1) ∧ (log 3 9 = 2) := by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l3255_325565


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l3255_325584

theorem arithmetic_mean_geq_geometric_mean (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l3255_325584


namespace NUMINAMATH_CALUDE_equal_roots_right_triangle_equilateral_triangle_roots_l3255_325520

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The quadratic equation associated with the triangle -/
def triangle_quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.a - t.c)

theorem equal_roots_right_triangle (t : Triangle) :
  (∃ x : ℝ, (∀ y : ℝ, triangle_quadratic t y = 0 ↔ y = x)) →
  t.a^2 = t.b^2 + t.c^2 :=
sorry

theorem equilateral_triangle_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∀ x : ℝ, triangle_quadratic t x = 0 ↔ x = 0 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equal_roots_right_triangle_equilateral_triangle_roots_l3255_325520


namespace NUMINAMATH_CALUDE_log_equation_solution_l3255_325556

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ log5 x - 4 * log5 2 = -3 ∧ x = 16 / 125 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3255_325556


namespace NUMINAMATH_CALUDE_homothety_composition_l3255_325554

-- Define a homothety
structure Homothety (α : Type*) [AddCommGroup α] :=
  (center : α)
  (coefficient : ℝ)

-- Define a parallel translation
structure ParallelTranslation (α : Type*) [AddCommGroup α] :=
  (vector : α)

-- Define the composition of two homotheties
def compose_homotheties {α : Type*} [AddCommGroup α] [Module ℝ α]
  (h1 h2 : Homothety α) : (ParallelTranslation α) ⊕ (Homothety α) :=
  sorry

-- Theorem statement
theorem homothety_composition {α : Type*} [AddCommGroup α] [Module ℝ α]
  (h1 h2 : Homothety α) :
  (∃ (t : ParallelTranslation α), compose_homotheties h1 h2 = Sum.inl t ∧
    ∃ (v : α), t.vector = v ∧ (∃ (c : ℝ), v = c • (h2.center - h1.center)) ∧
    h1.coefficient * h2.coefficient = 1) ∨
  (∃ (h : Homothety α), compose_homotheties h1 h2 = Sum.inr h ∧
    ∃ (c : ℝ), h.center = h1.center + c • (h2.center - h1.center) ∧
    h.coefficient = h1.coefficient * h2.coefficient ∧
    h1.coefficient * h2.coefficient ≠ 1) :=
  sorry

end NUMINAMATH_CALUDE_homothety_composition_l3255_325554


namespace NUMINAMATH_CALUDE_binary_table_theorem_l3255_325533

/-- Represents a table filled with 0s and 1s -/
def BinaryTable := List (List Bool)

/-- Checks if all rows in the table are unique -/
def allRowsUnique (table : BinaryTable) : Prop :=
  ∀ i j, i ≠ j → table.get! i ≠ table.get! j

/-- Checks if any 4×2 sub-table has two identical rows -/
def anySubTableHasTwoIdenticalRows (table : BinaryTable) : Prop :=
  ∀ c₁ c₂ r₁ r₂ r₃ r₄, 
    c₁ < table.head!.length → c₂ < table.head!.length → c₁ ≠ c₂ →
    r₁ < table.length → r₂ < table.length → r₃ < table.length → r₄ < table.length →
    r₁ ≠ r₂ → r₁ ≠ r₃ → r₁ ≠ r₄ → r₂ ≠ r₃ → r₂ ≠ r₄ → r₃ ≠ r₄ →
    ∃ i j, i ≠ j ∧ 
      (table.get! i).get! c₁ = (table.get! j).get! c₁ ∧
      (table.get! i).get! c₂ = (table.get! j).get! c₂

/-- Checks if a column has exactly one occurrence of a number -/
def columnHasExactlyOneOccurrence (table : BinaryTable) (col : Nat) : Prop :=
  (table.map (λ row => row.get! col)).count true = 1 ∨
  (table.map (λ row => row.get! col)).count false = 1

theorem binary_table_theorem (table : BinaryTable) 
  (h1 : allRowsUnique table)
  (h2 : anySubTableHasTwoIdenticalRows table) :
  ∃ col, columnHasExactlyOneOccurrence table col := by
  sorry


end NUMINAMATH_CALUDE_binary_table_theorem_l3255_325533


namespace NUMINAMATH_CALUDE_room_population_change_l3255_325502

theorem room_population_change (initial_men initial_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  ∃ (current_women : ℕ),
    initial_men + 2 = 14 ∧
    current_women = 2 * (initial_women - 3) ∧
    current_women = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_change_l3255_325502


namespace NUMINAMATH_CALUDE_two_week_riding_time_l3255_325569

-- Define the riding schedule
def riding_schedule : List (String × Float) := [
  ("Monday", 1),
  ("Tuesday", 0.5),
  ("Wednesday", 1),
  ("Thursday", 0.5),
  ("Friday", 1),
  ("Saturday", 2),
  ("Sunday", 0)
]

-- Calculate the total riding time for one week
def weekly_riding_time : Float :=
  (riding_schedule.map (λ (_, time) => time)).sum

-- Theorem: The total riding time for a 2-week period is 12 hours
theorem two_week_riding_time :
  weekly_riding_time * 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_week_riding_time_l3255_325569


namespace NUMINAMATH_CALUDE_f_2017_equals_3_l3255_325534

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_2017_equals_3 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_value : f (-1) = -3) :
  f 2017 = 3 := by
sorry

end NUMINAMATH_CALUDE_f_2017_equals_3_l3255_325534


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l3255_325503

/-- The number of peaches Sally picked up at the orchard -/
def peaches_picked (initial current : ℕ) : ℕ := current - initial

/-- Proof that Sally picked up 42 peaches -/
theorem sally_picked_42_peaches (initial current : ℕ) 
  (h_initial : initial = 13)
  (h_current : current = 55) :
  peaches_picked initial current = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l3255_325503


namespace NUMINAMATH_CALUDE_sequence_term_proof_l3255_325532

/-- Given a sequence where the sum of the first n terms is 5n + 2n^2,
    this function represents the rth term of the sequence. -/
def sequence_term (r : ℕ) : ℕ := 4 * r + 3

/-- The sum of the first n terms of the sequence. -/
def sequence_sum (n : ℕ) : ℕ := 5 * n + 2 * n^2

/-- Theorem stating that the rth term of the sequence is 4r + 3,
    given that the sum of the first n terms is 5n + 2n^2 for all n. -/
theorem sequence_term_proof (r : ℕ) : 
  sequence_term r = sequence_sum r - sequence_sum (r - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l3255_325532


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3255_325596

theorem fractional_equation_solution :
  ∃ x : ℚ, (3 / x = 1 / (x - 1)) ∧ (x = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3255_325596


namespace NUMINAMATH_CALUDE_jills_age_l3255_325524

theorem jills_age (henry_age jill_age : ℕ) : 
  henry_age + jill_age = 43 →
  henry_age - 5 = 2 * (jill_age - 5) →
  jill_age = 16 := by
sorry

end NUMINAMATH_CALUDE_jills_age_l3255_325524
