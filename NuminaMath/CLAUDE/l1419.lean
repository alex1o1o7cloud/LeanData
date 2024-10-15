import Mathlib

namespace NUMINAMATH_CALUDE_spider_plant_babies_l1419_141967

/-- The number of baby plants produced by a spider plant in a given time period -/
def baby_plants (plants_per_time : ℕ) (times_per_year : ℕ) (years : ℕ) : ℕ :=
  plants_per_time * times_per_year * years

/-- Theorem: A spider plant producing 2 baby plants 2 times a year will have 16 baby plants after 4 years -/
theorem spider_plant_babies : baby_plants 2 2 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_spider_plant_babies_l1419_141967


namespace NUMINAMATH_CALUDE_x_minus_y_positive_l1419_141989

theorem x_minus_y_positive (x y a : ℝ) 
  (h1 : x + y > 0) 
  (h2 : a < 0) 
  (h3 : a * y > 0) : 
  x - y > 0 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_positive_l1419_141989


namespace NUMINAMATH_CALUDE_four_pepperoni_slices_left_l1419_141926

/-- Represents the pizza sharing scenario -/
structure PizzaSharing where
  total_people : ℕ
  pepperoni_slices : ℕ
  cheese_slices : ℕ
  cheese_left : ℕ
  pepperoni_only_eaters : ℕ

/-- Calculate the number of pepperoni slices left -/
def pepperoni_left (ps : PizzaSharing) : ℕ :=
  let cheese_eaten := ps.cheese_slices - ps.cheese_left
  let slices_per_person := cheese_eaten / (ps.total_people - ps.pepperoni_only_eaters)
  let pepperoni_eaten := slices_per_person * (ps.total_people - ps.pepperoni_only_eaters) + 
                         slices_per_person * ps.pepperoni_only_eaters
  ps.pepperoni_slices - pepperoni_eaten

/-- Theorem stating that given the conditions, 4 pepperoni slices are left -/
theorem four_pepperoni_slices_left : 
  ∀ (ps : PizzaSharing), 
  ps.total_people = 4 ∧ 
  ps.pepperoni_slices = 16 ∧ 
  ps.cheese_slices = 16 ∧ 
  ps.cheese_left = 7 ∧ 
  ps.pepperoni_only_eaters = 1 →
  pepperoni_left ps = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_pepperoni_slices_left_l1419_141926


namespace NUMINAMATH_CALUDE_no_integer_roots_l1419_141922

theorem no_integer_roots : ¬ ∃ (x : ℤ), x^3 - 3*x^2 - 16*x + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1419_141922


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l1419_141932

/-- Prove that the profit percentage in the previous year was 10% -/
theorem profit_percentage_previous_year
  (R : ℝ) -- Revenue in the previous year
  (h1 : R > 0) -- Assume positive revenue
  (h2 : 0.8 * R = revenue_2009) -- Revenue in 2009 was 80% of previous year
  (h3 : 0.13 * revenue_2009 = profit_2009) -- Profit in 2009 was 13% of 2009 revenue
  (h4 : profit_2009 = 1.04 * profit_previous) -- Profit in 2009 was 104% of previous year's profit
  (h5 : profit_previous = P / 100 * R) -- Definition of profit percentage
  : P = 10 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l1419_141932


namespace NUMINAMATH_CALUDE_iris_spending_l1419_141911

/-- Calculates the total amount spent by Iris on clothes, including discount and tax --/
def total_spent (jacket_price : ℚ) (jacket_quantity : ℕ)
                (shorts_price : ℚ) (shorts_quantity : ℕ)
                (pants_price : ℚ) (pants_quantity : ℕ)
                (tops_price : ℚ) (tops_quantity : ℕ)
                (skirts_price : ℚ) (skirts_quantity : ℕ)
                (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

/-- The theorem stating that Iris spent $230.16 on clothes --/
theorem iris_spending : 
  total_spent 15 3 10 2 18 4 7 6 12 5 (10/100) (7/100) = 230.16 := by
  sorry

end NUMINAMATH_CALUDE_iris_spending_l1419_141911


namespace NUMINAMATH_CALUDE_no_integral_solution_l1419_141978

theorem no_integral_solution : ¬∃ (n m : ℤ), n^2 + (n+1)^2 + (n+2)^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integral_solution_l1419_141978


namespace NUMINAMATH_CALUDE_cake_angle_theorem_l1419_141995

theorem cake_angle_theorem (n : ℕ) (initial_angle : ℝ) (final_angle : ℝ) : 
  n = 10 →
  initial_angle = 360 / n →
  final_angle = 360 / (n - 1) →
  final_angle - initial_angle = 4 := by
  sorry

end NUMINAMATH_CALUDE_cake_angle_theorem_l1419_141995


namespace NUMINAMATH_CALUDE_t_square_four_equal_parts_l1419_141975

/-- A figure composed of three equal squares -/
structure TSquareFigure where
  square_area : ℝ
  total_area : ℝ
  h_total_area : total_area = 3 * square_area

/-- A division of the figure into four parts -/
structure FourPartDivision (fig : TSquareFigure) where
  part_area : ℝ
  h_part_count : ℕ
  h_part_count_eq : h_part_count = 4
  h_total_area : fig.total_area = h_part_count * part_area

/-- Theorem stating that a T-square figure can be divided into four equal parts -/
theorem t_square_four_equal_parts (fig : TSquareFigure) : 
  ∃ (div : FourPartDivision fig), div.part_area = 3 * fig.square_area / 4 := by
  sorry

end NUMINAMATH_CALUDE_t_square_four_equal_parts_l1419_141975


namespace NUMINAMATH_CALUDE_circle_equation_l1419_141977

/-- The standard equation of a circle with center on y = 2x and tangent to x-axis at (-1, 0) -/
theorem circle_equation : ∃ (h k : ℝ), 
  (h = -1 ∧ k = -2) ∧  -- Center on y = 2x
  ((x : ℝ) + 1)^2 + ((y : ℝ) + 2)^2 = 4 ∧  -- Standard equation
  (∀ (x y : ℝ), y = 2*x → (x - h)^2 + (y - k)^2 = 4) ∧  -- Center on y = 2x
  ((-1 : ℝ) - h)^2 + (0 - k)^2 = 4  -- Tangent to x-axis at (-1, 0)
  := by sorry

end NUMINAMATH_CALUDE_circle_equation_l1419_141977


namespace NUMINAMATH_CALUDE_rectangle_area_is_75_l1419_141904

/-- Represents a rectangle with length and breadth -/
structure Rectangle where
  length : ℝ
  breadth : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.breadth)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- Theorem: A rectangle with length thrice its breadth and perimeter 40 has an area of 75 -/
theorem rectangle_area_is_75 (r : Rectangle) 
    (h1 : r.length = 3 * r.breadth) 
    (h2 : perimeter r = 40) : 
  area r = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_75_l1419_141904


namespace NUMINAMATH_CALUDE_grassy_width_is_60_l1419_141952

/-- Represents a rectangular plot with a gravel path around it. -/
structure RectangularPlot where
  length : ℝ
  totalWidth : ℝ
  pathWidth : ℝ

/-- Calculates the width of the grassy area in a rectangular plot. -/
def grassyWidth (plot : RectangularPlot) : ℝ :=
  plot.totalWidth - 2 * plot.pathWidth

/-- Theorem stating that for a given rectangular plot with specified dimensions,
    the width of the grassy area is 60 meters. -/
theorem grassy_width_is_60 (plot : RectangularPlot)
    (h1 : plot.length = 110)
    (h2 : plot.totalWidth = 65)
    (h3 : plot.pathWidth = 2.5) :
  grassyWidth plot = 60 := by
  sorry

end NUMINAMATH_CALUDE_grassy_width_is_60_l1419_141952


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1419_141907

theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) : 
  x = 3 + 2^p → y = 3 + 2^(-p) → y = (3*x - 8) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1419_141907


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1419_141951

theorem floor_ceil_sum : ⌊(0.998 : ℝ)⌋ + ⌈(2.002 : ℝ)⌉ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1419_141951


namespace NUMINAMATH_CALUDE_freshmen_liberal_arts_percentage_l1419_141931

theorem freshmen_liberal_arts_percentage 
  (total_students : ℝ) 
  (freshmen_percentage : ℝ) 
  (liberal_arts_freshmen_percentage : ℝ) 
  (psychology_majors_percentage : ℝ) 
  (freshmen_psychology_liberal_arts_percentage : ℝ) 
  (h1 : freshmen_percentage = 0.5)
  (h2 : psychology_majors_percentage = 0.2)
  (h3 : freshmen_psychology_liberal_arts_percentage = 0.04)
  (h4 : freshmen_psychology_liberal_arts_percentage * total_students = 
        psychology_majors_percentage * liberal_arts_freshmen_percentage * freshmen_percentage * total_students) :
  liberal_arts_freshmen_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_freshmen_liberal_arts_percentage_l1419_141931


namespace NUMINAMATH_CALUDE_percentage_relation_l1419_141965

theorem percentage_relation (A B x : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : A = (x / 100) * B) : 
  x = 100 * (A / B) := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1419_141965


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1419_141945

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where a₃ + a₅ = 10, prove that a₄ = 5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 10) : 
  a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1419_141945


namespace NUMINAMATH_CALUDE_inscribed_circle_height_difference_l1419_141901

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x

/-- Represents a circle inscribed in the parabola -/
structure InscribedCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangentPoint : ℝ
  isTangent : (tangentPoint, parabola tangentPoint) ∈ frontier {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

/-- The height difference between the circle's center and a tangent point -/
def heightDifference (circle : InscribedCircle) : ℝ :=
  circle.center.2 - parabola circle.tangentPoint

theorem inscribed_circle_height_difference (circle : InscribedCircle) :
  heightDifference circle = -2 * circle.tangentPoint^2 - 4 * circle.tangentPoint + 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_height_difference_l1419_141901


namespace NUMINAMATH_CALUDE_square_with_hole_l1419_141953

theorem square_with_hole (n m : ℕ) (h1 : n^2 - m^2 = 209) (h2 : n > m) : n^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_square_with_hole_l1419_141953


namespace NUMINAMATH_CALUDE_range_of_x_l1419_141979

theorem range_of_x (x : ℝ) : 
  (6 - 3 * x ≥ 0) ∧ (1 / (x + 1) ≥ 0) → x ∈ Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1419_141979


namespace NUMINAMATH_CALUDE_smallest_number_less_than_negative_one_l1419_141963

theorem smallest_number_less_than_negative_one :
  let numbers : List ℝ := [-1/2, 0, |(-2)|, -3]
  ∀ x ∈ numbers, x < -1 ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_less_than_negative_one_l1419_141963


namespace NUMINAMATH_CALUDE_car_speed_l1419_141928

/-- Given a car traveling 810 km in 5 hours, its speed is 162 km/h -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 810) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : 
  speed = 162 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l1419_141928


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1419_141929

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1419_141929


namespace NUMINAMATH_CALUDE_deer_weight_l1419_141996

/-- Calculates the weight of each deer given hunting frequency, season length, and total kept weight --/
theorem deer_weight 
  (hunts_per_month : ℕ)
  (season_fraction : ℚ)
  (deer_per_hunt : ℕ)
  (kept_fraction : ℚ)
  (total_kept_weight : ℕ)
  (h1 : hunts_per_month = 6)
  (h2 : season_fraction = 1/4)
  (h3 : deer_per_hunt = 2)
  (h4 : kept_fraction = 1/2)
  (h5 : total_kept_weight = 10800) :
  (total_kept_weight / kept_fraction) / (hunts_per_month * (season_fraction * 12) * deer_per_hunt) = 600 := by
  sorry

end NUMINAMATH_CALUDE_deer_weight_l1419_141996


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l1419_141944

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_prime_factorization :
  ∃ (i k m p q : ℕ+),
    factorial 12 = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) * 11^(q.val) ∧
    i.val + k.val + m.val + p.val + q.val = 28 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l1419_141944


namespace NUMINAMATH_CALUDE_circles_tangent_implies_m_equals_four_l1419_141976

-- Define the circles
def circle_C (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 5 - m}
def circle_E : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Define the condition for external tangency
def externally_tangent (C E : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ E ∧ 
  ∀ (q : ℝ × ℝ), q ∈ C ∩ E → q = p

-- State the theorem
theorem circles_tangent_implies_m_equals_four :
  ∀ (m : ℝ), externally_tangent (circle_C m) circle_E → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_m_equals_four_l1419_141976


namespace NUMINAMATH_CALUDE_rain_probability_l1419_141913

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end NUMINAMATH_CALUDE_rain_probability_l1419_141913


namespace NUMINAMATH_CALUDE_purse_cost_multiple_l1419_141917

theorem purse_cost_multiple (wallet_cost purse_cost : ℚ) : 
  wallet_cost = 22 →
  wallet_cost + purse_cost = 107 →
  ∃ n : ℕ, n ≤ 4 ∧ purse_cost < n * wallet_cost :=
by sorry

end NUMINAMATH_CALUDE_purse_cost_multiple_l1419_141917


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_4_eq_one_third_l1419_141933

theorem tan_x_minus_pi_4_eq_one_third (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.cos (2 * x - Real.pi / 2) = Real.sin x ^ 2) : 
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_4_eq_one_third_l1419_141933


namespace NUMINAMATH_CALUDE_marathon_yards_remainder_l1419_141930

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Represents a total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def marathon : MarathonDistance :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

/-- Calculates the total distance for a given number of marathons -/
def total_distance (n : ℕ) (d : MarathonDistance) : TotalDistance :=
  { miles := n * d.miles,
    yards := n * d.yards }

/-- Converts excess yards to miles and updates the TotalDistance -/
def normalize_distance (d : TotalDistance) : TotalDistance :=
  { miles := d.miles + d.yards / yards_per_mile,
    yards := d.yards % yards_per_mile }

theorem marathon_yards_remainder :
  (normalize_distance (total_distance num_marathons marathon)).yards = 495 := by
  sorry

end NUMINAMATH_CALUDE_marathon_yards_remainder_l1419_141930


namespace NUMINAMATH_CALUDE_product_calculation_l1419_141918

theorem product_calculation : 3.5 * 7.2 * (6.3 - 1.4) = 122.5 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l1419_141918


namespace NUMINAMATH_CALUDE_mersenne_prime_condition_l1419_141935

theorem mersenne_prime_condition (a b : ℕ) (h1 : a ≥ 1) (h2 : b ≥ 2) 
  (h3 : Nat.Prime (a^b - 1)) : a = 2 ∧ Nat.Prime b := by
  sorry

end NUMINAMATH_CALUDE_mersenne_prime_condition_l1419_141935


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l1419_141991

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 5

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select 2 shoes out of the total -/
def total_selections : ℕ := (total_shoes.choose 2)

/-- The number of ways to select a matching pair -/
def matching_selections : ℕ := num_pairs

/-- The probability of selecting a matching pair -/
def probability_matching : ℚ := matching_selections / total_selections

theorem matching_shoes_probability : probability_matching = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l1419_141991


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1419_141973

/-- The value of a repeating decimal 0.abcabcabc... where a, b, c are digits -/
def repeating_decimal (a b c : Nat) : ℚ := (100 * a + 10 * b + c : ℚ) / 999

/-- Theorem stating that 0.246246246... - 0.135135135... - 0.012012012... = 1/9 -/
theorem repeating_decimal_subtraction :
  repeating_decimal 2 4 6 - repeating_decimal 1 3 5 - repeating_decimal 0 1 2 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1419_141973


namespace NUMINAMATH_CALUDE_product_a4b4_equals_negative_six_l1419_141900

theorem product_a4b4_equals_negative_six
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ)
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_product_a4b4_equals_negative_six_l1419_141900


namespace NUMINAMATH_CALUDE_birch_tree_probability_l1419_141988

/-- The probability of no two birch trees being adjacent when planting trees in a row -/
theorem birch_tree_probability (maple oak birch : ℕ) (h1 : maple = 4) (h2 : oak = 3) (h3 : birch = 6) :
  let total := maple + oak + birch
  let non_birch := maple + oak
  let favorable := Nat.choose (non_birch + 1) birch
  let total_arrangements := Nat.choose total birch
  (favorable : ℚ) / total_arrangements = 7 / 429 :=
by sorry

end NUMINAMATH_CALUDE_birch_tree_probability_l1419_141988


namespace NUMINAMATH_CALUDE_sean_houses_bought_l1419_141943

theorem sean_houses_bought (initial_houses : ℕ) (traded_houses : ℕ) (final_houses : ℕ) 
  (h1 : initial_houses = 27)
  (h2 : traded_houses = 8)
  (h3 : final_houses = 31) :
  final_houses - (initial_houses - traded_houses) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sean_houses_bought_l1419_141943


namespace NUMINAMATH_CALUDE_jason_egg_consumption_l1419_141984

/-- The number of eggs Jason uses for one omelet -/
def eggs_per_omelet : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks we're considering -/
def weeks_considered : ℕ := 2

/-- Theorem: Jason consumes 42 eggs in two weeks -/
theorem jason_egg_consumption :
  eggs_per_omelet * days_per_week * weeks_considered = 42 := by
  sorry

end NUMINAMATH_CALUDE_jason_egg_consumption_l1419_141984


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1419_141910

theorem min_value_quadratic (x : ℝ) :
  ∃ (z_min : ℝ), ∀ (z : ℝ), z = 3 * x^2 + 18 * x + 11 → z ≥ z_min ∧ ∃ (x_min : ℝ), 3 * x_min^2 + 18 * x_min + 11 = z_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1419_141910


namespace NUMINAMATH_CALUDE_updated_mean_after_correction_l1419_141994

theorem updated_mean_after_correction (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n = 100 →
  original_mean = 350 →
  decrement = 63 →
  (n : ℝ) * original_mean + n * decrement = n * 413 := by
sorry

end NUMINAMATH_CALUDE_updated_mean_after_correction_l1419_141994


namespace NUMINAMATH_CALUDE_negation_equivalence_l1419_141966

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 0, x^2 ≤ |x|) ↔ (∀ x ∈ Set.Ioo (-1 : ℝ) 0, x^2 > |x|) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1419_141966


namespace NUMINAMATH_CALUDE_real_part_of_z_l1419_141964

theorem real_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.abs (3 - 4 * Complex.I)) : 
  z.re = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1419_141964


namespace NUMINAMATH_CALUDE_roots_properties_l1419_141980

def equation_roots (b : ℝ) (θ : ℝ) : Prop :=
  169 * (Real.sin θ)^2 - b * (Real.sin θ) + 60 = 0 ∧
  169 * (Real.cos θ)^2 - b * (Real.cos θ) + 60 = 0

theorem roots_properties (θ : ℝ) (h : π/4 < θ ∧ θ < 3*π/4) :
  ∃ b : ℝ, equation_roots b θ ∧
    b = 221 ∧
    (Real.sin θ / (1 - Real.cos θ)) + ((1 + Real.cos θ) / Real.sin θ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_properties_l1419_141980


namespace NUMINAMATH_CALUDE_star_four_three_l1419_141986

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2

-- Theorem statement
theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l1419_141986


namespace NUMINAMATH_CALUDE_max_sundays_in_53_days_l1419_141956

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we're considering -/
def total_days : ℕ := 53

/-- A function that returns the number of Sundays in a given number of days -/
def sundays_in_days (days : ℕ) : ℕ := days / days_in_week

theorem max_sundays_in_53_days : 
  sundays_in_days total_days = 7 := by sorry

end NUMINAMATH_CALUDE_max_sundays_in_53_days_l1419_141956


namespace NUMINAMATH_CALUDE_derivative_negative_two_exp_times_sin_l1419_141905

theorem derivative_negative_two_exp_times_sin (x : ℝ) :
  deriv (λ x => -2 * Real.exp x * Real.sin x) x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_negative_two_exp_times_sin_l1419_141905


namespace NUMINAMATH_CALUDE_no_pentagon_cross_section_l1419_141987

/-- A cube in 3D space -/
structure Cube

/-- A plane in 3D space -/
structure Plane

/-- Possible shapes that can result from the intersection of a plane and a cube -/
inductive CrossSection
  | EquilateralTriangle
  | Square
  | RegularPentagon
  | RegularHexagon

/-- The intersection of a plane and a cube -/
def intersect (c : Cube) (p : Plane) : CrossSection := sorry

/-- Theorem stating that a regular pentagon cannot be a cross-section of a cube -/
theorem no_pentagon_cross_section (c : Cube) (p : Plane) :
  intersect c p ≠ CrossSection.RegularPentagon := by sorry

end NUMINAMATH_CALUDE_no_pentagon_cross_section_l1419_141987


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l1419_141961

def third_smallest_prime : ℕ := 5

theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l1419_141961


namespace NUMINAMATH_CALUDE_inequality_solution_l1419_141959

theorem inequality_solution (x : ℤ) : 
  Real.sqrt (3 * x - 7) - Real.sqrt (3 * x^2 - 13 * x + 13) ≥ 3 * x^2 - 16 * x + 20 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1419_141959


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1419_141915

theorem complex_product_magnitude (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1419_141915


namespace NUMINAMATH_CALUDE_misread_weight_l1419_141962

theorem misread_weight (n : ℕ) (initial_avg correct_avg correct_weight : ℝ) :
  n = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.9 →
  correct_weight = 66 →
  ∃ (misread_weight : ℝ),
    n * initial_avg + (correct_weight - misread_weight) = n * correct_avg ∧
    misread_weight = 56 :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_l1419_141962


namespace NUMINAMATH_CALUDE_cube_root_over_fifth_root_of_five_l1419_141920

theorem cube_root_over_fifth_root_of_five (x : ℝ) (hx : x > 0) :
  (x^(1/3)) / (x^(1/5)) = x^(2/15) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_over_fifth_root_of_five_l1419_141920


namespace NUMINAMATH_CALUDE_christmas_gifts_theorem_l1419_141940

/-- The number of gifts left under the Christmas tree -/
def gifts_left (initial_gifts additional_gifts gifts_sent : ℕ) : ℕ :=
  initial_gifts + additional_gifts - gifts_sent

/-- Theorem: Given the initial gifts, additional gifts, and gifts sent,
    prove that the number of gifts left under the tree is 44. -/
theorem christmas_gifts_theorem :
  gifts_left 77 33 66 = 44 := by
  sorry

end NUMINAMATH_CALUDE_christmas_gifts_theorem_l1419_141940


namespace NUMINAMATH_CALUDE_ellipse_left_vertex_l1419_141903

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The theorem stating the properties of the ellipse and its left vertex -/
theorem ellipse_left_vertex 
  (E : Ellipse) 
  (C : Circle) 
  (h_focus : C.h = 3 ∧ C.k = 0) -- One focus is the center of the circle
  (h_circle_eq : ∀ x y, x^2 + y^2 - 6*x + 8 = 0 ↔ (x - C.h)^2 + (y - C.k)^2 = C.r^2) -- Circle equation
  (h_minor_axis : E.b = 4) -- Minor axis is 8 in length
  : ∃ x, x = -5 ∧ (x / E.a)^2 + 0^2 / E.b^2 = 1 -- Left vertex is at (-5, 0)
:= by sorry

end NUMINAMATH_CALUDE_ellipse_left_vertex_l1419_141903


namespace NUMINAMATH_CALUDE_boat_problem_l1419_141925

theorem boat_problem (total_students : ℕ) (big_boat_capacity small_boat_capacity : ℕ) (total_boats : ℕ) :
  total_students = 52 →
  big_boat_capacity = 8 →
  small_boat_capacity = 4 →
  total_boats = 9 →
  ∃ (big_boats small_boats : ℕ),
    big_boats + small_boats = total_boats ∧
    big_boats * big_boat_capacity + small_boats * small_boat_capacity = total_students ∧
    big_boats = 4 :=
by sorry

end NUMINAMATH_CALUDE_boat_problem_l1419_141925


namespace NUMINAMATH_CALUDE_inequality_solution_l1419_141902

theorem inequality_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x - 1) / 2 + 1 ≥ 2 * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1419_141902


namespace NUMINAMATH_CALUDE_sharon_oranges_l1419_141923

theorem sharon_oranges (janet_oranges total_oranges : ℕ) 
  (h1 : janet_oranges = 9) 
  (h2 : total_oranges = 16) : 
  total_oranges - janet_oranges = 7 := by
  sorry

end NUMINAMATH_CALUDE_sharon_oranges_l1419_141923


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_dot_product_l1419_141960

-- Define the hyperbola C'
def hyperbola_C' (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Define the dot product of AP and BP
def AP_dot_BP (x y : ℝ) : ℝ :=
  (x + 1) * (x - 1) + y * y

-- Define the range of AP⋅BP
def range_AP_dot_BP : Set ℝ :=
  {z : ℝ | 191/34 ≤ z ∧ z ≤ 24}

-- Theorem statement
theorem hyperbola_ellipse_dot_product :
  -- Conditions
  (∀ x y : ℝ, 3*x = 4*y ∨ 3*x = -4*y → ¬(hyperbola_C' x y)) →  -- Asymptotes
  (hyperbola_C' 5 (9/4)) →  -- Hyperbola passes through (5, 9/4)
  (∃ x₀ : ℝ, x₀ > 0 ∧ ellipse_M x₀ 0 ∧ hyperbola_C' x₀ 0) →  -- Shared focus/vertex
  (∀ x y : ℝ, ellipse_M x y → x ≤ 5 ∧ y ≤ 3) →  -- Bounds on ellipse
  -- Conclusions
  (∀ x y : ℝ, hyperbola_C' x y ↔ x^2 / 16 - y^2 / 9 = 1) ∧
  (∀ x y : ℝ, ellipse_M x y ↔ x^2 / 25 + y^2 / 9 = 1) ∧
  (∀ x y : ℝ, ellipse_M x y ∧ x ≥ 0 → AP_dot_BP x y ∈ range_AP_dot_BP) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_dot_product_l1419_141960


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1419_141969

/-- A hyperbola with foci on the x-axis, passing through (4√2, -3), and having perpendicular lines
    connecting (0, 5) to its foci, has the standard equation x²/16 - y²/9 = 1. -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c^2 = a^2 + b^2) : 
  (32 / a^2 - 9 / b^2 = 1) → (25 / c^2 = 1) → (a = 4 ∧ b = 3) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l1419_141969


namespace NUMINAMATH_CALUDE_at_least_eight_nonzero_digits_l1419_141941

/-- Given a natural number n, returns a number consisting of n repeating 9's -/
def repeating_nines (n : ℕ) : ℕ := 10^n - 1

/-- Counts the number of non-zero digits in the decimal representation of a natural number -/
def count_nonzero_digits (k : ℕ) : ℕ := sorry

theorem at_least_eight_nonzero_digits 
  (k : ℕ) (n : ℕ) (h1 : k > 0) (h2 : k % repeating_nines n = 0) : 
  count_nonzero_digits k ≥ 8 := by sorry

end NUMINAMATH_CALUDE_at_least_eight_nonzero_digits_l1419_141941


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1419_141992

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1419_141992


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l1419_141914

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧  -- Three different integer side lengths
  a + b + c = 24 ∧ -- Perimeter is 24 units
  a + b > c →      -- Triangle inequality
  c ≤ 10 :=        -- Maximum length of any side is 10
by sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l1419_141914


namespace NUMINAMATH_CALUDE_cos_inequality_range_l1419_141970

theorem cos_inequality_range (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) := by
sorry

end NUMINAMATH_CALUDE_cos_inequality_range_l1419_141970


namespace NUMINAMATH_CALUDE_bankers_discount_example_l1419_141921

/-- Given a bill with face value and true discount, calculates the banker's discount -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  true_discount + (true_discount^2 / present_value)

/-- Theorem stating that for a bill with face value 540 and true discount 90, 
    the banker's discount is 108 -/
theorem bankers_discount_example : bankers_discount 540 90 = 108 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_example_l1419_141921


namespace NUMINAMATH_CALUDE_product_of_largest_primes_eq_679679_l1419_141993

/-- The largest one-digit prime number -/
def largest_one_digit_prime : ℕ := 7

/-- The largest two-digit prime number -/
def largest_two_digit_prime : ℕ := 97

/-- The largest three-digit prime number -/
def largest_three_digit_prime : ℕ := 997

/-- The product of the largest one-digit, two-digit, and three-digit primes -/
def product_of_largest_primes : ℕ := 
  largest_one_digit_prime * largest_two_digit_prime * largest_three_digit_prime

theorem product_of_largest_primes_eq_679679 : 
  product_of_largest_primes = 679679 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_eq_679679_l1419_141993


namespace NUMINAMATH_CALUDE_min_positive_translation_for_symmetry_l1419_141981

open Real

theorem min_positive_translation_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = sin (2 * x + π / 4)) →
  (∀ x, f (x - φ) = f (-x)) →
  (φ > 0) →
  (∀ ψ, ψ > 0 → ψ < φ → ¬(∀ x, f (x - ψ) = f (-x))) →
  φ = 3 * π / 8 := by
sorry

end NUMINAMATH_CALUDE_min_positive_translation_for_symmetry_l1419_141981


namespace NUMINAMATH_CALUDE_consecutive_years_product_l1419_141972

theorem consecutive_years_product : (2014 - 2013) * (2013 - 2012) = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_years_product_l1419_141972


namespace NUMINAMATH_CALUDE_tangent_slope_condition_l1419_141968

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

-- State the theorem
theorem tangent_slope_condition (a b : ℝ) :
  (∀ x, (deriv (f a b)) x = 2 * a * x) →  -- Derivative of f
  (deriv (f a b)) 1 = 2 →  -- Slope of tangent line at x = 1 is 2
  f a b 1 = 3 →  -- Function value at x = 1 is 3
  b / a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_condition_l1419_141968


namespace NUMINAMATH_CALUDE_complex_number_problem_l1419_141924

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2 * Complex.I = r)
  (h2 : ∃ (m : ℝ), z / (2 - Complex.I) = m) : 
  z = 4 - 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1419_141924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1419_141954

/-- Given an arithmetic sequence {a_n} where a_5/a_3 = 5/9, prove that S_9/S_5 = 1 -/
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (h : a 5 / a 3 = 5 / 9) :
  let S : ℕ → ℝ := λ n => (n / 2) * (a 1 + a n)
  S 9 / S 5 = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1419_141954


namespace NUMINAMATH_CALUDE_AB_range_l1419_141985

/-- Represents an acute triangle ABC with specific properties -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  acute : A < 90 ∧ B < 90 ∧ C < 90
  angle_sum : A + B + C = 180
  angle_A : A = 60
  side_BC : BC = 6
  side_AB : AB > 0

/-- Theorem stating the range of AB in the specific acute triangle -/
theorem AB_range (t : AcuteTriangle) : 2 * Real.sqrt 3 < t.AB ∧ t.AB < 4 * Real.sqrt 3 := by
  sorry

#check AB_range

end NUMINAMATH_CALUDE_AB_range_l1419_141985


namespace NUMINAMATH_CALUDE_coffee_mix_theorem_l1419_141908

/-- Calculates the price per pound of a coffee mix given the prices and quantities of two types of coffee. -/
def coffee_mix_price (price1 price2 : ℚ) (quantity1 quantity2 : ℚ) : ℚ :=
  (price1 * quantity1 + price2 * quantity2) / (quantity1 + quantity2)

/-- Theorem stating that mixing equal quantities of two types of coffee priced at $2.15 and $2.45 per pound
    results in a mix priced at $2.30 per pound. -/
theorem coffee_mix_theorem :
  let price1 : ℚ := 215 / 100
  let price2 : ℚ := 245 / 100
  let quantity1 : ℚ := 9
  let quantity2 : ℚ := 9
  coffee_mix_price price1 price2 quantity1 quantity2 = 230 / 100 := by
  sorry

#eval coffee_mix_price (215/100) (245/100) 9 9

end NUMINAMATH_CALUDE_coffee_mix_theorem_l1419_141908


namespace NUMINAMATH_CALUDE_max_bananas_is_7_l1419_141912

def budget : ℕ := 10
def single_banana_cost : ℕ := 2
def bundle_4_cost : ℕ := 6
def bundle_6_cost : ℕ := 8

def max_bananas (b s b4 b6 : ℕ) : ℕ := 
  sorry

theorem max_bananas_is_7 : 
  max_bananas budget single_banana_cost bundle_4_cost bundle_6_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_bananas_is_7_l1419_141912


namespace NUMINAMATH_CALUDE_circles_intersect_l1419_141919

/-- Two circles in a 2D plane --/
structure TwoCircles where
  /-- First circle: (x-1)^2 + y^2 = 1 --/
  c1 : (ℝ × ℝ) → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 1
  /-- Second circle: x^2 + y^2 + 2x + 4y - 4 = 0 --/
  c2 : (ℝ × ℝ) → Prop := fun p => p.1^2 + p.2^2 + 2*p.1 + 4*p.2 - 4 = 0

/-- The two circles intersect --/
theorem circles_intersect (tc : TwoCircles) : ∃ p : ℝ × ℝ, tc.c1 p ∧ tc.c2 p := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1419_141919


namespace NUMINAMATH_CALUDE_line_intersects_at_least_one_l1419_141990

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of being skew lines
variable (skew : Line → Line → Prop)

-- Define the property of intersection
variable (intersects : Line → Line → Prop)

-- Define the property of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- State the theorem
theorem line_intersects_at_least_one 
  (m n l : Line) (α β : Plane) :
  skew m n →
  ¬(intersects l m) →
  ¬(intersects l n) →
  in_plane n β →
  plane_intersection α β = l →
  (intersects l m) ∨ (intersects l n) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_at_least_one_l1419_141990


namespace NUMINAMATH_CALUDE_round_310242_to_nearest_thousand_l1419_141949

def round_to_nearest_thousand (n : ℕ) : ℕ :=
  (n + 500) / 1000 * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 := by
  sorry

end NUMINAMATH_CALUDE_round_310242_to_nearest_thousand_l1419_141949


namespace NUMINAMATH_CALUDE_inheritance_problem_l1419_141937

theorem inheritance_problem (S₁ S₂ S₃ S₄ D N : ℕ) :
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ D > 0 ∧ N > 0 →
  Nat.sqrt S₁ = S₂ / 2 →
  Nat.sqrt S₁ = S₃ - 2 →
  Nat.sqrt S₁ = S₄ + 2 →
  Nat.sqrt S₁ = 2 * D →
  Nat.sqrt S₁ = N * N →
  S₁ + S₂ + S₃ + S₄ + D + N < 1500 →
  S₁ + S₂ + S₃ + S₄ + D + N = 1464 :=
by sorry

#eval Nat.sqrt 1296  -- Should output 36
#eval 72 / 2         -- Should output 36
#eval 38 - 2         -- Should output 36
#eval 34 + 2         -- Should output 36
#eval 2 * 18         -- Should output 36
#eval 6 * 6          -- Should output 36
#eval 1296 + 72 + 38 + 34 + 18 + 6  -- Should output 1464

end NUMINAMATH_CALUDE_inheritance_problem_l1419_141937


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l1419_141939

theorem candy_mixture_cost (first_candy_weight : ℝ) (second_candy_weight : ℝ) 
  (second_candy_price : ℝ) (mixture_price : ℝ) :
  first_candy_weight = 20 →
  second_candy_weight = 80 →
  second_candy_price = 5 →
  mixture_price = 6 →
  first_candy_weight + second_candy_weight = 100 →
  ∃ (first_candy_price : ℝ),
    first_candy_price * first_candy_weight + 
    second_candy_price * second_candy_weight = 
    mixture_price * (first_candy_weight + second_candy_weight) ∧
    first_candy_price = 10 := by
  sorry


end NUMINAMATH_CALUDE_candy_mixture_cost_l1419_141939


namespace NUMINAMATH_CALUDE_special_triangle_existence_l1419_141946

/-- A triangle with integer side lengths satisfying a special condition -/
def SpecialTriangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive integers
  a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
  a * b * c = 2 * (a - 1) * (b - 1) * (c - 1)  -- Special condition

theorem special_triangle_existence :
  ∃ a b c : ℕ, SpecialTriangle a b c ∧
  (∀ x y z : ℕ, SpecialTriangle x y z → (x, y, z) = (8, 7, 3) ∨ (x, y, z) = (6, 5, 4)) :=
by sorry


end NUMINAMATH_CALUDE_special_triangle_existence_l1419_141946


namespace NUMINAMATH_CALUDE_jenny_mother_age_problem_l1419_141950

/-- Given that Jenny is 10 years old in 2010 and her mother's age is five times Jenny's age,
    prove that the year when Jenny's mother's age will be twice Jenny's age is 2040. -/
theorem jenny_mother_age_problem (jenny_age_2010 : ℕ) (mother_age_2010 : ℕ) :
  jenny_age_2010 = 10 →
  mother_age_2010 = 5 * jenny_age_2010 →
  ∃ (years_after_2010 : ℕ),
    mother_age_2010 + years_after_2010 = 2 * (jenny_age_2010 + years_after_2010) ∧
    2010 + years_after_2010 = 2040 :=
by sorry

end NUMINAMATH_CALUDE_jenny_mother_age_problem_l1419_141950


namespace NUMINAMATH_CALUDE_a_perpendicular_b_l1419_141936

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Vector a in ℝ² -/
def a : ℝ × ℝ := (3, 4)

/-- Vector b in ℝ² -/
def b : ℝ × ℝ := (-8, 6)

/-- Theorem: Vectors a and b are perpendicular -/
theorem a_perpendicular_b : perpendicular a b := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_b_l1419_141936


namespace NUMINAMATH_CALUDE_smallest_sum_mn_l1419_141974

theorem smallest_sum_mn (m n : ℕ) (hm : m > n) (h_div : (70^2 : ℕ) ∣ (2023^m - 2023^n)) : m + n ≥ 24 ∧ ∃ (m₀ n₀ : ℕ), m₀ + n₀ = 24 ∧ m₀ > n₀ ∧ (70^2 : ℕ) ∣ (2023^m₀ - 2023^n₀) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_mn_l1419_141974


namespace NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l1419_141983

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l1419_141983


namespace NUMINAMATH_CALUDE_hockey_players_count_l1419_141934

theorem hockey_players_count (cricket football softball total : ℕ) 
  (h_cricket : cricket = 16)
  (h_football : football = 18)
  (h_softball : softball = 13)
  (h_total : total = 59)
  : total - (cricket + football + softball) = 12 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l1419_141934


namespace NUMINAMATH_CALUDE_height_of_A_l1419_141971

/-- The heights of four people A, B, C, and D satisfying certain conditions -/
structure Heights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_equality : A + B = C + D ∨ A + C = B + D ∨ A + D = B + C
  average_difference : (A + B) / 2 = (A + C) / 2 + 4
  D_taller : D = A + 10
  B_C_sum : B + C = 288

/-- The height of A is 139 cm -/
theorem height_of_A (h : Heights) : h.A = 139 := by
  sorry

end NUMINAMATH_CALUDE_height_of_A_l1419_141971


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1419_141906

/-- For a parabola y^2 = 2x, the x-coordinate of a point on the parabola
    that is at a distance of 3 from its focus is 5/2. -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = 2*x →  -- parabola equation
  (x + 1/2)^2 + y^2 = 3^2 →  -- distance from focus is 3
  x = 5/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1419_141906


namespace NUMINAMATH_CALUDE_document_word_count_l1419_141909

/-- Calculates the approximate total number of words in a document -/
def approx_total_words (num_pages : ℕ) (avg_words_per_page : ℕ) : ℕ :=
  num_pages * avg_words_per_page

/-- Theorem stating that a document with 8 pages and an average of 605 words per page has approximately 4800 words in total -/
theorem document_word_count : approx_total_words 8 605 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_document_word_count_l1419_141909


namespace NUMINAMATH_CALUDE_sequence_general_term_l1419_141938

theorem sequence_general_term (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h : ∀ k, S k = 2 * k^2 - 3 * k) : 
  a n = 4 * n - 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1419_141938


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1419_141955

theorem arithmetic_geometric_mean_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧
  ((x + y) / 2 = Real.sqrt (x * y) ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1419_141955


namespace NUMINAMATH_CALUDE_lapis_share_is_correct_l1419_141957

/-- Represents the share of treasure for a person -/
structure TreasureShare where
  amount : ℚ
  deriving Repr

/-- Calculates the share of treasure based on contribution -/
def calculateShare (contribution : ℚ) (totalContribution : ℚ) (treasureValue : ℚ) : TreasureShare :=
  { amount := (contribution / totalContribution) * treasureValue }

theorem lapis_share_is_correct (fonzie_contribution : ℚ) (aunt_bee_contribution : ℚ) (lapis_contribution : ℚ) (treasure_value : ℚ)
    (h1 : fonzie_contribution = 7000)
    (h2 : aunt_bee_contribution = 8000)
    (h3 : lapis_contribution = 9000)
    (h4 : treasure_value = 900000) :
  (calculateShare lapis_contribution (fonzie_contribution + aunt_bee_contribution + lapis_contribution) treasure_value).amount = 337500 := by
  sorry

#eval calculateShare 9000 24000 900000

end NUMINAMATH_CALUDE_lapis_share_is_correct_l1419_141957


namespace NUMINAMATH_CALUDE_limit_Sn_divided_by_n2Bn_l1419_141942

-- Define the set A
def A (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define B_n as the number of subsets of A
def B_n (n : ℕ) : ℕ := 2^n

-- Define S_n as the sum of elements in non-empty proper subsets of A
def S_n (n : ℕ) : ℕ := (n * (n + 1) / 2) * (2^(n - 1) - 1)

-- State the theorem
theorem limit_Sn_divided_by_n2Bn (ε : ℝ) (ε_pos : ε > 0) :
  ∃ N : ℕ, ∀ n ≥ N, |S_n n / (n^2 * B_n n : ℝ) - 1/4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_Sn_divided_by_n2Bn_l1419_141942


namespace NUMINAMATH_CALUDE_divides_power_minus_constant_l1419_141999

theorem divides_power_minus_constant (n : ℕ) : 13 ∣ 14^n - 27 := by
  sorry

end NUMINAMATH_CALUDE_divides_power_minus_constant_l1419_141999


namespace NUMINAMATH_CALUDE_domain_of_z_l1419_141948

def z (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x + 1) ^ (1/2)

theorem domain_of_z : 
  {x : ℝ | ∃ y, z x = y} = {x : ℝ | x ≥ 5} :=
sorry

end NUMINAMATH_CALUDE_domain_of_z_l1419_141948


namespace NUMINAMATH_CALUDE_min_sum_factors_2400_l1419_141916

/-- The minimum sum of two positive integer factors of 2400 -/
theorem min_sum_factors_2400 : ∀ a b : ℕ+, a * b = 2400 → (∀ c d : ℕ+, c * d = 2400 → a + b ≤ c + d) → a + b = 98 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_factors_2400_l1419_141916


namespace NUMINAMATH_CALUDE_luke_coin_count_l1419_141997

theorem luke_coin_count (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 5)
  (h2 : dime_piles = 5)
  (h3 : coins_per_pile = 3) : 
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l1419_141997


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l1419_141947

theorem solution_set_of_inequalities :
  let S := {x : ℝ | x - 2 > 1 ∧ x < 4}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l1419_141947


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1419_141958

theorem polynomial_remainder (p : ℤ) : (p^11 - 3) % (p - 2) = 2045 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1419_141958


namespace NUMINAMATH_CALUDE_roses_count_l1419_141927

/-- The number of pots of roses in the People's Park -/
def roses : ℕ := 65

/-- The number of pots of lilac flowers in the People's Park -/
def lilacs : ℕ := 180

/-- Theorem stating that the number of pots of roses is correct given the conditions -/
theorem roses_count :
  roses = 65 ∧ lilacs = 180 ∧ lilacs = 3 * roses - 15 :=
by sorry

end NUMINAMATH_CALUDE_roses_count_l1419_141927


namespace NUMINAMATH_CALUDE_special_function_form_l1419_141998

/-- A bijective, monotonic function from ℝ to ℝ satisfying f(t) + f⁻¹(t) = 2t for all t ∈ ℝ -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧ 
  Monotone f ∧ 
  ∀ t : ℝ, f t + Function.invFun f t = 2 * t

/-- The theorem stating that any SpecialFunction is of the form f(x) = x + c for some constant c -/
theorem special_function_form (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by sorry

end NUMINAMATH_CALUDE_special_function_form_l1419_141998


namespace NUMINAMATH_CALUDE_count_is_six_l1419_141982

/-- A type representing the blocks of digits --/
inductive DigitBlock
  | two
  | fortyfive
  | sixtyeight

/-- The set of all possible permutations of the digit blocks --/
def permutations : List (List DigitBlock) :=
  [DigitBlock.two, DigitBlock.fortyfive, DigitBlock.sixtyeight].permutations

/-- The count of all possible 5-digit numbers formed by the digits 2, 45, 68 --/
def count_five_digit_numbers : Nat := permutations.length

/-- Theorem stating that the count of possible 5-digit numbers is 6 --/
theorem count_is_six : count_five_digit_numbers = 6 := by sorry

end NUMINAMATH_CALUDE_count_is_six_l1419_141982
