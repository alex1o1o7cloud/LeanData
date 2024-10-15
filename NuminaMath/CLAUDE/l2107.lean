import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_angle_theorem_l2107_210767

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a predicate for convex quadrilaterals
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a predicate for correspondingly equal sides
def equalSides (q1 q2 : Quadrilateral) : Prop :=
  sideLength q1.A q1.B = sideLength q2.A q2.B ∧
  sideLength q1.B q1.C = sideLength q2.B q2.C ∧
  sideLength q1.C q1.D = sideLength q2.C q2.D ∧
  sideLength q1.D q1.A = sideLength q2.D q2.A

theorem quadrilateral_angle_theorem (q1 q2 : Quadrilateral) 
  (h_convex1 : isConvex q1) (h_convex2 : isConvex q2) 
  (h_equal_sides : equalSides q1 q2) 
  (h_angle_A : angle q1.D q1.A q1.B > angle q2.D q2.A q2.B) :
  angle q1.A q1.B q1.C < angle q2.A q2.B q2.C ∧
  angle q1.B q1.C q1.D > angle q2.B q2.C q2.D ∧
  angle q1.C q1.D q1.A < angle q2.C q2.D q2.A :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_theorem_l2107_210767


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2107_210777

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : b = 7)
  (h4 : (a + b + c) / 3 = a + 15)
  (h5 : (a + b + c) / 3 = c - 10) :
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2107_210777


namespace NUMINAMATH_CALUDE_complex_product_real_l2107_210794

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := x - I
  (z₁ * z₂).im = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l2107_210794


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l2107_210782

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- The number to be represented -/
def number : ℕ := 2400000

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  a := 2.4
  n := 6
  h1 := by sorry
  h2 := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem correct_scientific_notation : 
  (scientificForm.a * (10 : ℝ) ^ scientificForm.n) = number := by sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l2107_210782


namespace NUMINAMATH_CALUDE_fish_sample_count_l2107_210797

/-- Given a population of fish and a stratified sampling method, 
    calculate the number of black carp and common carp in the sample. -/
theorem fish_sample_count 
  (total_fish : ℕ) 
  (black_carp : ℕ) 
  (common_carp : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_fish = 200) 
  (h2 : black_carp = 20) 
  (h3 : common_carp = 40) 
  (h4 : sample_size = 20) : 
  (black_carp * sample_size / total_fish + common_carp * sample_size / total_fish : ℕ) = 6 := by
  sorry

#check fish_sample_count

end NUMINAMATH_CALUDE_fish_sample_count_l2107_210797


namespace NUMINAMATH_CALUDE_books_bought_at_fair_l2107_210773

theorem books_bought_at_fair (initial_books final_books : ℕ) 
  (h1 : initial_books = 9)
  (h2 : final_books = 12) :
  final_books - initial_books = 3 := by
sorry

end NUMINAMATH_CALUDE_books_bought_at_fair_l2107_210773


namespace NUMINAMATH_CALUDE_total_production_l2107_210743

/-- The daily production of fertilizer in tons -/
def daily_production : ℕ := 105

/-- The number of days of production -/
def days : ℕ := 24

/-- Theorem stating the total production over the given number of days -/
theorem total_production : daily_production * days = 2520 := by
  sorry

end NUMINAMATH_CALUDE_total_production_l2107_210743


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2107_210780

theorem max_value_of_expression (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 3) :
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 1 ∧
  ∃ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 3 ∧
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2107_210780


namespace NUMINAMATH_CALUDE_warehouse_inventory_l2107_210703

theorem warehouse_inventory (x y : ℝ) : 
  x + y = 92 ∧ 
  (2/5) * x + (1/4) * y = 26 → 
  x = 20 ∧ y = 72 := by
sorry

end NUMINAMATH_CALUDE_warehouse_inventory_l2107_210703


namespace NUMINAMATH_CALUDE_plot_length_is_75_l2107_210709

/-- Proves that the length of a rectangular plot is 75 meters given the specified conditions -/
theorem plot_length_is_75 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 50 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 75 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_75_l2107_210709


namespace NUMINAMATH_CALUDE_segmented_part_surface_area_l2107_210716

/-- Right prism with isosceles triangle base -/
structure Prism where
  height : ℝ
  baseLength : ℝ
  baseSide : ℝ

/-- Point on an edge of the prism -/
structure EdgePoint where
  edge : Fin 3
  position : ℝ

/-- Segmented part of the prism -/
structure SegmentedPart where
  prism : Prism
  pointX : EdgePoint
  pointY : EdgePoint
  pointZ : EdgePoint

/-- Surface area of the segmented part -/
def surfaceArea (part : SegmentedPart) : ℝ := sorry

/-- Main theorem -/
theorem segmented_part_surface_area 
  (p : Prism) 
  (x y z : EdgePoint) 
  (h1 : p.height = 20)
  (h2 : p.baseLength = 18)
  (h3 : p.baseSide = 15)
  (h4 : x.edge = 0 ∧ x.position = 1/2)
  (h5 : y.edge = 1 ∧ y.position = 1/2)
  (h6 : z.edge = 2 ∧ z.position = 1/2) :
  surfaceArea { prism := p, pointX := x, pointY := y, pointZ := z } = 108 := by sorry

end NUMINAMATH_CALUDE_segmented_part_surface_area_l2107_210716


namespace NUMINAMATH_CALUDE_square_areas_l2107_210723

theorem square_areas (a b : ℝ) (h1 : 4*a - 4*b = 12) (h2 : a^2 - b^2 = 69) :
  (a^2 = 169 ∧ b^2 = 100) :=
sorry

end NUMINAMATH_CALUDE_square_areas_l2107_210723


namespace NUMINAMATH_CALUDE_circle_area_l2107_210724

theorem circle_area (r : ℝ) (h : 3 / (2 * Real.pi * r) = r) : r ^ 2 * Real.pi = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l2107_210724


namespace NUMINAMATH_CALUDE_zoo_visitors_l2107_210708

theorem zoo_visitors (monday_children monday_adults tuesday_children : ℕ)
  (child_ticket_price adult_ticket_price : ℕ)
  (total_revenue : ℕ) :
  monday_children = 7 →
  monday_adults = 5 →
  tuesday_children = 4 →
  child_ticket_price = 3 →
  adult_ticket_price = 4 →
  total_revenue = 61 →
  ∃ tuesday_adults : ℕ,
    total_revenue =
      monday_children * child_ticket_price +
      monday_adults * adult_ticket_price +
      tuesday_children * child_ticket_price +
      tuesday_adults * adult_ticket_price ∧
    tuesday_adults = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l2107_210708


namespace NUMINAMATH_CALUDE_complex_roots_properties_l2107_210722

noncomputable def z₁ : ℂ := Real.sqrt 2 * Complex.exp (Complex.I * Real.pi / 4)

theorem complex_roots_properties (a b : ℝ) :
  z₁^2 + a * z₁ + b = 0 →
  ∃ z₂ : ℂ,
    z₁ = 1 + Complex.I ∧
    a = -2 ∧
    b = 2 ∧
    z₂ = 1 - Complex.I ∧
    Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂ ∧
    Complex.abs (z₁ * z₂) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_properties_l2107_210722


namespace NUMINAMATH_CALUDE_student_event_combinations_l2107_210793

theorem student_event_combinations : 
  let num_students : ℕ := 4
  let num_events : ℕ := 3
  num_events ^ num_students = 81 := by sorry

end NUMINAMATH_CALUDE_student_event_combinations_l2107_210793


namespace NUMINAMATH_CALUDE_total_chimpanzees_l2107_210753

/- Define the number of chimps moving to the new cage -/
def chimps_new_cage : ℕ := 18

/- Define the number of chimps staying in the old cage -/
def chimps_old_cage : ℕ := 27

/- Theorem stating that the total number of chimpanzees is 45 -/
theorem total_chimpanzees : chimps_new_cage + chimps_old_cage = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_chimpanzees_l2107_210753


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l2107_210727

theorem square_root_three_expansion 
  (a b c d : ℕ+) 
  (h : (a : ℝ) + (b : ℝ) * Real.sqrt 3 = ((c : ℝ) + (d : ℝ) * Real.sqrt 3) ^ 2) : 
  (a : ℝ) = (c : ℝ) ^ 2 + 3 * (d : ℝ) ^ 2 ∧ (b : ℝ) = 2 * (c : ℝ) * (d : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l2107_210727


namespace NUMINAMATH_CALUDE_coupon_one_best_l2107_210706

/-- Represents the discount offered by a coupon given a price --/
def discount (price : ℝ) : ℕ → ℝ
  | 1 => 0.1 * price
  | 2 => 20
  | 3 => 0.18 * (price - 100)
  | _ => 0  -- Default case for invalid coupon numbers

theorem coupon_one_best (price : ℝ) (h : price > 100) :
  (discount price 1 > discount price 2 ∧ discount price 1 > discount price 3) ↔ 
  (200 < price ∧ price < 225) := by
sorry

end NUMINAMATH_CALUDE_coupon_one_best_l2107_210706


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2107_210707

theorem complex_power_magnitude : Complex.abs ((1 - Complex.I * 2) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2107_210707


namespace NUMINAMATH_CALUDE_complementary_angle_supplement_l2107_210700

theorem complementary_angle_supplement (A B : Real) : 
  (A + B = 90) → (180 - A = 90 + B) := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_supplement_l2107_210700


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l2107_210721

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
  n ≥ 3 → 
  (180 * (n - 2) : ℝ) = 150 * n → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l2107_210721


namespace NUMINAMATH_CALUDE_undeclared_majors_fraction_l2107_210784

/-- Represents the distribution of students across different years -/
structure StudentDistribution where
  firstYear : Rat
  secondYear : Rat
  thirdYear : Rat
  fourthYear : Rat
  postgraduate : Rat

/-- Represents the proportion of students who have not declared a major in each year -/
structure UndeclaredMajors where
  firstYear : Rat
  secondYear : Rat
  thirdYear : Rat
  fourthYear : Rat
  postgraduate : Rat

/-- Calculates the fraction of all students who have not declared a major -/
def fractionUndeclaredMajors (dist : StudentDistribution) (undeclared : UndeclaredMajors) : Rat :=
  dist.firstYear * undeclared.firstYear +
  dist.secondYear * undeclared.secondYear +
  dist.thirdYear * undeclared.thirdYear +
  dist.fourthYear * undeclared.fourthYear +
  dist.postgraduate * undeclared.postgraduate

theorem undeclared_majors_fraction 
  (dist : StudentDistribution)
  (undeclared : UndeclaredMajors)
  (h1 : dist.firstYear = 1/5)
  (h2 : dist.secondYear = 2/5)
  (h3 : dist.thirdYear = 1/5)
  (h4 : dist.fourthYear = 1/10)
  (h5 : dist.postgraduate = 1/10)
  (h6 : undeclared.firstYear = 4/5)
  (h7 : undeclared.secondYear = 3/4)
  (h8 : undeclared.thirdYear = 1/3)
  (h9 : undeclared.fourthYear = 1/6)
  (h10 : undeclared.postgraduate = 1/12) :
  fractionUndeclaredMajors dist undeclared = 14/25 := by
  sorry


end NUMINAMATH_CALUDE_undeclared_majors_fraction_l2107_210784


namespace NUMINAMATH_CALUDE_restaurant_tip_calculation_l2107_210764

theorem restaurant_tip_calculation 
  (food_cost : ℝ) 
  (service_fee_percentage : ℝ) 
  (total_spent : ℝ) 
  (h1 : food_cost = 50) 
  (h2 : service_fee_percentage = 0.12) 
  (h3 : total_spent = 61) : 
  total_spent - (food_cost + food_cost * service_fee_percentage) = 5 := by
sorry

end NUMINAMATH_CALUDE_restaurant_tip_calculation_l2107_210764


namespace NUMINAMATH_CALUDE_deck_size_l2107_210783

/-- The number of cards in a deck of playing cards. -/
def num_cards : ℕ := 52

/-- The number of hearts on each card. -/
def hearts_per_card : ℕ := 4

/-- The cost of each cow in dollars. -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars. -/
def total_cost : ℕ := 83200

/-- The number of cows in Devonshire. -/
def num_cows : ℕ := total_cost / cost_per_cow

/-- The number of hearts in the deck. -/
def num_hearts : ℕ := num_cows / 2

theorem deck_size :
  num_cards = num_hearts / hearts_per_card ∧
  num_cows = 2 * num_hearts ∧
  num_cows * cost_per_cow = total_cost :=
by sorry

end NUMINAMATH_CALUDE_deck_size_l2107_210783


namespace NUMINAMATH_CALUDE_snack_eaters_problem_l2107_210745

/-- The number of new outsiders who joined for snacks after the first group left -/
def new_outsiders : ℕ := sorry

theorem snack_eaters_problem (initial_people : ℕ) (initial_snackers : ℕ) (first_outsiders : ℕ) 
  (more_left : ℕ) (final_snackers : ℕ) :
  initial_people = 200 →
  initial_snackers = 100 →
  first_outsiders = 20 →
  more_left = 30 →
  final_snackers = 20 →
  new_outsiders = 40 := by sorry

end NUMINAMATH_CALUDE_snack_eaters_problem_l2107_210745


namespace NUMINAMATH_CALUDE_wednesday_distance_l2107_210766

/-- Terese's running schedule for the week -/
structure RunningSchedule where
  monday : Float
  tuesday : Float
  wednesday : Float
  thursday : Float

/-- Theorem: Given Terese's running schedule and average distance, prove she runs 3.6 miles on Wednesday -/
theorem wednesday_distance (schedule : RunningSchedule) 
  (h1 : schedule.monday = 4.2)
  (h2 : schedule.tuesday = 3.8)
  (h3 : schedule.thursday = 4.4)
  (h4 : (schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday) / 4 = 4) :
  schedule.wednesday = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_wednesday_distance_l2107_210766


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l2107_210715

theorem square_sum_lower_bound (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  x^2 + y^2 + z^2 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l2107_210715


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2107_210717

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 2) (h2 : b = 7) (h3 : Odd c) : a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2107_210717


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_S_l2107_210730

def S : Set ℤ := {x | ∃ n : ℤ, x = n^5 - 5*n^3 + 4*n ∧ ¬(3 ∣ n)}

theorem largest_common_divisor_of_S : 
  ∀ k : ℤ, (∀ x ∈ S, k ∣ x) → k ≤ 360 ∧ 
  ∀ x ∈ S, 360 ∣ x :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_S_l2107_210730


namespace NUMINAMATH_CALUDE_weight_lowering_feel_l2107_210736

theorem weight_lowering_feel (num_plates : ℕ) (weight_per_plate : ℝ) (increase_percentage : ℝ) :
  num_plates = 10 →
  weight_per_plate = 30 →
  increase_percentage = 0.2 →
  (num_plates : ℝ) * weight_per_plate * (1 + increase_percentage) = 360 := by
  sorry

end NUMINAMATH_CALUDE_weight_lowering_feel_l2107_210736


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2107_210772

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (0 < x ∧ x < 7) → (|x - 2| < 5)) ∧
  (∃ x : ℝ, |x - 2| < 5 ∧ ¬(0 < x ∧ x < 7)) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2107_210772


namespace NUMINAMATH_CALUDE_function_expression_l2107_210769

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = x + 1) :
  ∀ x, f x = (1/2) * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_function_expression_l2107_210769


namespace NUMINAMATH_CALUDE_journey_length_l2107_210785

theorem journey_length : 
  ∀ (x : ℚ), 
  (x / 4 : ℚ) + 25 + (x / 6 : ℚ) = x → 
  x = 300 / 7 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l2107_210785


namespace NUMINAMATH_CALUDE_plumber_max_shower_charge_l2107_210731

def plumber_problem (sink_charge toilet_charge shower_charge : ℕ) : Prop :=
  let job1 := 3 * toilet_charge + 3 * sink_charge
  let job2 := 2 * toilet_charge + 5 * sink_charge
  let job3 := toilet_charge + 2 * shower_charge + 3 * sink_charge
  sink_charge = 30 ∧
  toilet_charge = 50 ∧
  (job1 ≤ 250 ∧ job2 ≤ 250 ∧ job3 ≤ 250) ∧
  (job1 = 250 ∨ job2 = 250 ∨ job3 = 250) →
  shower_charge ≤ 55

theorem plumber_max_shower_charge :
  ∃ (shower_charge : ℕ), plumber_problem 30 50 shower_charge ∧
  ∀ (x : ℕ), x > shower_charge → ¬ plumber_problem 30 50 x :=
sorry

end NUMINAMATH_CALUDE_plumber_max_shower_charge_l2107_210731


namespace NUMINAMATH_CALUDE_rectangles_must_be_squares_l2107_210771

theorem rectangles_must_be_squares (n : ℕ) (is_prime : ℕ → Prop) 
  (total_squares : ℕ) (h_prime : is_prime total_squares) : 
  ∀ (a b : ℕ) (h_rect : ∀ i : Fin n, ∃ (k : ℕ), a * b = (total_squares / n) * k^2), a = b :=
by
  sorry

end NUMINAMATH_CALUDE_rectangles_must_be_squares_l2107_210771


namespace NUMINAMATH_CALUDE_cyclist_speeds_l2107_210734

-- Define the distance between A and B
def total_distance : ℝ := 240

-- Define the time difference between starts
def start_time_diff : ℝ := 0.5

-- Define the speed difference between cyclists
def speed_diff : ℝ := 3

-- Define the time taken to fix the bike
def fix_time : ℝ := 1.5

-- Define the speeds of cyclists A and B
def speed_A : ℝ := 12
def speed_B : ℝ := speed_A + speed_diff

-- Theorem to prove
theorem cyclist_speeds :
  -- Person B reaches midpoint when bike breaks down
  (total_distance / 2) / speed_B = total_distance / speed_A - start_time_diff - fix_time :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speeds_l2107_210734


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2107_210751

theorem vector_equation_solution :
  let a : ℚ := 23/7
  let b : ℚ := -1/7
  let v1 : Fin 2 → ℚ := ![1, 4]
  let v2 : Fin 2 → ℚ := ![3, -2]
  let result : Fin 2 → ℚ := ![2, 10]
  (a • v1 + b • v2 = result) := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2107_210751


namespace NUMINAMATH_CALUDE_carolyn_sticker_count_l2107_210799

/-- Given that Belle collected 97 stickers and Carolyn collected 18 fewer stickers than Belle,
    prove that Carolyn collected 79 stickers. -/
theorem carolyn_sticker_count :
  ∀ (belle_stickers carolyn_stickers : ℕ),
    belle_stickers = 97 →
    carolyn_stickers = belle_stickers - 18 →
    carolyn_stickers = 79 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_sticker_count_l2107_210799


namespace NUMINAMATH_CALUDE_no_eight_digit_six_times_l2107_210768

theorem no_eight_digit_six_times : ¬ ∃ (N : ℕ), 
  (10000000 ≤ N) ∧ (N < 100000000) ∧
  (∃ (p q : ℕ), N = 10000 * p + q ∧ q < 10000 ∧ 10000 * q + p = 6 * N) :=
sorry

end NUMINAMATH_CALUDE_no_eight_digit_six_times_l2107_210768


namespace NUMINAMATH_CALUDE_final_value_of_A_l2107_210760

theorem final_value_of_A : ∀ A : ℤ, A = 15 → (A = -15 + 5) → A = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l2107_210760


namespace NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l2107_210779

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l2107_210779


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l2107_210704

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*y - 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 5

/-- Theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five :
  ∀ x y : ℝ, circle_equation x y → ∃ center_x center_y : ℝ,
    (x - center_x)^2 + (y - center_y)^2 = circle_radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l2107_210704


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l2107_210778

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_first_three : a 0 + a 1 + a 2 = 168
  specific_diff : a 1 - a 4 = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l2107_210778


namespace NUMINAMATH_CALUDE_complex_on_line_with_magnitude_l2107_210750

theorem complex_on_line_with_magnitude (z : ℂ) :
  (z.im = 2 * z.re) → (Complex.abs z = Real.sqrt 5) →
  (z = Complex.mk 1 2 ∨ z = Complex.mk (-1) (-2)) := by
  sorry

end NUMINAMATH_CALUDE_complex_on_line_with_magnitude_l2107_210750


namespace NUMINAMATH_CALUDE_fraction_conversion_equivalence_l2107_210765

theorem fraction_conversion_equivalence (x : ℚ) : 
  (x + 1) / (2 / 5) - (2 / 10 * x - 1) / (7 / 10) = 1 ↔ 
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_conversion_equivalence_l2107_210765


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2107_210781

theorem no_positive_integer_solutions :
  ¬∃ (x y z : ℕ+), x^2 + y^2 = 7 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2107_210781


namespace NUMINAMATH_CALUDE_zero_exists_in_interval_l2107_210792

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x - 3

-- State the theorem
theorem zero_exists_in_interval :
  ∃ c ∈ Set.Ioo (1/2 : ℝ) 1, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_exists_in_interval_l2107_210792


namespace NUMINAMATH_CALUDE_point_on_line_implies_m_value_l2107_210758

/-- Given a point P(1, -2) on the line 4x - my + 12 = 0, prove that m = -8 -/
theorem point_on_line_implies_m_value (m : ℝ) : 
  (4 * 1 - m * (-2) + 12 = 0) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_m_value_l2107_210758


namespace NUMINAMATH_CALUDE_correlation_coefficient_and_fit_l2107_210776

/-- Represents the correlation coefficient in regression analysis -/
def correlation_coefficient : ℝ := sorry

/-- Represents the goodness of fit of a regression model -/
def goodness_of_fit : ℝ := sorry

/-- States that as the absolute value of the correlation coefficient 
    approaches 1, the goodness of fit improves -/
theorem correlation_coefficient_and_fit :
  ∀ ε > 0, ∃ δ > 0, ∀ R : ℝ,
    |R| > 1 - δ → goodness_of_fit > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_and_fit_l2107_210776


namespace NUMINAMATH_CALUDE_circle_equation_l2107_210702

theorem circle_equation (h : ℝ) :
  (∃ (x : ℝ), (x - 2)^2 + (-3)^2 = 5^2 ∧ h = x) →
  ((h - 6)^2 + y^2 = 25 ∨ (h + 2)^2 + y^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2107_210702


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2107_210795

/-- Given two parallel vectors a and b, prove that x + y = -9 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![-1, 2, 1]
  let b : Fin 3 → ℝ := ![3, x, y]
  (∃ (k : ℝ), ∀ i, b i = k * (a i)) →
  x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2107_210795


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l2107_210763

-- Define the function f
def f (x : ℝ) : ℝ := (2 + x)^2 - 3*x

-- State the theorem
theorem derivative_f_at_1 :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l2107_210763


namespace NUMINAMATH_CALUDE_f_min_at_zero_l2107_210737

def f (x : ℝ) : ℝ := (x^2 - 4)^3 + 1

theorem f_min_at_zero :
  (∀ x : ℝ, f 0 ≤ f x) ∧ f 0 = -63 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_zero_l2107_210737


namespace NUMINAMATH_CALUDE_inequality_solution_l2107_210739

theorem inequality_solution (x : ℝ) : 
  (1 / (x^2 + 1) > 4/x + 21/10) ↔ (-2 < x ∧ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2107_210739


namespace NUMINAMATH_CALUDE_complex_number_equality_l2107_210719

theorem complex_number_equality (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (1 - Complex.I)) = 
   Complex.im ((1 + b * Complex.I) / (1 - Complex.I))) → b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2107_210719


namespace NUMINAMATH_CALUDE_C_power_50_l2107_210738

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l2107_210738


namespace NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l2107_210725

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

-- Define parallel condition for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem x_eq_3_sufficient_not_necessary :
  (∀ x : ℝ, x = 3 → parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, parallel (a x) (b x) → x = 3) :=
sorry

end NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l2107_210725


namespace NUMINAMATH_CALUDE_veena_payment_fraction_l2107_210789

/-- Represents the payment amounts of 6 friends at a restaurant -/
structure DinnerPayment where
  akshitha : ℚ
  veena : ℚ
  lasya : ℚ
  sandhya : ℚ
  ramesh : ℚ
  kavya : ℚ

/-- Theorem stating that Veena paid 1/8 of the total bill -/
theorem veena_payment_fraction (p : DinnerPayment) 
  (h1 : p.akshitha = 3/4 * p.veena)
  (h2 : p.veena = 1/2 * p.lasya)
  (h3 : p.lasya = 5/6 * p.sandhya)
  (h4 : p.sandhya = 4/8 * p.ramesh)
  (h5 : p.ramesh = 3/5 * p.kavya)
  : p.veena = 1/8 * (p.akshitha + p.veena + p.lasya + p.sandhya + p.ramesh + p.kavya) := by
  sorry


end NUMINAMATH_CALUDE_veena_payment_fraction_l2107_210789


namespace NUMINAMATH_CALUDE_paint_used_approx_253_33_l2107_210798

/-- Calculate the amount of paint used over five weeks given an initial amount and weekly usage fractions. -/
def paintUsed (initialPaint : ℝ) (week1Fraction week2Fraction week3Fraction week4Fraction week5Fraction : ℝ) : ℝ :=
  let remainingAfterWeek1 := initialPaint * (1 - week1Fraction)
  let remainingAfterWeek2 := remainingAfterWeek1 * (1 - week2Fraction)
  let remainingAfterWeek3 := remainingAfterWeek2 * (1 - week3Fraction)
  let remainingAfterWeek4 := remainingAfterWeek3 * (1 - week4Fraction)
  let usedInWeek5 := remainingAfterWeek4 * week5Fraction
  initialPaint - remainingAfterWeek4 + usedInWeek5

/-- Theorem stating that given the initial paint amount and weekly usage fractions, 
    the total paint used after five weeks is approximately 253.33 gallons. -/
theorem paint_used_approx_253_33 :
  ∃ ε > 0, ε < 0.01 ∧ 
  |paintUsed 360 (1/9) (1/5) (1/3) (1/4) (1/6) - 253.33| < ε :=
sorry

end NUMINAMATH_CALUDE_paint_used_approx_253_33_l2107_210798


namespace NUMINAMATH_CALUDE_unique_solution_l2107_210729

/-- Represents the quantities and prices of two batches of products --/
structure BatchData where
  quantity1 : ℕ
  quantity2 : ℕ
  price1 : ℚ
  price2 : ℚ

/-- Checks if the given batch data satisfies the problem conditions --/
def satisfiesConditions (data : BatchData) : Prop :=
  data.quantity1 * data.price1 = 4000 ∧
  data.quantity2 * data.price2 = 8800 ∧
  data.quantity2 = 2 * data.quantity1 ∧
  data.price2 = data.price1 + 4

/-- Theorem stating that the only solution satisfying the conditions is 100 and 200 units --/
theorem unique_solution :
  ∀ data : BatchData, satisfiesConditions data →
    data.quantity1 = 100 ∧ data.quantity2 = 200 := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_unique_solution_l2107_210729


namespace NUMINAMATH_CALUDE_building_has_seven_floors_l2107_210740

/-- Represents a building with floors -/
structure Building where
  totalFloors : ℕ
  ninasFloor : ℕ
  shurasFloor : ℕ

/-- Calculates the distance of Shura's mistaken path -/
def mistakenPathDistance (b : Building) : ℕ :=
  (b.totalFloors - b.ninasFloor) + (b.totalFloors - b.shurasFloor)

/-- Calculates the distance of Shura's direct path -/
def directPathDistance (b : Building) : ℕ :=
  if b.ninasFloor ≥ b.shurasFloor then b.ninasFloor - b.shurasFloor
  else b.shurasFloor - b.ninasFloor

/-- Theorem stating the conditions and conclusion about the building -/
theorem building_has_seven_floors :
  ∃ (b : Building),
    b.ninasFloor = 6 ∧
    b.totalFloors > b.ninasFloor ∧
    (mistakenPathDistance b : ℚ) = 1.5 * (directPathDistance b : ℚ) ∧
    b.totalFloors = 7 := by
  sorry

end NUMINAMATH_CALUDE_building_has_seven_floors_l2107_210740


namespace NUMINAMATH_CALUDE_consecutive_number_sums_contradiction_l2107_210762

theorem consecutive_number_sums_contradiction (a : Fin 15 → ℤ) :
  (∀ i : Fin 13, a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i : Fin 12, a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by sorry

end NUMINAMATH_CALUDE_consecutive_number_sums_contradiction_l2107_210762


namespace NUMINAMATH_CALUDE_tan_sin_intersection_count_l2107_210757

open Real

theorem tan_sin_intersection_count :
  let f : ℝ → ℝ := λ x => tan x - sin x
  ∃! (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, -2*π ≤ x ∧ x ≤ 2*π ∧ f x = 0) ∧
    (∀ x, -2*π ≤ x ∧ x ≤ 2*π ∧ f x = 0 → x ∈ s) :=
by
  sorry

end NUMINAMATH_CALUDE_tan_sin_intersection_count_l2107_210757


namespace NUMINAMATH_CALUDE_product_reciprocal_sum_l2107_210728

theorem product_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_product : a * b = 16) (h_reciprocal : 1 / a = 3 * (1 / b)) : 
  a + b = 16 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_product_reciprocal_sum_l2107_210728


namespace NUMINAMATH_CALUDE_xyz_value_l2107_210787

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b - c) * (x + 2))
  (eq2 : b = (a - c) * (y + 2))
  (eq3 : c = (a - b) * (z + 2))
  (eq4 : x * y + x * z + y * z = 12)
  (eq5 : x + y + z = 6) :
  x * y * z = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2107_210787


namespace NUMINAMATH_CALUDE_medicine_weight_l2107_210754

/-- Represents the weight system used for measurement -/
inductive WeightSystem
  | Ancient
  | Modern

/-- Represents a weight measurement -/
structure Weight where
  jin : ℕ
  liang : ℕ
  system : WeightSystem

/-- Converts a Weight to grams -/
def Weight.toGrams (w : Weight) : ℕ :=
  match w.system with
  | WeightSystem.Ancient => w.jin * 600 + w.liang * (600 / 16)
  | WeightSystem.Modern => w.jin * 500 + w.liang * (500 / 10)

/-- The theorem to be proved -/
theorem medicine_weight (w₁ w₂ : Weight) 
  (h₁ : w₁.system = WeightSystem.Ancient)
  (h₂ : w₂.system = WeightSystem.Modern)
  (h₃ : w₁.jin + w₂.jin = 5)
  (h₄ : w₁.liang + w₂.liang = 68)
  (h₅ : w₁.jin * 16 + w₁.liang = w₁.liang)
  (h₆ : w₂.jin * 10 + w₂.liang = w₂.liang) :
  w₁.toGrams + w₂.toGrams = 2800 := by
  sorry


end NUMINAMATH_CALUDE_medicine_weight_l2107_210754


namespace NUMINAMATH_CALUDE_certain_number_exists_l2107_210756

theorem certain_number_exists : ∃ x : ℝ, 
  3500 - (1000 / x) = 3451.2195121951218 ∧ 
  abs (x - 20.5) < 0.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l2107_210756


namespace NUMINAMATH_CALUDE_train_meetings_l2107_210726

-- Define the travel time in minutes
def travel_time : ℕ := 210

-- Define the departure interval in minutes
def departure_interval : ℕ := 60

-- Define the time difference between the 9:00 AM train and the first train in minutes
def time_difference : ℕ := 180

-- Define a function to calculate the number of meetings
def number_of_meetings (travel_time departure_interval time_difference : ℕ) : ℕ :=
  -- The actual calculation would go here, but we're using sorry as per instructions
  sorry

-- Theorem statement
theorem train_meetings :
  number_of_meetings travel_time departure_interval time_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_train_meetings_l2107_210726


namespace NUMINAMATH_CALUDE_sqrt_relationship_l2107_210746

theorem sqrt_relationship (h1 : Real.sqrt 23.6 = 4.858) (h2 : Real.sqrt 2.36 = 1.536) :
  Real.sqrt 0.00236 = 0.04858 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_relationship_l2107_210746


namespace NUMINAMATH_CALUDE_unique_plane_through_skew_lines_l2107_210796

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a simplified representation

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane in 3D space
  -- This is a simplified representation

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines to be skew
  sorry

/-- Predicate to check if a plane passes through a line -/
def passes_through (p : Plane3D) (l : Line3D) : Prop :=
  -- Define the condition for a plane to pass through a line
  sorry

/-- Predicate to check if a plane is parallel to a line -/
def is_parallel_to (p : Plane3D) (l : Line3D) : Prop :=
  -- Define the condition for a plane to be parallel to a line
  sorry

/-- Theorem stating the existence and uniqueness of a plane passing through one skew line and parallel to another -/
theorem unique_plane_through_skew_lines (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! p : Plane3D, passes_through p l1 ∧ is_parallel_to p l2 :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_skew_lines_l2107_210796


namespace NUMINAMATH_CALUDE_complex_power_sum_l2107_210701

theorem complex_power_sum (z : ℂ) (h : z = (1 - I) / Real.sqrt 2) : 
  z^100 + z^50 + 1 = -I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2107_210701


namespace NUMINAMATH_CALUDE_third_side_length_l2107_210761

theorem third_side_length (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 →
  a > 0 → b > 0 → c > 0 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  ∃ (x y z : ℝ), x = a ∧ y = b ∧ z = c ∧ 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y > z ∧ y + z > x ∧ z + x > y :=
by sorry

end NUMINAMATH_CALUDE_third_side_length_l2107_210761


namespace NUMINAMATH_CALUDE_cookies_baked_l2107_210714

theorem cookies_baked (pans : ℕ) (cookies_per_pan : ℕ) (h1 : pans = 12) (h2 : cookies_per_pan = 15) :
  pans * cookies_per_pan = 180 := by
  sorry

end NUMINAMATH_CALUDE_cookies_baked_l2107_210714


namespace NUMINAMATH_CALUDE_polygon_diagonals_l2107_210744

theorem polygon_diagonals (n : ℕ) : 
  (n ≥ 3) → (n - 3 ≤ 5) → (n = 8) :=
by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l2107_210744


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2107_210711

theorem quadratic_solution_difference_squared : 
  ∀ f g : ℝ, (4 * f^2 + 8 * f - 48 = 0) → (4 * g^2 + 8 * g - 48 = 0) → (f - g)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2107_210711


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_l2107_210748

theorem cos_75_cos_15_minus_sin_75_sin_15 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_l2107_210748


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2107_210752

theorem algebraic_expression_value (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 3 / (a + 2)) / ((a^2 - 1) / (a + 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2107_210752


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l2107_210741

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l2107_210741


namespace NUMINAMATH_CALUDE_max_product_constraint_l2107_210713

theorem max_product_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 1) :
  x * y ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l2107_210713


namespace NUMINAMATH_CALUDE_election_result_l2107_210786

/-- The percentage of votes received by candidate A out of the total valid votes -/
def candidate_A_percentage : ℝ := 65

/-- The percentage of invalid votes out of the total votes -/
def invalid_vote_percentage : ℝ := 15

/-- The total number of votes cast in the election -/
def total_votes : ℕ := 560000

/-- The number of valid votes polled in favor of candidate A -/
def votes_for_candidate_A : ℕ := 309400

theorem election_result :
  (candidate_A_percentage / 100) * ((100 - invalid_vote_percentage) / 100) * total_votes = votes_for_candidate_A := by
  sorry

end NUMINAMATH_CALUDE_election_result_l2107_210786


namespace NUMINAMATH_CALUDE_twenty_times_nineteen_plus_twenty_plus_nineteen_l2107_210732

theorem twenty_times_nineteen_plus_twenty_plus_nineteen : 20 * 19 + 20 + 19 = 419 := by
  sorry

end NUMINAMATH_CALUDE_twenty_times_nineteen_plus_twenty_plus_nineteen_l2107_210732


namespace NUMINAMATH_CALUDE_five_students_three_events_outcomes_l2107_210742

/-- The number of different possible outcomes for champions in a sports competition. -/
def championOutcomes (numStudents : ℕ) (numEvents : ℕ) : ℕ :=
  numStudents ^ numEvents

/-- Theorem stating that with 5 students and 3 events, there are 125 possible outcomes. -/
theorem five_students_three_events_outcomes :
  championOutcomes 5 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_events_outcomes_l2107_210742


namespace NUMINAMATH_CALUDE_chocolates_remaining_on_day_five_l2107_210749

/-- Chocolates eaten per day --/
def chocolates_eaten (day : Nat) : Nat :=
  match day with
  | 1 => 4
  | 2 => 2 * 4 - 3
  | 3 => 4 - 2
  | 4 => (4 - 2) - 1
  | _ => 0

/-- Total chocolates eaten up to a given day --/
def total_eaten (day : Nat) : Nat :=
  match day with
  | 0 => 0
  | n + 1 => total_eaten n + chocolates_eaten (n + 1)

theorem chocolates_remaining_on_day_five : 
  24 - total_eaten 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_remaining_on_day_five_l2107_210749


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2107_210720

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  15 / (x * (35 - 8 * x^3)^(1/3)) = 2 * x + (35 - 8 * x^3)^(1/3)

/-- The set of all solutions to the equation -/
def solution_set : Set ℝ :=
  {x : ℝ | equation x ∧ x ≠ 0 ∧ 35 - 8 * x^3 > 0}

/-- The theorem stating that the sum of all solutions is 2.5 -/
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = 2.5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2107_210720


namespace NUMINAMATH_CALUDE_tangent_inclination_range_implies_x_coordinate_range_l2107_210735

/-- The curve C defined by y = x^2 + 2x + 3 -/
def C (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The derivative of C -/
def C' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_inclination_range_implies_x_coordinate_range :
  ∀ x : ℝ,
  (∃ y : ℝ, y = C x) →
  (π/4 ≤ Real.arctan (C' x) ∧ Real.arctan (C' x) ≤ π/2) →
  x ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_inclination_range_implies_x_coordinate_range_l2107_210735


namespace NUMINAMATH_CALUDE_max_overlapping_squares_theorem_l2107_210788

/-- Represents a square on the checkerboard -/
structure CheckerboardSquare where
  sideLength : Real
  (side_positive : sideLength > 0)

/-- Represents the square card -/
structure Card where
  sideLength : Real
  (side_positive : sideLength > 0)

/-- Calculates the maximum number of squares a card can overlap -/
def maxOverlappingSquares (square : CheckerboardSquare) (card : Card) (minOverlap : Real) : Nat :=
  sorry

theorem max_overlapping_squares_theorem (square : CheckerboardSquare) (card : Card) :
  square.sideLength = 0.75 →
  card.sideLength = 2 →
  maxOverlappingSquares square card 0.25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_overlapping_squares_theorem_l2107_210788


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2107_210774

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2107_210774


namespace NUMINAMATH_CALUDE_jose_to_john_ratio_l2107_210733

-- Define the total amount and the ratios
def total_amount : ℕ := 4800
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6

-- Define John's share
def john_share : ℕ := 1600

-- Theorem to prove
theorem jose_to_john_ratio :
  let total_ratio := ratio_john + ratio_jose + ratio_binoy
  let share_value := total_amount / total_ratio
  let jose_share := share_value * ratio_jose
  jose_share / john_share = 2 := by
  sorry

end NUMINAMATH_CALUDE_jose_to_john_ratio_l2107_210733


namespace NUMINAMATH_CALUDE_units_digit_100_factorial_l2107_210759

theorem units_digit_100_factorial (n : ℕ) : n = 100 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_100_factorial_l2107_210759


namespace NUMINAMATH_CALUDE_pet_food_cost_l2107_210705

theorem pet_food_cost (total_cost rabbit_toy_cost cage_cost found_money : ℚ)
  (h1 : total_cost = 24.81)
  (h2 : rabbit_toy_cost = 6.51)
  (h3 : cage_cost = 12.51)
  (h4 : found_money = 1.00) :
  total_cost - (rabbit_toy_cost + cage_cost) + found_money = 6.79 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_cost_l2107_210705


namespace NUMINAMATH_CALUDE_count_fours_to_1000_l2107_210712

/-- Count of digit 4 in a single number -/
def count_fours (n : ℕ) : ℕ := sorry

/-- Sum of count_fours for all numbers from 1 to n -/
def total_fours (n : ℕ) : ℕ := sorry

/-- The count of the digit 4 appearing in the integers from 1 to 1000 is equal to 300 -/
theorem count_fours_to_1000 : total_fours 1000 = 300 := by sorry

end NUMINAMATH_CALUDE_count_fours_to_1000_l2107_210712


namespace NUMINAMATH_CALUDE_employee_reduction_l2107_210747

theorem employee_reduction (original : ℕ) : 
  let after_first := (9 : ℚ) / 10 * original
  let after_second := (19 : ℚ) / 20 * after_first
  let after_third := (22 : ℚ) / 25 * after_second
  after_third = 195 → original = 259 := by
sorry

end NUMINAMATH_CALUDE_employee_reduction_l2107_210747


namespace NUMINAMATH_CALUDE_interference_facts_l2107_210775

/-- A fact about light -/
inductive LightFact
  | SignalTransmission
  | SurfaceFlatness
  | PrismSpectrum
  | OilFilmColors

/-- Predicate to determine if a light fact involves interference -/
def involves_interference (fact : LightFact) : Prop :=
  match fact with
  | LightFact.SurfaceFlatness => true
  | LightFact.OilFilmColors => true
  | _ => false

/-- Theorem stating that only facts 2 and 4 involve interference -/
theorem interference_facts :
  (∀ f : LightFact, involves_interference f ↔ (f = LightFact.SurfaceFlatness ∨ f = LightFact.OilFilmColors)) :=
by sorry

end NUMINAMATH_CALUDE_interference_facts_l2107_210775


namespace NUMINAMATH_CALUDE_complex_power_sum_l2107_210755

theorem complex_power_sum (z : ℂ) (h : z = -(1 - Complex.I) / Real.sqrt 2) : 
  z^100 + z^50 + 1 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2107_210755


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_two_minus_two_l2107_210718

theorem absolute_value_sqrt_two_minus_two :
  (1 : ℝ) < Real.sqrt 2 ∧ Real.sqrt 2 < 2 →
  |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_two_minus_two_l2107_210718


namespace NUMINAMATH_CALUDE_football_tournament_scheduling_l2107_210790

theorem football_tournament_scheduling (n : ℕ) (h_even : Even n) :
  ∃ schedule : Fin (n - 1) → Fin n → Fin n,
    (∀ round : Fin (n - 1), ∀ team : Fin n, 
      schedule round team ≠ team ∧ 
      (∀ other_team : Fin n, schedule round team = other_team → schedule round other_team = team)) ∧
    (∀ team1 team2 : Fin n, team1 ≠ team2 → 
      ∃! round : Fin (n - 1), schedule round team1 = team2 ∨ schedule round team2 = team1) := by
  sorry

end NUMINAMATH_CALUDE_football_tournament_scheduling_l2107_210790


namespace NUMINAMATH_CALUDE_triangle_side_length_l2107_210791

theorem triangle_side_length (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) →
  (|a + b - c| + |a - b - c| = 10) → 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2107_210791


namespace NUMINAMATH_CALUDE_six_point_configuration_exists_l2107_210770

/-- A configuration of six points in 3D space -/
def Configuration := Fin 6 → ℝ × ℝ × ℝ

/-- Predicate to check if two line segments intersect only at their endpoints -/
def valid_intersection (a b c d : ℝ × ℝ × ℝ) : Prop :=
  (a = c ∧ b ≠ d) ∨ (a = d ∧ b ≠ c) ∨ (b = c ∧ a ≠ d) ∨ (b = d ∧ a ≠ c) ∨ (a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d)

/-- Predicate to check if a configuration is valid -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ i j k l : Fin 6, i ≠ j → k ≠ l → i ≠ k ∨ i ≠ l → j ≠ k ∨ j ≠ l →
    valid_intersection (config i) (config j) (config k) (config l)

theorem six_point_configuration_exists : ∃ config : Configuration, valid_configuration config :=
sorry

end NUMINAMATH_CALUDE_six_point_configuration_exists_l2107_210770


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l2107_210710

theorem chicken_rabbit_problem (total_animals total_feet : ℕ) 
  (h1 : total_animals = 35)
  (h2 : total_feet = 94) :
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_animals ∧
    2 * chickens + 4 * rabbits = total_feet ∧
    chickens = 23 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l2107_210710
