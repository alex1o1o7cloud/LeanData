import Mathlib

namespace NUMINAMATH_CALUDE_max_children_theorem_l1694_169443

/-- Represents the movie theater pricing and budget scenario -/
structure MovieTheater where
  budget : ℕ
  adultTicketCost : ℕ
  childTicketCost : ℕ
  childTicketGroupDiscount : ℕ
  groupDiscountThreshold : ℕ
  snackCost : ℕ

/-- Calculates the maximum number of children that can be taken to the movies -/
def maxChildren (mt : MovieTheater) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of children is 12 with group discount -/
theorem max_children_theorem (mt : MovieTheater) 
  (h1 : mt.budget = 100)
  (h2 : mt.adultTicketCost = 12)
  (h3 : mt.childTicketCost = 6)
  (h4 : mt.childTicketGroupDiscount = 4)
  (h5 : mt.groupDiscountThreshold = 5)
  (h6 : mt.snackCost = 3) :
  maxChildren mt = 12 ∧ 
  12 * mt.childTicketGroupDiscount + 12 * mt.snackCost + mt.adultTicketCost ≤ mt.budget :=
sorry

end NUMINAMATH_CALUDE_max_children_theorem_l1694_169443


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l1694_169453

-- Define proposition p
def p : Prop := ∃ x : ℝ, 2^x > x

-- Define evenness for a function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a line
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x, f (a + x) = f (a - x)

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, is_even (fun x => f (x - 1)) → 
  symmetric_about f 1

-- Theorem to prove
theorem p_or_q_is_true : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l1694_169453


namespace NUMINAMATH_CALUDE_min_value_sum_product_equality_condition_l1694_169446

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) ≥ 4 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) = 4 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_product_equality_condition_l1694_169446


namespace NUMINAMATH_CALUDE_exponent_calculations_l1694_169448

theorem exponent_calculations (x m : ℝ) (hx : x ≠ 0) (hm : m ≠ 0) :
  (x^7 / x^3 * x^4 = x^8) ∧ (m * m^3 + (-m^2)^3 / m^2 = 0) := by sorry

end NUMINAMATH_CALUDE_exponent_calculations_l1694_169448


namespace NUMINAMATH_CALUDE_red_item_count_l1694_169487

/-- Represents the number of items of a specific color in the box -/
structure ColorCount where
  hats : ℕ
  gloves : ℕ

/-- Represents the contents of the box -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount

/-- The maximum number of draws needed to guarantee a pair of each color -/
def max_draws (b : Box) : ℕ :=
  max (b.red.hats + b.red.gloves) (max (b.green.hats + b.green.gloves) (b.orange.hats + b.orange.gloves)) + 2

/-- The theorem stating that if it takes 66 draws to guarantee a pair of each color,
    given 23 green items and 11 orange items, then there must be 30 red items -/
theorem red_item_count (b : Box) 
  (h_green : b.green.hats + b.green.gloves = 23)
  (h_orange : b.orange.hats + b.orange.gloves = 11)
  (h_draws : max_draws b = 66) :
  b.red.hats + b.red.gloves = 30 := by
  sorry

end NUMINAMATH_CALUDE_red_item_count_l1694_169487


namespace NUMINAMATH_CALUDE_class_composition_l1694_169468

/-- Represents a pair of numbers reported by a student -/
structure ReportedPair :=
  (classmates : ℕ)
  (female_classmates : ℕ)

/-- Checks if a reported pair is valid given the actual numbers of boys and girls -/
def is_valid_report (report : ReportedPair) (boys girls : ℕ) : Prop :=
  (report.classmates = boys + girls - 1 ∧ (report.female_classmates = girls ∨ report.female_classmates = girls + 2 ∨ report.female_classmates = girls - 2)) ∨
  (report.female_classmates = girls ∧ (report.classmates = boys + girls - 1 + 2 ∨ report.classmates = boys + girls - 1 - 2))

theorem class_composition 
  (reports : List ReportedPair)
  (h1 : (12, 18) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h2 : (15, 15) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h3 : (11, 15) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h4 : ∀ r ∈ reports, is_valid_report r 13 16) :
  ∃ (boys girls : ℕ), boys = 13 ∧ girls = 16 ∧ 
    (∀ r ∈ reports, is_valid_report r boys girls) :=
by
  sorry

end NUMINAMATH_CALUDE_class_composition_l1694_169468


namespace NUMINAMATH_CALUDE_sequence_upper_bound_l1694_169498

theorem sequence_upper_bound 
  (a : ℕ → ℝ) 
  (h_nonneg : ∀ n, a n ≥ 0) 
  (h_decr : ∃ A > 0, ∀ m, a m - a (m + 1) ≥ A * (a m)^2) : 
  ∃ B > 0, ∀ n ≥ 1, a n ≤ B / n :=
sorry

end NUMINAMATH_CALUDE_sequence_upper_bound_l1694_169498


namespace NUMINAMATH_CALUDE_inequality_solution_l1694_169481

theorem inequality_solution (x : ℝ) : 
  (4 ≤ x^2 - 3*x - 6 ∧ x^2 - 3*x - 6 ≤ 2*x + 8) ↔ 
  ((5 ≤ x ∧ x ≤ 7) ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1694_169481


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l1694_169493

/-- The speed of the moon around the Earth in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.03

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the moon around the Earth in kilometers per hour -/
def moon_speed_km_per_hour : ℝ := moon_speed_km_per_sec * seconds_per_hour

theorem moon_speed_conversion :
  moon_speed_km_per_hour = 3708 := by sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l1694_169493


namespace NUMINAMATH_CALUDE_division_problem_l1694_169417

theorem division_problem (remainder quotient divisor dividend : ℕ) :
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  dividend = divisor * quotient + remainder →
  dividend = 86 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1694_169417


namespace NUMINAMATH_CALUDE_birds_on_fence_l1694_169445

theorem birds_on_fence (initial_storks : ℕ) (additional_storks : ℕ) (total_after : ℕ) 
  (h1 : initial_storks = 4)
  (h2 : additional_storks = 6)
  (h3 : total_after = 13) :
  ∃ initial_birds : ℕ, initial_birds + initial_storks + additional_storks = total_after ∧ initial_birds = 3 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1694_169445


namespace NUMINAMATH_CALUDE_sunghoon_scores_l1694_169482

theorem sunghoon_scores (korean math english : ℝ) 
  (h1 : korean / math = 1.2) 
  (h2 : math / english = 5/6) : 
  korean / english = 1 := by
  sorry

end NUMINAMATH_CALUDE_sunghoon_scores_l1694_169482


namespace NUMINAMATH_CALUDE_village_population_l1694_169456

theorem village_population (population_percentage : ℝ) (partial_population : ℕ) (total_population : ℕ) :
  population_percentage = 80 →
  partial_population = 64000 →
  (population_percentage / 100) * total_population = partial_population →
  total_population = 80000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1694_169456


namespace NUMINAMATH_CALUDE_remaining_distance_condition_l1694_169479

/-- The total distance between points A and B in kilometers -/
def total_distance : ℕ := 500

/-- Alpha's daily cycling distance in kilometers -/
def alpha_daily_distance : ℕ := 30

/-- Beta's cycling distance on active days in kilometers -/
def beta_active_day_distance : ℕ := 50

/-- The number of days after which the condition is met -/
def condition_day : ℕ := 15

/-- The remaining distance for Alpha after n days -/
def alpha_remaining (n : ℕ) : ℕ := total_distance - n * alpha_daily_distance

/-- The remaining distance for Beta after n days -/
def beta_remaining (n : ℕ) : ℕ := total_distance - n * (beta_active_day_distance / 2)

/-- Theorem stating that on the 15th day, Beta's remaining distance is twice Alpha's -/
theorem remaining_distance_condition :
  beta_remaining condition_day = 2 * alpha_remaining condition_day :=
sorry

end NUMINAMATH_CALUDE_remaining_distance_condition_l1694_169479


namespace NUMINAMATH_CALUDE_train_length_l1694_169450

theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 100 → speed_kmh = 90 → 
  crossing_time * (speed_kmh * (1000 / 3600)) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1694_169450


namespace NUMINAMATH_CALUDE_rice_division_l1694_169476

/-- Proves that dividing 25/4 pounds of rice equally among 4 containers results in 25 ounces per container. -/
theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (pound_to_ounce : ℕ) :
  total_weight = 25 / 4 →
  num_containers = 4 →
  pound_to_ounce = 16 →
  (total_weight / num_containers) * pound_to_ounce = 25 := by
  sorry

end NUMINAMATH_CALUDE_rice_division_l1694_169476


namespace NUMINAMATH_CALUDE_general_inequality_l1694_169491

theorem general_inequality (x : ℝ) (n : ℕ) (h : x > 0) (hn : n > 0) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_general_inequality_l1694_169491


namespace NUMINAMATH_CALUDE_w_share_is_375_l1694_169466

/-- A structure representing the distribution of money among four individuals -/
structure MoneyDistribution where
  total : ℝ
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  proportion_w : ℝ := 1
  proportion_x : ℝ := 6
  proportion_y : ℝ := 2
  proportion_z : ℝ := 4
  sum_proportions : ℝ := proportion_w + proportion_x + proportion_y + proportion_z
  proportional_distribution :
    w / proportion_w = x / proportion_x ∧
    x / proportion_x = y / proportion_y ∧
    y / proportion_y = z / proportion_z ∧
    w + x + y + z = total
  x_exceeds_y : x = y + 1500

theorem w_share_is_375 (d : MoneyDistribution) : d.w = 375 := by
  sorry

end NUMINAMATH_CALUDE_w_share_is_375_l1694_169466


namespace NUMINAMATH_CALUDE_veranda_area_l1694_169416

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) : 
  room_length = 21 →
  room_width = 12 →
  veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 148 := by
  sorry


end NUMINAMATH_CALUDE_veranda_area_l1694_169416


namespace NUMINAMATH_CALUDE_matrix_rank_two_l1694_169488

/-- Given an n×n matrix A where A_ij = i + j, prove that the rank of A is 2 -/
theorem matrix_rank_two (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
  (h : ∀ (i j : Fin n), A i j = (i.val + 1 : ℝ) + (j.val + 1 : ℝ)) :
  Matrix.rank A = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_rank_two_l1694_169488


namespace NUMINAMATH_CALUDE_mark_sprint_distance_l1694_169494

/-- The distance traveled by Mark given his sprint time and speed -/
theorem mark_sprint_distance (time : ℝ) (speed : ℝ) (h1 : time = 24.0) (h2 : speed = 6.0) :
  time * speed = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_mark_sprint_distance_l1694_169494


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1694_169407

theorem tan_alpha_plus_pi_fourth (x y : ℝ) (α : ℝ) : 
  (x < 0 ∧ y > 0) →  -- terminal side in second quadrant
  (3 * x + 4 * y = 0) →  -- m ⊥ OA
  (Real.tan α = -3/4) →  -- derived from m ⊥ OA
  Real.tan (α + π/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1694_169407


namespace NUMINAMATH_CALUDE_cubic_function_minimum_l1694_169486

theorem cubic_function_minimum (a b c : ℝ) : 
  let f := fun x => a * x^3 + b * x^2 + c * x - 34
  let f' := fun x => 3 * a * x^2 + 2 * b * x + c
  (∀ x, f' x ≤ 0 ↔ -2 ≤ x ∧ x ≤ 3) →
  (∃ x₀, ∀ x, f x ≥ f x₀) →
  f 3 = -115 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_minimum_l1694_169486


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l1694_169447

/-- The number of white balls in the box -/
def num_white : ℕ := 6

/-- The number of black balls in the box -/
def num_black : ℕ := 6

/-- The total number of balls in the box -/
def total_balls : ℕ := num_white + num_black

/-- The number of ways to arrange all balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_white

/-- The number of alternating color sequences -/
def alternating_sequences : ℕ := 2

/-- The probability of drawing balls with alternating colors -/
def alternating_probability : ℚ := alternating_sequences / total_arrangements

theorem alternating_draw_probability : alternating_probability = 1 / 462 := by
  sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l1694_169447


namespace NUMINAMATH_CALUDE_chess_team_boys_count_l1694_169404

theorem chess_team_boys_count 
  (total_members : ℕ) 
  (meeting_attendance : ℕ) 
  (h1 : total_members = 26)
  (h2 : meeting_attendance = 16)
  : ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + girls / 2 = meeting_attendance ∧
    boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_boys_count_l1694_169404


namespace NUMINAMATH_CALUDE_firm_employs_80_looms_l1694_169410

/-- Represents a textile manufacturing firm with looms -/
structure TextileFirm where
  totalSales : ℕ
  manufacturingExpenses : ℕ
  establishmentCharges : ℕ
  profitDecreaseOnBreakdown : ℕ

/-- Calculates the number of looms employed by the firm -/
def calculateLooms (firm : TextileFirm) : ℕ :=
  (firm.totalSales - firm.manufacturingExpenses) / firm.profitDecreaseOnBreakdown

/-- Theorem stating that the firm employs 80 looms -/
theorem firm_employs_80_looms (firm : TextileFirm) 
  (h1 : firm.totalSales = 500000)
  (h2 : firm.manufacturingExpenses = 150000)
  (h3 : firm.establishmentCharges = 75000)
  (h4 : firm.profitDecreaseOnBreakdown = 4375) :
  calculateLooms firm = 80 := by
  sorry

#eval calculateLooms { totalSales := 500000, 
                       manufacturingExpenses := 150000, 
                       establishmentCharges := 75000, 
                       profitDecreaseOnBreakdown := 4375 }

end NUMINAMATH_CALUDE_firm_employs_80_looms_l1694_169410


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_fifteen_l1694_169415

/-- The sum of the tens digit and the ones digit of (3 + 4)^15 is 7 -/
theorem sum_of_digits_of_seven_to_fifteen (n : ℕ) : n = (3 + 4)^15 → 
  (n / 10 % 10 + n % 10 = 7) := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_fifteen_l1694_169415


namespace NUMINAMATH_CALUDE_ellipse_b_squared_value_l1694_169413

/-- Given an ellipse and a hyperbola with coinciding foci, prove the value of b^2 for the ellipse -/
theorem ellipse_b_squared_value (b : ℝ) : 
  (∀ x y, x^2/25 + y^2/b^2 = 1 → x^2/169 - y^2/64 = 1/36) → 
  (∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 233/36) →
  b^2 = 667/36 := by
sorry

end NUMINAMATH_CALUDE_ellipse_b_squared_value_l1694_169413


namespace NUMINAMATH_CALUDE_M_intersect_N_l1694_169425

def M : Set Int := {m | -3 < m ∧ m < 2}
def N : Set Int := {n | -1 ≤ n ∧ n ≤ 3}

theorem M_intersect_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1694_169425


namespace NUMINAMATH_CALUDE_runner_stops_in_quarter_A_l1694_169429

def track_circumference : ℝ := 80
def distance_run : ℝ := 2000
def num_quarters : ℕ := 4

theorem runner_stops_in_quarter_A :
  ∀ (start_point : ℝ) (quarters : Fin num_quarters),
  start_point ∈ Set.Icc 0 track_circumference →
  ∃ (n : ℕ), distance_run = n * track_circumference + start_point :=
by sorry

end NUMINAMATH_CALUDE_runner_stops_in_quarter_A_l1694_169429


namespace NUMINAMATH_CALUDE_integral_comparison_l1694_169470

theorem integral_comparison : ∫ x in (0:ℝ)..1, x > ∫ x in (0:ℝ)..1, x^3 := by
  sorry

end NUMINAMATH_CALUDE_integral_comparison_l1694_169470


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l1694_169480

/-- Proves the ordering of y-coordinates for three points on an inverse proportion function -/
theorem inverse_proportion_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_pos : k > 0)
  (h_A : y₁ = k / (-3))
  (h_B : y₂ = k / (-2))
  (h_C : y₃ = k / 2) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l1694_169480


namespace NUMINAMATH_CALUDE_sum_of_f_values_l1694_169418

noncomputable def f (x : ℝ) : ℝ := (x * Real.exp x + x + 2) / (Real.exp x + 1) + Real.sin x

theorem sum_of_f_values : 
  f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l1694_169418


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1694_169408

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * Real.pi * r ^ 2 = 16 * Real.pi) →
  (4 / 3 * Real.pi * r ^ 3 = 32 / 3 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1694_169408


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1694_169405

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x - f y)^2 - 4 * x^2 * f y :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1694_169405


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l1694_169462

/-- The difference in miles traveled between two cyclists after 3 hours -/
theorem cyclist_distance_difference (carlos_start : ℝ) (carlos_total : ℝ) (dana_total : ℝ) : 
  carlos_start = 5 → 
  carlos_total = 50 → 
  dana_total = 40 → 
  carlos_total - dana_total = 10 := by
  sorry

#check cyclist_distance_difference

end NUMINAMATH_CALUDE_cyclist_distance_difference_l1694_169462


namespace NUMINAMATH_CALUDE_negation_of_inequality_statement_l1694_169457

theorem negation_of_inequality_statement :
  (¬ ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x) ↔
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 < Real.log x₀) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_inequality_statement_l1694_169457


namespace NUMINAMATH_CALUDE_john_needs_60_bags_l1694_169475

/-- Calculates the number of half-ton bags of horse food needed for a given number of horses, 
    feedings per day, pounds per feeding, and number of days. --/
def bags_needed (num_horses : ℕ) (feedings_per_day : ℕ) (pounds_per_feeding : ℕ) (days : ℕ) : ℕ :=
  let daily_food_per_horse := feedings_per_day * pounds_per_feeding
  let total_daily_food := daily_food_per_horse * num_horses
  let total_food := total_daily_food * days
  let bag_weight := 1000  -- half-ton in pounds
  total_food / bag_weight

/-- Theorem stating that John needs 60 bags of food for his horses over 60 days. --/
theorem john_needs_60_bags : 
  bags_needed 25 2 20 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_60_bags_l1694_169475


namespace NUMINAMATH_CALUDE_water_ratio_is_two_to_one_l1694_169436

/-- Represents the water usage scenario of a water tower and four neighborhoods --/
structure WaterUsage where
  total : ℕ
  first : ℕ
  fourth : ℕ
  third_excess : ℕ

/-- Calculates the ratio of water used by the second neighborhood to the first neighborhood --/
def water_ratio (w : WaterUsage) : ℚ :=
  let second := (w.total - w.first - w.fourth - (w.total - w.first - w.fourth - w.third_excess)) / 2
  second / w.first

/-- Theorem stating that given the specific conditions, the water ratio is 2:1 --/
theorem water_ratio_is_two_to_one (w : WaterUsage) 
  (h1 : w.total = 1200)
  (h2 : w.first = 150)
  (h3 : w.fourth = 350)
  (h4 : w.third_excess = 100) :
  water_ratio w = 2 := by
  sorry

#eval water_ratio { total := 1200, first := 150, fourth := 350, third_excess := 100 }

end NUMINAMATH_CALUDE_water_ratio_is_two_to_one_l1694_169436


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l1694_169471

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children) / (total_families - childless_families) = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l1694_169471


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l1694_169428

/-- Approximation of square root of 11 -/
def sqrt11_approx : ℝ := 3.31662

/-- Approximation of square root of 6 -/
def sqrt6_approx : ℝ := 2.44948

/-- The result we want to prove is close to the actual difference -/
def result : ℝ := 0.87

/-- Theorem stating that the difference between sqrt(11) and sqrt(6) is close to 0.87 -/
theorem sqrt_difference_approximation : |Real.sqrt 11 - Real.sqrt 6 - result| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l1694_169428


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l1694_169439

/-- Calculates the percentage of earnings left over after Dhoni's expenses --/
theorem dhoni_leftover_earnings (rent : ℝ) (utilities : ℝ) (groceries : ℝ) (transportation : ℝ)
  (h_rent : rent = 25)
  (h_utilities : utilities = 15)
  (h_groceries : groceries = 20)
  (h_transportation : transportation = 12) :
  100 - (rent + (rent - rent * 0.1) + utilities + groceries + transportation) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l1694_169439


namespace NUMINAMATH_CALUDE_equation_solution_l1694_169459

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1694_169459


namespace NUMINAMATH_CALUDE_area_ratio_l1694_169473

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (30, 50, 54)

-- Define points D and E
def point_D (t : Triangle) : ℝ × ℝ := sorry
def point_E (t : Triangle) : ℝ × ℝ := sorry

-- Define the distances AD and AE
def dist_AD (t : Triangle) : ℝ := 21
def dist_AE (t : Triangle) : ℝ := 18

-- Define the areas of triangle ADE and quadrilateral BCED
def area_ADE (t : Triangle) : ℝ := sorry
def area_BCED (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_ratio (t : Triangle) :
  area_ADE t / area_BCED t = 49 / 51 := by sorry

end NUMINAMATH_CALUDE_area_ratio_l1694_169473


namespace NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l1694_169465

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l1694_169465


namespace NUMINAMATH_CALUDE_angle_a_is_sixty_degrees_l1694_169409

/-- In a triangle ABC, if the sum of angles B and C is twice angle A, then angle A is 60 degrees. -/
theorem angle_a_is_sixty_degrees (A B C : ℝ) (h1 : A + B + C = 180) (h2 : B + C = 2 * A) : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_a_is_sixty_degrees_l1694_169409


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l1694_169478

theorem complex_arithmetic_result :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -3
  let T : ℂ := 2*I
  let U : ℂ := 1 + 5*I
  2*B - Q + 3*T + U = 10 + 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l1694_169478


namespace NUMINAMATH_CALUDE_binary_1011_to_decimal_l1694_169454

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1011 -/
def binary_1011 : List Bool := [true, true, false, true]

/-- Theorem stating that the decimal representation of binary 1011 is 11 -/
theorem binary_1011_to_decimal :
  binary_to_decimal binary_1011 = 11 := by sorry

end NUMINAMATH_CALUDE_binary_1011_to_decimal_l1694_169454


namespace NUMINAMATH_CALUDE_geometric_sequence_cos_ratio_l1694_169427

open Real

/-- Given an arithmetic sequence {a_n} with first term a₁ and common difference d,
    where 0 < d < 2π, if {cos a_n} forms a geometric sequence,
    then the common ratio of {cos a_n} is -1. -/
theorem geometric_sequence_cos_ratio
  (a₁ : ℝ) (d : ℝ) (h_d : 0 < d ∧ d < 2 * π)
  (h_geom : ∀ n : ℕ, n ≥ 1 → cos (a₁ + n * d) / cos (a₁ + (n - 1) * d) =
                           cos (a₁ + d) / cos a₁) :
  ∀ n : ℕ, n ≥ 1 → cos (a₁ + (n + 1) * d) / cos (a₁ + n * d) = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_cos_ratio_l1694_169427


namespace NUMINAMATH_CALUDE_division_result_l1694_169440

theorem division_result : 75 / 0.05 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1694_169440


namespace NUMINAMATH_CALUDE_dogs_not_liking_either_l1694_169485

theorem dogs_not_liking_either (total : ℕ) (watermelon : ℕ) (salmon : ℕ) (both : ℕ)
  (h1 : total = 75)
  (h2 : watermelon = 12)
  (h3 : salmon = 55)
  (h4 : both = 7) :
  total - (watermelon + salmon - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_either_l1694_169485


namespace NUMINAMATH_CALUDE_max_net_revenue_l1694_169430

/-- Represents the net revenue function for a movie theater --/
def net_revenue (x : ℕ) : ℤ :=
  if x ≤ 10 then 1000 * x - 5750
  else -30 * x * x + 1300 * x - 5750

/-- Theorem stating the maximum net revenue and optimal ticket price --/
theorem max_net_revenue :
  ∃ (max_revenue : ℕ) (optimal_price : ℕ),
    max_revenue = 8830 ∧
    optimal_price = 22 ∧
    (∀ (x : ℕ), x ≥ 6 → x ≤ 38 → net_revenue x ≤ net_revenue optimal_price) :=
by sorry

end NUMINAMATH_CALUDE_max_net_revenue_l1694_169430


namespace NUMINAMATH_CALUDE_part_one_part_two_l1694_169460

/-- Definition of the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

/-- Theorem for the first part of the problem -/
theorem part_one (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x ∈ Set.Ioo (-1) 1, f a (1 - m) + f a (1 - m^2) < 0) :
  1 < m ∧ m < Real.sqrt 2 := by sorry

/-- Theorem for the second part of the problem -/
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x < 2, f a x - 4 < 0) :
  a ∈ Set.Ioo (2 - Real.sqrt 3) 1 ∪ Set.Ioo 1 (2 + Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1694_169460


namespace NUMINAMATH_CALUDE_discretionary_income_ratio_l1694_169489

/-- Represents Jill's financial situation --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFundPercentage : ℝ
  savingsPercentage : ℝ
  socializingPercentage : ℝ
  remainingAmount : ℝ

/-- Theorem stating the ratio of discretionary income to net salary --/
theorem discretionary_income_ratio (j : JillFinances) 
  (h1 : j.netSalary = 3700)
  (h2 : j.vacationFundPercentage = 0.3)
  (h3 : j.savingsPercentage = 0.2)
  (h4 : j.socializingPercentage = 0.35)
  (h5 : j.remainingAmount = 111)
  (h6 : j.discretionaryIncome * (1 - (j.vacationFundPercentage + j.savingsPercentage + j.socializingPercentage)) = j.remainingAmount) :
  j.discretionaryIncome / j.netSalary = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_discretionary_income_ratio_l1694_169489


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_lcm_l1694_169435

theorem product_of_numbers_with_given_sum_and_lcm :
  ∃ (a b : ℕ+), 
    (a + b : ℕ) = 210 ∧ 
    Nat.lcm a b = 1547 → 
    (a * b : ℕ) = 10829 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_lcm_l1694_169435


namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1694_169422

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_6_8_10 :
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 1 1 2 ∧
  ¬ is_pythagorean_triple 1 2 2 ∧
  ¬ is_pythagorean_triple 5 12 15 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1694_169422


namespace NUMINAMATH_CALUDE_second_pile_magazines_l1694_169461

/-- A sequence of 5 terms representing the number of magazines in each pile. -/
def MagazineSequence : Type := Fin 5 → ℕ

/-- The properties of the magazine sequence based on the given information. -/
def IsValidMagazineSequence (s : MagazineSequence) : Prop :=
  s 0 = 3 ∧ s 2 = 6 ∧ s 3 = 9 ∧ s 4 = 13 ∧
  ∀ i : Fin 4, s (i + 1) - s i = s 1 - s 0

/-- Theorem stating that for any valid magazine sequence, the second term (index 1) must be 3. -/
theorem second_pile_magazines (s : MagazineSequence) 
  (h : IsValidMagazineSequence s) : s 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_pile_magazines_l1694_169461


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1694_169451

theorem arithmetic_sequence_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 8 →
  aₙ = 347 →
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ) / 2) % 8 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1694_169451


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1694_169402

theorem reciprocal_of_sum : (1 / (1/3 + 3/4)) = 12/13 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1694_169402


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1694_169484

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : i * (1 + a * i) = 1 + b * i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1694_169484


namespace NUMINAMATH_CALUDE_pave_hall_l1694_169431

/-- Calculates the number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).ceil.toNat

/-- Theorem stating that 5400 stones are required to pave the given hall -/
theorem pave_hall : stones_required 36 15 2 5 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_pave_hall_l1694_169431


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1694_169444

theorem cost_price_calculation (C : ℝ) : C = 400 :=
  let SP := 0.8 * C
  have selling_price : SP = 0.8 * C := by sorry
  have increased_price : SP + 100 = 1.05 * C := by sorry
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1694_169444


namespace NUMINAMATH_CALUDE_vertex_angle_of_identical_cones_l1694_169496

/-- A cone with vertex A -/
structure Cone where
  vertexAngle : ℝ

/-- The configuration of four cones as described in the problem -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  cone4 : Cone
  cone1_eq_cone2 : cone1 = cone2
  cone3_angle : cone3.vertexAngle = π / 3
  cone4_angle : cone4.vertexAngle = 5 * π / 6
  externally_tangent : Bool  -- Represents that cone1, cone2, and cone3 are externally tangent
  internally_tangent : Bool  -- Represents that cone1, cone2, and cone3 are internally tangent to cone4

/-- The theorem stating the vertex angle of the first two cones -/
theorem vertex_angle_of_identical_cones (config : ConeConfiguration) :
  config.cone1.vertexAngle = 2 * Real.arctan (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_of_identical_cones_l1694_169496


namespace NUMINAMATH_CALUDE_maria_workday_end_l1694_169449

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def Time.add (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes
  let newHours := (t.hours + d.hours + totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

def Duration.add (d1 d2 : Duration) : Duration :=
  let totalMinutes := d1.minutes + d2.minutes + (d1.hours + d2.hours) * 60
  ⟨totalMinutes / 60, totalMinutes % 60⟩

def workDay (start : Time) (workDuration : Duration) (lunchStart : Time) (lunchDuration : Duration) (breakStart : Time) (breakDuration : Duration) : Time :=
  let lunchEnd := lunchStart.add lunchDuration
  let breakEnd := breakStart.add breakDuration
  let totalBreakDuration := Duration.add lunchDuration breakDuration
  start.add (Duration.add workDuration totalBreakDuration)

theorem maria_workday_end :
  let start : Time := ⟨8, 0, by sorry, by sorry⟩
  let workDuration : Duration := ⟨8, 0⟩
  let lunchStart : Time := ⟨13, 0, by sorry, by sorry⟩
  let lunchDuration : Duration := ⟨1, 0⟩
  let breakStart : Time := ⟨15, 30, by sorry, by sorry⟩
  let breakDuration : Duration := ⟨0, 15⟩
  let endTime : Time := workDay start workDuration lunchStart lunchDuration breakStart breakDuration
  endTime = ⟨18, 0, by sorry, by sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_maria_workday_end_l1694_169449


namespace NUMINAMATH_CALUDE_wheel_radius_l1694_169455

/-- The radius of a wheel given its circumference and number of revolutions --/
theorem wheel_radius (distance : ℝ) (revolutions : ℕ) (h : distance = 760.57 ∧ revolutions = 500) :
  ∃ (radius : ℝ), abs (radius - 0.242) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_wheel_radius_l1694_169455


namespace NUMINAMATH_CALUDE_profit_difference_theorem_l1694_169483

/-- Represents the profit distribution for a business partnership. -/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C. -/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating the difference between A's and C's profit shares. -/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 4000) :
  profit_difference pd = 1600 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_theorem_l1694_169483


namespace NUMINAMATH_CALUDE_twenty_nine_is_perfect_factorization_condition_equation_solution_perfect_number_condition_l1694_169414

-- Definition of perfect number
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Statement 1
theorem twenty_nine_is_perfect : is_perfect_number 29 := by sorry

-- Statement 2
theorem factorization_condition (m n : ℝ) :
  (∀ x : ℝ, x^2 - 6*x + 5 = (x - m)^2 + n) → m*n = -12 := by sorry

-- Statement 3
theorem equation_solution :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 5 = 0 → x + y = -1 := by sorry

-- Statement 4
theorem perfect_number_condition :
  ∃ k : ℤ, ∀ x y : ℤ, ∃ p q : ℤ, x^2 + 4*y^2 + 4*x - 12*y + k = p^2 + q^2 := by sorry

end NUMINAMATH_CALUDE_twenty_nine_is_perfect_factorization_condition_equation_solution_perfect_number_condition_l1694_169414


namespace NUMINAMATH_CALUDE_race_time_difference_l1694_169424

/-- Represents the time difference in minutes between two runners finishing a race -/
def timeDifference (malcolmSpeed Joshua : ℝ) (raceDistance : ℝ) : ℝ :=
  raceDistance * Joshua - raceDistance * malcolmSpeed

theorem race_time_difference :
  let malcolmSpeed := 6
  let Joshua := 8
  let raceDistance := 10
  timeDifference malcolmSpeed Joshua raceDistance = 20 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1694_169424


namespace NUMINAMATH_CALUDE_purple_candies_count_l1694_169419

/-- The number of purple candies in a box of rainbow nerds -/
def purple_candies : ℕ := 10

/-- The number of yellow candies in a box of rainbow nerds -/
def yellow_candies : ℕ := purple_candies + 4

/-- The number of green candies in a box of rainbow nerds -/
def green_candies : ℕ := yellow_candies - 2

/-- The total number of candies in the box -/
def total_candies : ℕ := 36

/-- Theorem stating that the number of purple candies is 10 -/
theorem purple_candies_count : 
  purple_candies = 10 ∧ 
  yellow_candies = purple_candies + 4 ∧ 
  green_candies = yellow_candies - 2 ∧ 
  purple_candies + yellow_candies + green_candies = total_candies :=
by sorry

end NUMINAMATH_CALUDE_purple_candies_count_l1694_169419


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1694_169400

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b ∧ focus ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}}

-- Define the intersection points A and B
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ p ∈ line_through_focus m b}

-- State the theorem
theorem parabola_intersection_length 
  (m b : ℝ) 
  (A B : ℝ × ℝ) 
  (h_A : A ∈ intersection_points m b) 
  (h_B : B ∈ intersection_points m b) 
  (h_midpoint : (A.1 + B.1) / 2 = 3) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1694_169400


namespace NUMINAMATH_CALUDE_square_of_difference_l1694_169441

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1694_169441


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1694_169433

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1694_169433


namespace NUMINAMATH_CALUDE_max_value_of_a_l1694_169474

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a) →
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1694_169474


namespace NUMINAMATH_CALUDE_no_valid_grid_l1694_169437

/-- Represents a 3x3 grid with elements from 1 to 4 -/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List (Fin 4)) : Prop :=
  l.Nodup

/-- Checks if a row in the grid contains distinct elements -/
def rowDistinct (g : Grid) (i : Fin 3) : Prop :=
  allDistinct [g i 0, g i 1, g i 2]

/-- Checks if a column in the grid contains distinct elements -/
def colDistinct (g : Grid) (j : Fin 3) : Prop :=
  allDistinct [g 0 j, g 1 j, g 2 j]

/-- Checks if the main diagonal contains distinct elements -/
def mainDiagDistinct (g : Grid) : Prop :=
  allDistinct [g 0 0, g 1 1, g 2 2]

/-- Checks if the anti-diagonal contains distinct elements -/
def antiDiagDistinct (g : Grid) : Prop :=
  allDistinct [g 0 2, g 1 1, g 2 0]

/-- A grid is valid if all rows, columns, and diagonals contain distinct elements -/
def validGrid (g : Grid) : Prop :=
  (∀ i, rowDistinct g i) ∧
  (∀ j, colDistinct g j) ∧
  mainDiagDistinct g ∧
  antiDiagDistinct g

theorem no_valid_grid : ¬∃ g : Grid, validGrid g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_l1694_169437


namespace NUMINAMATH_CALUDE_finite_zero_additions_l1694_169406

/-- Represents the state of the board at any given time -/
def BoardState := List ℕ

/-- The process of updating the board -/
def update_board (a b : ℕ) (state : BoardState) : BoardState :=
  sorry

/-- Predicate to check if we need to add two zeros -/
def need_zeros (state : BoardState) : Prop :=
  sorry

/-- The main theorem statement -/
theorem finite_zero_additions (a b : ℕ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    let state := (update_board a b)^[n] []
    ¬(need_zeros state) :=
  sorry

end NUMINAMATH_CALUDE_finite_zero_additions_l1694_169406


namespace NUMINAMATH_CALUDE_complex_fraction_real_condition_l1694_169463

theorem complex_fraction_real_condition (a : ℝ) : 
  (((1 : ℂ) + 2 * I) / (a + I)).im = 0 ↔ a = (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_real_condition_l1694_169463


namespace NUMINAMATH_CALUDE_geometric_sequence_logarithm_l1694_169499

theorem geometric_sequence_logarithm (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = -Real.sqrt 2 * a n) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_logarithm_l1694_169499


namespace NUMINAMATH_CALUDE_no_natural_squares_diff_2014_l1694_169438

theorem no_natural_squares_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_diff_2014_l1694_169438


namespace NUMINAMATH_CALUDE_min_vertical_distance_l1694_169492

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 - 3*x - 5

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), vertical_distance x₀ ≤ vertical_distance x ∧ vertical_distance x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l1694_169492


namespace NUMINAMATH_CALUDE_min_value_theorem_l1694_169469

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b = 5*a*b → 3*x + 4*y ≤ 3*a + 4*b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c + 3*d = 5*c*d ∧ 3*c + 4*d = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1694_169469


namespace NUMINAMATH_CALUDE_expression_approximation_l1694_169412

theorem expression_approximation : 
  let x := Real.sqrt 1.1 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49
  ∃ ε > 0, ε < 0.00005 ∧ |x - 2.8793| < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_approximation_l1694_169412


namespace NUMINAMATH_CALUDE_sum_upper_bound_l1694_169442

/-- Given positive real numbers x and y satisfying 2x + 8y - xy = 0,
    the sum x + y is always less than or equal to 18. -/
theorem sum_upper_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 2 * x + 8 * y - x * y = 0) : 
  x + y ≤ 18 := by
sorry

end NUMINAMATH_CALUDE_sum_upper_bound_l1694_169442


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1694_169426

def M : Set ℝ := {x | |x| < 3}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem set_intersection_theorem : M ∩ N = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1694_169426


namespace NUMINAMATH_CALUDE_multiply_polynomials_l1694_169472

theorem multiply_polynomials (x : ℝ) :
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 + 12*x^4 - 144*x^2 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l1694_169472


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1694_169420

theorem bicycle_cost_price (final_price : ℝ) (profit_percentage : ℝ) : 
  final_price = 225 →
  profit_percentage = 25 →
  ∃ (original_cost : ℝ), 
    original_cost * (1 + profit_percentage / 100)^2 = final_price ∧
    original_cost = 144 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l1694_169420


namespace NUMINAMATH_CALUDE_completing_square_form_l1694_169423

theorem completing_square_form (x : ℝ) :
  x^2 - 2*x - 1 = 0 ↔ (x - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_form_l1694_169423


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l1694_169477

/-- Given David's marks in four subjects and the average across five subjects, 
    prove that his marks in Chemistry must be 87. -/
theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 86) 
  (h2 : mathematics = 85) 
  (h3 : physics = 92) 
  (h4 : biology = 95) 
  (h5 : average = 89) 
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) : 
  chemistry = 87 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l1694_169477


namespace NUMINAMATH_CALUDE_epidemic_supplies_theorem_l1694_169421

-- Define the prices of type A and B supplies
def price_A : ℕ := 16
def price_B : ℕ := 4

-- Define the conditions
axiom condition1 : 60 * price_A + 45 * price_B = 1140
axiom condition2 : 45 * price_A + 30 * price_B = 840

-- Define the total units and budget
def total_units : ℕ := 600
def total_budget : ℕ := 8000

-- Define the function to calculate the maximum number of type A units
def max_type_A : ℕ :=
  (total_budget - price_B * total_units) / (price_A - price_B)

-- Theorem to prove
theorem epidemic_supplies_theorem :
  price_A = 16 ∧ price_B = 4 ∧ max_type_A = 466 :=
sorry

end NUMINAMATH_CALUDE_epidemic_supplies_theorem_l1694_169421


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1694_169495

/-- Represents a pyramid with a regular hexagonal base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side_length : ℝ)

/-- Represents a cube placed inside the pyramid -/
structure Cube :=
  (side_length : ℝ)

/-- Calculate the volume of a cube -/
def cube_volume (c : Cube) : ℝ := c.side_length ^ 3

/-- The configuration of the pyramid and cube as described in the problem -/
def pyramid_cube_configuration (p : Pyramid) (c : Cube) : Prop :=
  p.base_side_length = 2 ∧
  c.side_length = 2 * Real.sqrt 3 / 9

/-- The theorem stating the volume of the cube in the given configuration -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : Cube) :
  pyramid_cube_configuration p c →
  cube_volume c = 8 * Real.sqrt 3 / 243 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1694_169495


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_iff_m_eq_two_l1694_169403

/-- A system of linear equations in x and y with parameter m -/
structure LinearSystem (m : ℝ) where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  h1 : ∀ x y, eq1 x y = m * x + 4 * y - (m + 2)
  h2 : ∀ x y, eq2 x y = x + m * y - m

/-- The system has infinitely many solutions -/
def HasInfinitelySolutions (sys : LinearSystem m) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ sys.eq1 x₁ y₁ = 0 ∧ sys.eq2 x₁ y₁ = 0 ∧ sys.eq1 x₂ y₂ = 0 ∧ sys.eq2 x₂ y₂ = 0

/-- The main theorem: the system has infinitely many solutions iff m = 2 -/
theorem infinitely_many_solutions_iff_m_eq_two (m : ℝ) (sys : LinearSystem m) :
  HasInfinitelySolutions sys ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_iff_m_eq_two_l1694_169403


namespace NUMINAMATH_CALUDE_total_cash_realized_l1694_169434

/-- Calculates the cash realized from selling a stock -/
def cashRealized (value : ℝ) (returnRate : ℝ) (brokerageFeeRate : ℝ) : ℝ :=
  let grossValue := value * (1 + returnRate)
  grossValue * (1 - brokerageFeeRate)

/-- Theorem: The total cash realized from selling all three stocks is $65,120.75 -/
theorem total_cash_realized :
  let stockA := cashRealized 10000 0.14 0.0025
  let stockB := cashRealized 20000 0.10 0.005
  let stockC := cashRealized 30000 0.07 0.0075
  stockA + stockB + stockC = 65120.75 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_realized_l1694_169434


namespace NUMINAMATH_CALUDE_diplomats_speaking_french_l1694_169458

theorem diplomats_speaking_french (total : ℕ) (not_russian : ℕ) (neither : ℕ) (both : ℕ) :
  total = 100 →
  not_russian = 32 →
  neither = 20 →
  both = 10 →
  ∃ french : ℕ, french = 22 ∧ french = total - not_russian + both :=
by sorry

end NUMINAMATH_CALUDE_diplomats_speaking_french_l1694_169458


namespace NUMINAMATH_CALUDE_power_equations_correctness_l1694_169452

theorem power_equations_correctness :
  (∃ (correct : Finset (Fin 4)) (incorrect : Finset (Fin 4)), 
    correct.card = 2 ∧ 
    incorrect.card = 2 ∧ 
    correct ∩ incorrect = ∅ ∧
    correct ∪ incorrect = Finset.univ ∧
    (∀ i ∈ correct, 
      (i = 0 → ∀ x : ℝ, (x^4)^4 = x^8) ∨
      (i = 1 → ∀ y : ℝ, ((y^2)^2)^2 = y^8) ∨
      (i = 2 → ∀ y : ℝ, (-y^2)^3 = y^6) ∨
      (i = 3 → ∀ x : ℝ, ((-x)^3)^2 = x^6)) ∧
    (∀ i ∈ incorrect, 
      (i = 0 → ∃ x : ℝ, (x^4)^4 ≠ x^8) ∨
      (i = 1 → ∃ y : ℝ, ((y^2)^2)^2 ≠ y^8) ∨
      (i = 2 → ∃ y : ℝ, (-y^2)^3 ≠ y^6) ∨
      (i = 3 → ∃ x : ℝ, ((-x)^3)^2 ≠ x^6))) :=
by sorry

end NUMINAMATH_CALUDE_power_equations_correctness_l1694_169452


namespace NUMINAMATH_CALUDE_number_of_selection_schemes_l1694_169467

/-- The number of male teachers -/
def num_male : ℕ := 5

/-- The number of female teachers -/
def num_female : ℕ := 4

/-- The total number of teachers -/
def total_teachers : ℕ := num_male + num_female

/-- The number of teachers to be selected -/
def teachers_to_select : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating the number of valid selection schemes -/
theorem number_of_selection_schemes : 
  permutations total_teachers teachers_to_select - 
  (permutations num_male teachers_to_select + 
   permutations num_female teachers_to_select) = 420 := by
  sorry

end NUMINAMATH_CALUDE_number_of_selection_schemes_l1694_169467


namespace NUMINAMATH_CALUDE_machines_needed_for_multiple_production_l1694_169432

/-- Given that 4 machines produce x units in 6 days, prove that 4m machines are needed to produce m*x units in 6 days, where all machines work at the same constant rate. -/
theorem machines_needed_for_multiple_production 
  (x : ℝ) (m : ℝ) (rate : ℝ) (h1 : x > 0) (h2 : m > 0) (h3 : rate > 0) :
  4 * rate * 6 = x → (4 * m) * rate * 6 = m * x :=
by
  sorry

#check machines_needed_for_multiple_production

end NUMINAMATH_CALUDE_machines_needed_for_multiple_production_l1694_169432


namespace NUMINAMATH_CALUDE_greatest_rope_piece_length_l1694_169464

theorem greatest_rope_piece_length : Nat.gcd 48 (Nat.gcd 60 72) = 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_rope_piece_length_l1694_169464


namespace NUMINAMATH_CALUDE_conic_sections_decomposition_decomposition_into_ellipse_and_hyperbola_l1694_169401

/-- The equation y^4 - 9x^4 = 3y^2 - 1 represents two conic sections -/
theorem conic_sections_decomposition (x y : ℝ) :
  y^4 - 9*x^4 = 3*y^2 - 1 ↔
  ((y^2 - 3/2 = 3*x^2 + Real.sqrt 5/2) ∨ (y^2 - 3/2 = -(3*x^2 + Real.sqrt 5/2))) :=
by sorry

/-- The first equation represents an ellipse -/
def is_ellipse (x y : ℝ) : Prop :=
  y^2 - 3/2 = 3*x^2 + Real.sqrt 5/2

/-- The second equation represents a hyperbola -/
def is_hyperbola (x y : ℝ) : Prop :=
  y^2 - 3/2 = -(3*x^2 + Real.sqrt 5/2)

/-- The original equation decomposes into an ellipse and a hyperbola -/
theorem decomposition_into_ellipse_and_hyperbola (x y : ℝ) :
  y^4 - 9*x^4 = 3*y^2 - 1 ↔ (is_ellipse x y ∨ is_hyperbola x y) :=
by sorry

end NUMINAMATH_CALUDE_conic_sections_decomposition_decomposition_into_ellipse_and_hyperbola_l1694_169401


namespace NUMINAMATH_CALUDE_walking_speed_difference_l1694_169490

theorem walking_speed_difference (child_distance child_time elderly_distance elderly_time : ℝ) :
  child_distance = 15 ∧ 
  child_time = 3.5 ∧ 
  elderly_distance = 10 ∧ 
  elderly_time = 4 → 
  (elderly_time * 60 / elderly_distance) - (child_time * 60 / child_distance) = 10 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_difference_l1694_169490


namespace NUMINAMATH_CALUDE_fraction_value_l1694_169411

theorem fraction_value (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1694_169411


namespace NUMINAMATH_CALUDE_tenth_observation_value_l1694_169497

def average_of_9 : ℝ := 15.3
def new_average : ℝ := average_of_9 - 1.7
def num_observations : ℕ := 9

theorem tenth_observation_value :
  let sum_9 := average_of_9 * num_observations
  let sum_10 := new_average * (num_observations + 1)
  sum_10 - sum_9 = -1.7 := by sorry

end NUMINAMATH_CALUDE_tenth_observation_value_l1694_169497
