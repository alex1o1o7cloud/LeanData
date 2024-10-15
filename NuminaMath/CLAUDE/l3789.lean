import Mathlib

namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l3789_378949

theorem gcd_power_minus_one (k : ℤ) : Int.gcd (k^1024 - 1) (k^1035 - 1) = k - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l3789_378949


namespace NUMINAMATH_CALUDE_orchid_count_l3789_378982

/-- Time in minutes to paint each type of flower or vine -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def vine_time : ℕ := 2

/-- Number of each type of flower or vine painted -/
def lily_count : ℕ := 17
def rose_count : ℕ := 10
def vine_count : ℕ := 20

/-- Total time spent painting -/
def total_time : ℕ := 213

/-- Theorem stating the number of orchids painted -/
theorem orchid_count : 
  ∃ (x : ℕ), x * orchid_time = total_time - (lily_count * lily_time + rose_count * rose_time + vine_count * vine_time) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_orchid_count_l3789_378982


namespace NUMINAMATH_CALUDE_terminal_side_of_negative_400_degrees_l3789_378981

/-- The quadrant of an angle in degrees -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Normalizes an angle to the range [0, 360) -/
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

/-- Determines the quadrant of a normalized angle -/
def quadrantOfNormalizedAngle (angle : Int) : Quadrant :=
  if 0 ≤ angle ∧ angle < 90 then Quadrant.first
  else if 90 ≤ angle ∧ angle < 180 then Quadrant.second
  else if 180 ≤ angle ∧ angle < 270 then Quadrant.third
  else Quadrant.fourth

/-- Determines the quadrant of any angle -/
def quadrantOfAngle (angle : Int) : Quadrant :=
  quadrantOfNormalizedAngle (normalizeAngle angle)

theorem terminal_side_of_negative_400_degrees :
  quadrantOfAngle (-400) = Quadrant.fourth := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_of_negative_400_degrees_l3789_378981


namespace NUMINAMATH_CALUDE_cyclist_journey_l3789_378991

theorem cyclist_journey 
  (v : ℝ) -- original speed in mph
  (t : ℝ) -- original time in hours
  (d : ℝ) -- distance in miles
  (h₁ : d = v * t) -- distance = speed * time
  (h₂ : d = (v + 1/3) * (3/4 * t)) -- increased speed condition
  (h₃ : d = (v - 1/3) * (t + 3/2)) -- decreased speed condition
  : v = 1 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_journey_l3789_378991


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3789_378920

theorem cubic_equation_solution (x : ℚ) : (5*x - 2)^3 + 125 = 0 ↔ x = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3789_378920


namespace NUMINAMATH_CALUDE_bomb_defusal_probability_l3789_378908

theorem bomb_defusal_probability :
  let n : ℕ := 4  -- Total number of wires
  let k : ℕ := 2  -- Number of wires that need to be cut
  let total_combinations : ℕ := n.choose k  -- Total number of possible combinations
  let successful_combinations : ℕ := 1  -- Number of successful combinations
  (successful_combinations : ℚ) / total_combinations = 1 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bomb_defusal_probability_l3789_378908


namespace NUMINAMATH_CALUDE_abs_min_value_min_value_at_two_unique_min_value_l3789_378924

theorem abs_min_value (x : ℝ) : |x - 2| + 3 ≥ 3 := by sorry

theorem min_value_at_two : ∃ (x : ℝ), |x - 2| + 3 = 3 := by sorry

theorem unique_min_value (x : ℝ) : |x - 2| + 3 = 3 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_abs_min_value_min_value_at_two_unique_min_value_l3789_378924


namespace NUMINAMATH_CALUDE_roots_transformation_l3789_378986

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + r₁ + 6 = 0) → 
  (r₂^3 - 4*r₂^2 + r₂ + 6 = 0) → 
  (r₃^3 - 4*r₃^2 + r₃ + 6 = 0) → 
  ∀ x, (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃) = x^3 - 12*x^2 + 9*x + 162 :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l3789_378986


namespace NUMINAMATH_CALUDE_number_of_boats_l3789_378909

/-- Given a lake with boats, where each boat has 3 people and there are 15 people on boats,
    prove that the number of boats is 5. -/
theorem number_of_boats (people_per_boat : ℕ) (total_people : ℕ) (num_boats : ℕ) :
  people_per_boat = 3 →
  total_people = 15 →
  num_boats * people_per_boat = total_people →
  num_boats = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_boats_l3789_378909


namespace NUMINAMATH_CALUDE_probability_green_is_81_160_l3789_378948

structure Container where
  red : ℕ
  green : ℕ

def containerA : Container := ⟨3, 5⟩
def containerB : Container := ⟨5, 5⟩
def containerC : Container := ⟨7, 3⟩
def containerD : Container := ⟨4, 6⟩

def containers : List Container := [containerA, containerB, containerC, containerD]

def probabilityGreenFromContainer (c : Container) : ℚ :=
  c.green / (c.red + c.green)

def probabilityGreen : ℚ :=
  (1 / containers.length) * (containers.map probabilityGreenFromContainer).sum

theorem probability_green_is_81_160 : probabilityGreen = 81 / 160 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_is_81_160_l3789_378948


namespace NUMINAMATH_CALUDE_point_alignment_implies_m_value_l3789_378923

/-- Three points lie on the same straight line if and only if 
    the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = (y₃ - y₁) / (x₃ - x₁)

theorem point_alignment_implies_m_value :
  ∀ m : ℝ, collinear 1 (-2) 3 4 6 (m/3) → m = 39 := by
  sorry


end NUMINAMATH_CALUDE_point_alignment_implies_m_value_l3789_378923


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l3789_378940

theorem fraction_equality_solution : ∃! x : ℝ, (4 + x) / (6 + x) = (2 + x) / (3 + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l3789_378940


namespace NUMINAMATH_CALUDE_erica_saw_three_warthogs_l3789_378966

/-- Represents the number of animals Erica saw on each day of her safari --/
structure SafariCount where
  saturday : Nat
  sunday : Nat
  monday_rhinos : Nat
  monday_warthogs : Nat

/-- The total number of animals seen during the safari --/
def total_animals : Nat := 20

/-- The number of animals Erica saw on Saturday --/
def saturday_count : Nat := 3 + 2

/-- The number of animals Erica saw on Sunday --/
def sunday_count : Nat := 2 + 5

/-- The number of rhinos Erica saw on Monday --/
def monday_rhinos : Nat := 5

/-- Theorem stating that Erica saw 3 warthogs on Monday --/
theorem erica_saw_three_warthogs (safari : SafariCount) :
  safari.saturday = saturday_count →
  safari.sunday = sunday_count →
  safari.monday_rhinos = monday_rhinos →
  safari.saturday + safari.sunday + safari.monday_rhinos + safari.monday_warthogs = total_animals →
  safari.monday_warthogs = 3 := by
  sorry


end NUMINAMATH_CALUDE_erica_saw_three_warthogs_l3789_378966


namespace NUMINAMATH_CALUDE_peanut_distribution_theorem_l3789_378959

/-- Represents the distribution of peanuts among three people -/
structure PeanutDistribution where
  alex : ℕ
  betty : ℕ
  charlie : ℕ

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  ∃ d : ℤ, b = a + d ∧ c = b + d

/-- The main theorem about the peanut distribution -/
theorem peanut_distribution_theorem (init : PeanutDistribution) 
  (h_total : init.alex + init.betty + init.charlie = 444)
  (h_order : init.alex < init.betty ∧ init.betty < init.charlie)
  (h_geometric : is_geometric_progression init.alex init.betty init.charlie)
  (final : PeanutDistribution)
  (h_eating : final.alex = init.alex - 5 ∧ final.betty = init.betty - 9 ∧ final.charlie = init.charlie - 25)
  (h_arithmetic : is_arithmetic_progression final.alex final.betty final.charlie) :
  init.alex = 108 := by
  sorry

end NUMINAMATH_CALUDE_peanut_distribution_theorem_l3789_378959


namespace NUMINAMATH_CALUDE_side_e_length_l3789_378929

-- Define the triangle DEF
structure Triangle where
  D : Real
  E : Real
  F : Real
  d : Real
  e : Real
  f : Real

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.E = 4 * t.D ∧ t.d = 18 ∧ t.f = 27

-- State the theorem
theorem side_e_length (t : Triangle) 
  (h : triangle_conditions t) : t.e = 27 := by
  sorry

end NUMINAMATH_CALUDE_side_e_length_l3789_378929


namespace NUMINAMATH_CALUDE_three_draw_probability_l3789_378955

def blue_chips : ℕ := 6
def yellow_chips : ℕ := 4
def total_chips : ℕ := blue_chips + yellow_chips

def prob_different_colors : ℚ := 72 / 625

theorem three_draw_probability :
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_diff_first_second : ℚ := prob_blue * prob_yellow + prob_yellow * prob_blue
  prob_diff_first_second * (prob_blue * prob_yellow + prob_yellow * prob_blue) = prob_different_colors :=
by sorry

end NUMINAMATH_CALUDE_three_draw_probability_l3789_378955


namespace NUMINAMATH_CALUDE_triangle_perimeter_triangle_perimeter_holds_l3789_378904

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being an odd number,
    the perimeter of the triangle is 12. -/
theorem triangle_perimeter : ℕ → Prop :=
  fun third_side =>
    (third_side > 0) →  -- Ensure positive length
    (third_side % 2 = 1) →  -- Odd number condition
    (2 < third_side) →  -- Lower bound from triangle inequality
    (third_side < 7) →  -- Upper bound from triangle inequality
    (2 + 5 + third_side = 12)

/-- The theorem holds. -/
theorem triangle_perimeter_holds : ∃ n, triangle_perimeter n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_triangle_perimeter_holds_l3789_378904


namespace NUMINAMATH_CALUDE_number_representation_l3789_378916

/-- Represents a number in terms of millions, ten thousands, and thousands -/
structure NumberComposition :=
  (millions : ℕ)
  (ten_thousands : ℕ)
  (thousands : ℕ)

/-- Converts a NumberComposition to its standard integer representation -/
def to_standard (n : NumberComposition) : ℕ :=
  n.millions * 1000000 + n.ten_thousands * 10000 + n.thousands * 1000

/-- Converts a natural number to its representation in ten thousands -/
def to_ten_thousands (n : ℕ) : ℚ :=
  (n : ℚ) / 10000

theorem number_representation (n : NumberComposition) 
  (h : n = ⟨6, 3, 4⟩) : 
  to_standard n = 6034000 ∧ to_ten_thousands (to_standard n) = 603.4 := by
  sorry

end NUMINAMATH_CALUDE_number_representation_l3789_378916


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l3789_378976

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 : 
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l3789_378976


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3789_378934

/-- A line with equation x = my + 2 is tangent to the circle x^2 + 2x + y^2 + 2y = 0 
    if and only if m = 1 or m = -7 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x = m * y + 2 → x^2 + 2*x + y^2 + 2*y ≠ 0) ∧ 
  (∃ x y : ℝ, x = m * y + 2 ∧ x^2 + 2*x + y^2 + 2*y = 0) ↔ 
  m = 1 ∨ m = -7 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3789_378934


namespace NUMINAMATH_CALUDE_pollen_mass_scientific_notation_l3789_378921

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem pollen_mass_scientific_notation :
  let mass : ℝ := 0.000037
  let scientific := toScientificNotation mass
  scientific.coefficient = 3.7 ∧ scientific.exponent = -5 :=
sorry

end NUMINAMATH_CALUDE_pollen_mass_scientific_notation_l3789_378921


namespace NUMINAMATH_CALUDE_maintenance_check_time_l3789_378919

/-- 
Proves that if an additive doubles the time between maintenance checks 
and the new time is 60 days, then the original time was 30 days.
-/
theorem maintenance_check_time (original_time : ℕ) : 
  (2 * original_time = 60) → original_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l3789_378919


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3789_378960

open Complex

theorem modulus_of_complex_fraction : 
  let z : ℂ := exp (π / 3 * I)
  ∀ (euler_formula : ∀ x : ℝ, exp (x * I) = cos x + I * sin x),
  abs (z / (1 - I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3789_378960


namespace NUMINAMATH_CALUDE_sum_of_digits_l3789_378922

theorem sum_of_digits (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  100 * a + 10 * b + c + 100 * d + 10 * c + b = 1100 →
  a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3789_378922


namespace NUMINAMATH_CALUDE_correct_solution_l3789_378994

theorem correct_solution (a b : ℚ) : 
  (∀ x y, x = 13 ∧ y = 7 → b * x - 7 * y = 16) →
  (∀ x y, x = 9 ∧ y = 4 → 2 * x + a * y = 6) →
  2 * 6 + a * 2 = 6 ∧ b * 6 - 7 * 2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_correct_solution_l3789_378994


namespace NUMINAMATH_CALUDE_candy_box_price_increase_candy_box_price_after_increase_l3789_378902

theorem candy_box_price_increase (initial_soda_price : ℝ) 
  (candy_increase_rate : ℝ) (soda_increase_rate : ℝ) 
  (initial_total_price : ℝ) : ℝ :=
  let initial_candy_price := initial_total_price - initial_soda_price
  let final_candy_price := initial_candy_price * (1 + candy_increase_rate)
  final_candy_price

theorem candy_box_price_after_increase :
  candy_box_price_increase 12 0.25 0.5 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_candy_box_price_after_increase_l3789_378902


namespace NUMINAMATH_CALUDE_sunRiseOnlyCertainEvent_l3789_378957

-- Define the type for events
inductive Event
  | SunRise
  | OpenBook
  | Thumbtack
  | Student

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.SunRise => true
  | _ => false

-- Theorem stating that SunRise is the only certain event
theorem sunRiseOnlyCertainEvent : 
  ∀ (e : Event), isCertain e ↔ e = Event.SunRise :=
by
  sorry


end NUMINAMATH_CALUDE_sunRiseOnlyCertainEvent_l3789_378957


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3789_378998

theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
  (h2 : a 2 = 2) 
  (h3 : q > 0) 
  (h4 : S 4 / S 2 = 10) : 
  a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3789_378998


namespace NUMINAMATH_CALUDE_janessas_initial_cards_l3789_378962

/-- The number of cards Janessa's father gave her -/
def fathers_cards : ℕ := 13

/-- The number of cards Janessa ordered from eBay -/
def ordered_cards : ℕ := 36

/-- The number of cards Janessa threw away -/
def discarded_cards : ℕ := 4

/-- The number of cards Janessa gave to Dexter -/
def cards_given_to_dexter : ℕ := 29

/-- The number of cards Janessa kept for herself -/
def cards_kept_for_self : ℕ := 20

/-- The initial number of cards Janessa had -/
def initial_cards : ℕ := 4

theorem janessas_initial_cards : 
  initial_cards + fathers_cards + ordered_cards - discarded_cards = 
  cards_given_to_dexter + cards_kept_for_self :=
by sorry

end NUMINAMATH_CALUDE_janessas_initial_cards_l3789_378962


namespace NUMINAMATH_CALUDE_curling_teams_l3789_378944

theorem curling_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_curling_teams_l3789_378944


namespace NUMINAMATH_CALUDE_line_through_center_perpendicular_to_axis_l3789_378978

/-- The polar equation of a circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- The center of the circle in Cartesian coordinates -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The polar equation of the line -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- The line passes through the center of the circle and is perpendicular to the polar axis -/
theorem line_through_center_perpendicular_to_axis :
  (∀ ρ θ : ℝ, circle_equation ρ θ → line_equation ρ θ) ∧
  (line_equation (circle_center.1) 0) ∧
  (∀ ρ : ℝ, line_equation ρ (Real.pi / 2)) :=
sorry

end NUMINAMATH_CALUDE_line_through_center_perpendicular_to_axis_l3789_378978


namespace NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l3789_378993

theorem min_sum_of_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → 
  (∃ (k : ℝ), k ≠ 0 ∧ (1 - x, x) = k • (1, -y)) →
  4 ≤ x + y ∧ (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    (∃ (k : ℝ), k ≠ 0 ∧ (1 - x₀, x₀) = k • (1, -y₀)) ∧ 
    x₀ + y₀ = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l3789_378993


namespace NUMINAMATH_CALUDE_box_2_neg1_3_equals_1_l3789_378996

def box (a b c : ℤ) : ℚ :=
  let k : ℤ := 2
  (k : ℚ) * (a : ℚ) ^ b - (b : ℚ) ^ c + (c : ℚ) ^ (a - k)

theorem box_2_neg1_3_equals_1 :
  box 2 (-1) 3 = 1 := by sorry

end NUMINAMATH_CALUDE_box_2_neg1_3_equals_1_l3789_378996


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3789_378911

theorem age_ratio_problem (albert mary betty : ℕ) 
  (h1 : ∃ k : ℕ, albert = k * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 22)
  (h4 : betty = 11) :
  albert / mary = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3789_378911


namespace NUMINAMATH_CALUDE_y_range_given_inequality_l3789_378999

/-- Custom multiplication operation on ℝ -/
def star (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of y given the condition -/
theorem y_range_given_inequality :
  (∀ x : ℝ, star (x - y) (x + y) < 1) →
  ∃ a b : ℝ, a = -1/2 ∧ b = 3/2 ∧ y ∈ Set.Ioo a b :=
by sorry

end NUMINAMATH_CALUDE_y_range_given_inequality_l3789_378999


namespace NUMINAMATH_CALUDE_spongebob_daily_earnings_l3789_378947

/-- Calculates Spongebob's earnings for the day based on burger and fries sales -/
def spongebob_earnings (num_burgers : ℕ) (burger_price : ℚ) (num_fries : ℕ) (fries_price : ℚ) : ℚ :=
  num_burgers * burger_price + num_fries * fries_price

/-- Theorem stating Spongebob's earnings for the day -/
theorem spongebob_daily_earnings :
  spongebob_earnings 30 2 12 (3/2) = 78 := by
  sorry


end NUMINAMATH_CALUDE_spongebob_daily_earnings_l3789_378947


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l3789_378952

theorem buffet_meal_combinations : ℕ := by
  -- Define the number of options for each food category
  let num_meats : ℕ := 4
  let num_vegetables : ℕ := 4
  let num_desserts : ℕ := 4
  let num_drinks : ℕ := 2

  -- Define the number of items Tyler chooses from each category
  let chosen_meats : ℕ := 2
  let chosen_vegetables : ℕ := 2
  let chosen_desserts : ℕ := 1
  let chosen_drinks : ℕ := 1

  -- Calculate the total number of meal combinations
  have h : (Nat.choose num_meats chosen_meats) * 
           (Nat.choose num_vegetables chosen_vegetables) * 
           num_desserts * num_drinks = 288 := by sorry

  exact 288

end NUMINAMATH_CALUDE_buffet_meal_combinations_l3789_378952


namespace NUMINAMATH_CALUDE_beatrix_pages_l3789_378933

theorem beatrix_pages (beatrix cristobal : ℕ) 
  (h1 : cristobal = 3 * beatrix + 15)
  (h2 : cristobal = beatrix + 1423) : 
  beatrix = 704 := by
sorry

end NUMINAMATH_CALUDE_beatrix_pages_l3789_378933


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3789_378971

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ c₁ c₂ k : ℝ) 
  (h1 : a₁ ≠ 0) (h2 : a₂ ≠ 0) (h3 : c₁ ≠ 0) (h4 : c₂ ≠ 0)
  (h5 : a₁ * b₁ * c₁ = k) (h6 : a₂ * b₂ * c₂ = k)
  (h7 : a₁ / a₂ = 3 / 4) (h8 : b₁ = 2 * b₂) : 
  c₁ / c₂ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3789_378971


namespace NUMINAMATH_CALUDE_restricted_arrangements_l3789_378925

/-- The number of ways to arrange n people in a row. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with one person fixed at the left end. -/
def permutations_with_left_fixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a row with one person fixed at the right end. -/
def permutations_with_right_fixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a row with two people fixed at both ends. -/
def permutations_with_both_ends_fixed (n : ℕ) : ℕ := Nat.factorial (n - 2)

theorem restricted_arrangements (n : ℕ) (h : n = 4) : 
  permutations n - permutations_with_left_fixed n - permutations_with_right_fixed n + permutations_with_both_ends_fixed n = 2 :=
sorry

end NUMINAMATH_CALUDE_restricted_arrangements_l3789_378925


namespace NUMINAMATH_CALUDE_wendy_ribbon_calculation_l3789_378961

/-- The amount of ribbon Wendy used to wrap presents, in inches. -/
def ribbon_used : ℕ := 46

/-- The amount of ribbon Wendy had left, in inches. -/
def ribbon_left : ℕ := 38

/-- The total amount of ribbon Wendy bought, in inches. -/
def total_ribbon : ℕ := ribbon_used + ribbon_left

theorem wendy_ribbon_calculation :
  total_ribbon = 84 := by
  sorry

end NUMINAMATH_CALUDE_wendy_ribbon_calculation_l3789_378961


namespace NUMINAMATH_CALUDE_b_age_is_twelve_l3789_378945

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - The total of their ages is 32
  Prove that b is 12 years old -/
theorem b_age_is_twelve (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 32) : 
  b = 12 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_twelve_l3789_378945


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l3789_378903

theorem solution_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, 2 * x + k = 3) → 
  (2 * 1 + k = 3) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l3789_378903


namespace NUMINAMATH_CALUDE_one_integer_solution_implies_a_range_l3789_378932

theorem one_integer_solution_implies_a_range (a : ℝ) :
  (∃! x : ℤ, (x : ℝ) - a ≥ 0 ∧ 2 * (x : ℝ) - 10 < 0) →
  3 < a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_one_integer_solution_implies_a_range_l3789_378932


namespace NUMINAMATH_CALUDE_no_winning_strategy_l3789_378913

/-- Represents a player in the game -/
inductive Player
| kezdo
| masodik

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 19
  col : Fin 19

/-- Represents a move in the game -/
structure Move where
  player : Player
  cell : Cell
  value : Fin 2

/-- Represents the state of the game after all moves -/
def GameState := List Move

/-- Calculates the sum of a row -/
def rowSum (state : GameState) (row : Fin 19) : Nat :=
  sorry

/-- Calculates the sum of a column -/
def colSum (state : GameState) (col : Fin 19) : Nat :=
  sorry

/-- Calculates the maximum row sum -/
def maxRowSum (state : GameState) : Nat :=
  sorry

/-- Calculates the maximum column sum -/
def maxColSum (state : GameState) : Nat :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: No winning strategy exists for either player -/
theorem no_winning_strategy :
  ∀ (kezdo_strategy : Strategy) (masodik_strategy : Strategy),
    ∃ (final_state : GameState),
      (maxRowSum final_state = maxColSum final_state) ∧
      (List.length final_state = 19 * 19) :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l3789_378913


namespace NUMINAMATH_CALUDE_earth_surface_available_for_living_l3789_378970

theorem earth_surface_available_for_living : 
  let earth_surface : ℝ := 1
  let land_fraction : ℝ := 1 / 3
  let inhabitable_fraction : ℝ := 1 / 4
  let residential_fraction : ℝ := 0.6
  earth_surface * land_fraction * inhabitable_fraction * residential_fraction = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_earth_surface_available_for_living_l3789_378970


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3789_378927

theorem decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ 3.36 = (n : ℚ) / (d : ℚ) ∧ (n.gcd d = 1) ∧ n = 84 ∧ d = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3789_378927


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3789_378990

theorem infinite_series_sum (a b : ℝ) 
  (h : (a / (b + 1)) / (1 - 1 / (b + 1)) = 3) : 
  (a / (a + 2*b)) / (1 - 1 / (a + 2*b)) = 3*(b + 1) / (5*b + 2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3789_378990


namespace NUMINAMATH_CALUDE_gcd_102_238_l3789_378918

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l3789_378918


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l3789_378965

theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 7 →
    associate_profs + 2 * assistant_profs = 11 →
    associate_profs + assistant_profs = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l3789_378965


namespace NUMINAMATH_CALUDE_root_product_cubic_polynomial_l3789_378936

theorem root_product_cubic_polynomial :
  let p := fun (x : ℝ) => 3 * x^3 - 4 * x^2 + x - 10
  ∃ a b c : ℝ, p a = 0 ∧ p b = 0 ∧ p c = 0 ∧ a * b * c = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_root_product_cubic_polynomial_l3789_378936


namespace NUMINAMATH_CALUDE_optimal_fence_placement_l3789_378975

/-- Represents the state of a tree (healthy or dead) --/
inductive TreeState
  | Healthy
  | Dead

/-- Represents a 6x6 grid of trees --/
def TreeGrid := Fin 6 → Fin 6 → TreeState

/-- Represents a fence placement --/
structure FencePlacement where
  vertical : Fin 3
  horizontal : Fin 3

/-- Checks if a tree is isolated by the given fence placement --/
def isIsolated (grid : TreeGrid) (fences : FencePlacement) (row col : Fin 6) : Prop :=
  sorry

/-- Counts the number of healthy trees in the grid --/
def countHealthyTrees (grid : TreeGrid) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem optimal_fence_placement
  (grid : TreeGrid)
  (healthy_count : countHealthyTrees grid = 20) :
  ∃ (fences : FencePlacement),
    ∀ (row col : Fin 6),
      grid row col = TreeState.Healthy →
      isIsolated grid fences row col :=
sorry

end NUMINAMATH_CALUDE_optimal_fence_placement_l3789_378975


namespace NUMINAMATH_CALUDE_system_solution_l3789_378958

/-- The system of linear equations -/
def system (x y : ℝ) : Prop :=
  x + y = 6 ∧ x = 2*y

/-- The solution set of the system -/
def solution_set : Set (ℝ × ℝ) :=
  {(4, 2)}

/-- Theorem stating that the solution set is correct -/
theorem system_solution : 
  {(x, y) | system x y} = solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l3789_378958


namespace NUMINAMATH_CALUDE_skew_lines_and_planes_l3789_378939

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the given conditions
variable (a b : Line)
variable (α : Plane)

-- Theorem statement
theorem skew_lines_and_planes 
  (h_skew : skew a b)
  (h_parallel : parallel a α) :
  (∃ β : Plane, parallel b β) ∧ 
  (∃ γ : Plane, subset b γ) ∧
  (∃ δ : Set Plane, Set.Infinite δ ∧ ∀ π ∈ δ, intersect b π) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_and_planes_l3789_378939


namespace NUMINAMATH_CALUDE_find_x_and_y_l3789_378988

theorem find_x_and_y :
  ∃ (x y : ℚ), 3 * (2 * x + 9 * y) = 75 ∧ x + y = 10 ∧ x = 65/7 ∧ y = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_find_x_and_y_l3789_378988


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_l3789_378963

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The area of a triangle given its base and height -/
def triangleArea (base height : ℝ) : ℝ := 0.5 * base * height

/-- The total area of the pentagon -/
def pentagonArea (p : Pentagon) : ℝ :=
  let rectangleABDE := rectangleArea 4 3
  let triangleBCD := triangleArea 4 (p.C.2 - 3)
  rectangleABDE + triangleBCD

theorem pentagon_y_coordinate (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 3))
  (h3 : p.C = (2, p.C.2))
  (h4 : p.D = (4, 3))
  (h5 : p.E = (4, 0))
  (h6 : pentagonArea p = 35) :
  p.C.2 = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_l3789_378963


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l3789_378917

theorem max_value_quadratic_function (f : ℝ → ℝ) (h : ∀ x ∈ (Set.Ioo 0 1), f x = x * (1 - x)) :
  ∃ x ∈ (Set.Ioo 0 1), ∀ y ∈ (Set.Ioo 0 1), f x ≥ f y ∧ f x = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l3789_378917


namespace NUMINAMATH_CALUDE_bread_calculation_l3789_378964

def initial_bread : ℕ := 200

def day1_fraction : ℚ := 1/4
def day2_fraction : ℚ := 2/5
def day3_fraction : ℚ := 1/2

def remaining_bread : ℕ := 45

theorem bread_calculation :
  (initial_bread - (day1_fraction * initial_bread).floor) -
  (day2_fraction * (initial_bread - (day1_fraction * initial_bread).floor)).floor -
  (day3_fraction * ((initial_bread - (day1_fraction * initial_bread).floor) -
    (day2_fraction * (initial_bread - (day1_fraction * initial_bread).floor)).floor)).floor = remaining_bread := by
  sorry

end NUMINAMATH_CALUDE_bread_calculation_l3789_378964


namespace NUMINAMATH_CALUDE_percentage_of_black_cats_l3789_378972

theorem percentage_of_black_cats 
  (total_cats : ℕ) 
  (white_cats : ℕ) 
  (grey_cats : ℕ) 
  (h1 : total_cats = 16) 
  (h2 : white_cats = 2) 
  (h3 : grey_cats = 10) :
  (((total_cats - white_cats - grey_cats) : ℚ) / total_cats) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_black_cats_l3789_378972


namespace NUMINAMATH_CALUDE_total_rainfall_five_days_l3789_378943

/-- Represents the rainfall data for a day -/
structure RainfallData where
  hours : ℝ
  rate : ℝ

/-- Calculates the total rainfall for a given day -/
def totalRainfall (data : RainfallData) : ℝ :=
  data.hours * data.rate

theorem total_rainfall_five_days (monday tuesday wednesday thursday friday : RainfallData)
  (h_monday : monday = { hours := 5, rate := 1 })
  (h_tuesday : tuesday = { hours := 3, rate := 1.5 })
  (h_wednesday : wednesday = { hours := 4, rate := 2 * monday.rate })
  (h_thursday : thursday = { hours := 6, rate := 0.5 * tuesday.rate })
  (h_friday : friday = { hours := 2, rate := 1.5 * wednesday.rate }) :
  totalRainfall monday + totalRainfall tuesday + totalRainfall wednesday +
  totalRainfall thursday + totalRainfall friday = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_five_days_l3789_378943


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l3789_378995

theorem polygon_diagonals_sides (n : ℕ) (h : n = 8) : (n * (n - 3)) / 2 = 2 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l3789_378995


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l3789_378900

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 8*x + k = (x + a)^2) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l3789_378900


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l3789_378915

theorem mod_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ -200 ≡ n [ZMOD 21] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l3789_378915


namespace NUMINAMATH_CALUDE_path_count_is_210_l3789_378928

/-- Number of paths on a grid from C to D -/
def num_paths (total_steps : ℕ) (right_steps : ℕ) (up_steps : ℕ) : ℕ :=
  Nat.choose total_steps up_steps

/-- Theorem: The number of different paths from C to D is 210 -/
theorem path_count_is_210 :
  num_paths 10 6 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_path_count_is_210_l3789_378928


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3789_378956

theorem difference_of_squares_special_case : (727 : ℤ) * 727 - 726 * 728 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3789_378956


namespace NUMINAMATH_CALUDE_ellipse_and_circle_l3789_378974

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  focal_length : ℝ
  short_axis_length : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_focal : focal_length = 2 * Real.sqrt 6
  h_short : short_axis_length = 2 * Real.sqrt 2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The line that intersects the ellipse -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 2

/-- Main theorem about the ellipse and its intersecting circle -/
theorem ellipse_and_circle (e : Ellipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 8 + y^2 / 2 = 1) ∧
  (∃ A B : ℝ × ℝ,
    ellipse_equation e A.1 A.2 ∧
    ellipse_equation e B.1 B.2 ∧
    intersecting_line A.1 A.2 ∧
    intersecting_line B.1 B.2 ∧
    ∀ x y, (x + 8/5)^2 + (y - 2/5)^2 = 48/25 ↔
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
        x = (1 - t) * A.1 + t * B.1 ∧
        y = (1 - t) * A.2 + t * B.2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_l3789_378974


namespace NUMINAMATH_CALUDE_books_remaining_l3789_378977

/-- Given Sandy has 10 books, Tim has 33 books, and Benny lost 24 of their books,
    prove that they have 19 books together now. -/
theorem books_remaining (sandy_books tim_books lost_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : lost_books = 24) : 
  sandy_books + tim_books - lost_books = 19 := by
  sorry

end NUMINAMATH_CALUDE_books_remaining_l3789_378977


namespace NUMINAMATH_CALUDE_rosy_work_days_l3789_378979

/-- Given that Mary can do a piece of work in 26 days and Rosy is 30% more efficient than Mary,
    prove that Rosy will take 20 days to do the same piece of work. -/
theorem rosy_work_days (mary_days : ℝ) (rosy_efficiency : ℝ) :
  mary_days = 26 →
  rosy_efficiency = 1.3 →
  (mary_days / rosy_efficiency : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rosy_work_days_l3789_378979


namespace NUMINAMATH_CALUDE_smallest_sum_of_quadratic_roots_l3789_378937

theorem smallest_sum_of_quadratic_roots (c d : ℝ) : 
  c > 0 → d > 0 → 
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) → 
  (∃ y : ℝ, y^2 + 3*d*y + c = 0) → 
  c + 3*d ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_quadratic_roots_l3789_378937


namespace NUMINAMATH_CALUDE_quarter_piles_count_l3789_378989

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of dimes -/
def dime_piles : ℕ := 6

/-- Represents the number of piles of nickels -/
def nickel_piles : ℕ := 9

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value of all coins in cents -/
def total_value : ℕ := 2100

/-- Theorem stating that the number of piles of quarters is 4 -/
theorem quarter_piles_count : 
  ∃ (quarter_piles : ℕ), 
    quarter_piles * coins_per_pile * quarter_value + 
    dime_piles * coins_per_pile * dime_value + 
    nickel_piles * coins_per_pile * nickel_value + 
    penny_piles * coins_per_pile * penny_value = total_value ∧ 
    quarter_piles = 4 := by
  sorry

end NUMINAMATH_CALUDE_quarter_piles_count_l3789_378989


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3789_378953

theorem min_value_and_inequality (x a b : ℝ) : x > 0 ∧ a > 0 ∧ b > 0 → 
  (∀ y : ℝ, y > 0 → x + 1/x ≤ y + 1/y) ∧ 
  (a * b ≤ ((a + b) / 2)^2) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3789_378953


namespace NUMINAMATH_CALUDE_sqrt_sum_inverse_squares_l3789_378954

theorem sqrt_sum_inverse_squares : 
  Real.sqrt (1 / 25 + 1 / 36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inverse_squares_l3789_378954


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l3789_378985

theorem imaginary_sum_zero (i : ℂ) (hi : i * i = -1) :
  1 / i + 1 / (i ^ 3) + 1 / (i ^ 5) + 1 / (i ^ 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l3789_378985


namespace NUMINAMATH_CALUDE_race_car_probability_l3789_378983

/-- A circular racetrack with a given length -/
structure CircularTrack where
  length : ℝ
  length_positive : length > 0

/-- A race car on the circular track -/
structure RaceCar where
  track : CircularTrack
  start_position : ℝ
  travel_distance : ℝ

/-- The probability of the car ending within a certain distance of a specific point -/
def end_probability (car : RaceCar) (target : ℝ) (range : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability for the specific problem -/
theorem race_car_probability (track : CircularTrack) 
  (h1 : track.length = 3)
  (car : RaceCar)
  (h2 : car.track = track)
  (h3 : car.travel_distance = 0.5)
  (target : ℝ)
  (h4 : target = 2.5) :
  end_probability car target 0.5 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_race_car_probability_l3789_378983


namespace NUMINAMATH_CALUDE_correct_calculation_l3789_378901

theorem correct_calculation (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3789_378901


namespace NUMINAMATH_CALUDE_money_division_l3789_378938

/-- Represents the share ratios of five individuals over five weeks -/
structure ShareRatios :=
  (a b c d e : Fin 5 → ℚ)

/-- Calculates the total ratio for a given week -/
def totalRatio (sr : ShareRatios) (week : Fin 5) : ℚ :=
  sr.a week + sr.b week + sr.c week + sr.d week + sr.e week

/-- Defines the initial ratios and weekly changes -/
def initialRatios : ShareRatios :=
  { a := λ _ => 1,
    b := λ w => 75/100 - w.val * 5/100,
    c := λ w => 60/100 - w.val * 5/100,
    d := λ w => 45/100 - w.val * 5/100,
    e := λ w => 30/100 + w.val * 15/100 }

/-- Theorem statement -/
theorem money_division (sr : ShareRatios) (h1 : sr = initialRatios) 
    (h2 : sr.e 4 * (totalRatio sr 0 / sr.e 4) = 413.33) : 
  sr.e 4 = 120 → totalRatio sr 0 = 413.33 := by
  sorry


end NUMINAMATH_CALUDE_money_division_l3789_378938


namespace NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l3789_378951

theorem largest_interior_angle_of_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a : ℝ) / 2 = b / 3 → b / 3 = c / 4 →
  a + b + c = 360 →
  180 - min a (min b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l3789_378951


namespace NUMINAMATH_CALUDE_temperature_change_over_700_years_l3789_378984

/-- Calculates the total temperature change over 700 years and converts it to Fahrenheit. -/
theorem temperature_change_over_700_years :
  let rate1 : ℝ := 3  -- rate for first 300 years (units per century)
  let rate2 : ℝ := 5  -- rate for next 200 years (units per century)
  let rate3 : ℝ := 2  -- rate for last 200 years (units per century)
  let period1 : ℝ := 3  -- first period in centuries
  let period2 : ℝ := 2  -- second period in centuries
  let period3 : ℝ := 2  -- third period in centuries
  let total_change_celsius : ℝ := rate1 * period1 + rate2 * period2 + rate3 * period3
  let total_change_fahrenheit : ℝ := total_change_celsius * (9/5) + 32
  total_change_celsius = 23 ∧ total_change_fahrenheit = 73.4 := by
  sorry

end NUMINAMATH_CALUDE_temperature_change_over_700_years_l3789_378984


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l3789_378987

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  (∀ x, 3*a - b ≥ x → x ≥ 3) ∧ 
  (∀ y, 3*a - b ≤ y → y ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l3789_378987


namespace NUMINAMATH_CALUDE_total_after_discount_rounded_l3789_378912

-- Define the purchases
def purchase1 : ℚ := 215 / 100
def purchase2 : ℚ := 749 / 100
def purchase3 : ℚ := 1285 / 100

-- Define the discount rate
def discount_rate : ℚ := 1 / 10

-- Function to apply discount to the most expensive item
def apply_discount (p1 p2 p3 : ℚ) (rate : ℚ) : ℚ :=
  let max_purchase := max p1 (max p2 p3)
  let discounted_max := max_purchase * (1 - rate)
  if p1 == max_purchase then discounted_max + p2 + p3
  else if p2 == max_purchase then p1 + discounted_max + p3
  else p1 + p2 + discounted_max

-- Function to round to nearest integer
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Theorem statement
theorem total_after_discount_rounded :
  round_to_nearest (apply_discount purchase1 purchase2 purchase3 discount_rate) = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_after_discount_rounded_l3789_378912


namespace NUMINAMATH_CALUDE_decimal_118_to_base6_l3789_378950

def decimal_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem decimal_118_to_base6 :
  decimal_to_base6 118 = [3, 1, 4] :=
sorry

end NUMINAMATH_CALUDE_decimal_118_to_base6_l3789_378950


namespace NUMINAMATH_CALUDE_parcel_weight_sum_l3789_378930

theorem parcel_weight_sum (x y z : ℝ) 
  (h1 : x + y = 168) 
  (h2 : y + z = 174) 
  (h3 : x + z = 180) : 
  x + y + z = 261 := by
  sorry

end NUMINAMATH_CALUDE_parcel_weight_sum_l3789_378930


namespace NUMINAMATH_CALUDE_concave_arithmetic_sequence_condition_l3789_378931

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

/-- A sequence is concave if a_{n-1} + a_{n+1} ≥ 2a_n for n ≥ 2 -/
def is_concave (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n - 1) + a (n + 1) ≥ 2 * a n

theorem concave_arithmetic_sequence_condition (d : ℝ) :
  let b := arithmetic_sequence 4 d
  is_concave (λ n => b n / n) → d ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_concave_arithmetic_sequence_condition_l3789_378931


namespace NUMINAMATH_CALUDE_remaining_bottles_calculation_l3789_378910

theorem remaining_bottles_calculation (small_initial big_initial medium_initial : ℕ)
  (small_sold_percent small_damaged_percent : ℚ)
  (big_sold_percent big_damaged_percent : ℚ)
  (medium_sold_percent medium_damaged_percent : ℚ)
  (h_small_initial : small_initial = 6000)
  (h_big_initial : big_initial = 15000)
  (h_medium_initial : medium_initial = 5000)
  (h_small_sold : small_sold_percent = 11/100)
  (h_small_damaged : small_damaged_percent = 3/100)
  (h_big_sold : big_sold_percent = 12/100)
  (h_big_damaged : big_damaged_percent = 2/100)
  (h_medium_sold : medium_sold_percent = 8/100)
  (h_medium_damaged : medium_damaged_percent = 4/100) :
  (small_initial - (small_initial * small_sold_percent).floor - (small_initial * small_damaged_percent).floor) +
  (big_initial - (big_initial * big_sold_percent).floor - (big_initial * big_damaged_percent).floor) +
  (medium_initial - (medium_initial * medium_sold_percent).floor - (medium_initial * medium_damaged_percent).floor) = 22560 := by
sorry

end NUMINAMATH_CALUDE_remaining_bottles_calculation_l3789_378910


namespace NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l3789_378946

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Checks if a point is on the extension of a line segment -/
def isOnExtension (A B H : Point) : Prop := sorry

/-- Checks if two line segments intersect at a point -/
def intersectsAt (P Q R S J : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem parallelogram_intersection_theorem (EFGH : Parallelogram) (H J K : Point) : 
  isOnExtension EFGH.E EFGH.F H →
  intersectsAt EFGH.G H EFGH.E EFGH.F J →
  intersectsAt EFGH.G H EFGH.F EFGH.G K →
  distance J K = 40 →
  distance H K = 30 →
  distance EFGH.G J = 20 := by sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l3789_378946


namespace NUMINAMATH_CALUDE_fraction_sum_and_lcd_l3789_378992

theorem fraction_sum_and_lcd : 
  let fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/9]
  let lcd := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)))))
  lcd = 360 ∧ fractions.sum = 607 / 360 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_lcd_l3789_378992


namespace NUMINAMATH_CALUDE_andrew_total_payment_l3789_378905

def grapes_quantity : ℕ := 15
def grapes_rate : ℕ := 98
def mangoes_quantity : ℕ := 8
def mangoes_rate : ℕ := 120
def pineapples_quantity : ℕ := 5
def pineapples_rate : ℕ := 75
def oranges_quantity : ℕ := 10
def oranges_rate : ℕ := 60

def total_cost : ℕ := 
  grapes_quantity * grapes_rate + 
  mangoes_quantity * mangoes_rate + 
  pineapples_quantity * pineapples_rate + 
  oranges_quantity * oranges_rate

theorem andrew_total_payment : total_cost = 3405 := by
  sorry

end NUMINAMATH_CALUDE_andrew_total_payment_l3789_378905


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3789_378997

/-- The ratio of a regular pentagon's side length to a rectangle's width, 
    given that they have the same perimeter and the rectangle's length is twice its width -/
theorem pentagon_rectangle_ratio (perimeter : ℝ) : 
  perimeter > 0 → 
  (5 : ℝ) * (perimeter / 5) / (perimeter / 6) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3789_378997


namespace NUMINAMATH_CALUDE_pushup_difference_l3789_378967

theorem pushup_difference (zachary_pushups john_pushups : ℕ) 
  (h1 : zachary_pushups = 51)
  (h2 : john_pushups = 69)
  (h3 : ∃ david_pushups : ℕ, david_pushups = john_pushups + 4) :
  ∃ david_pushups : ℕ, david_pushups - zachary_pushups = 22 :=
by sorry

end NUMINAMATH_CALUDE_pushup_difference_l3789_378967


namespace NUMINAMATH_CALUDE_regular_quadrilateral_pyramid_angle_l3789_378926

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The angle between a slant edge and the base plane -/
  slant_base_angle : ℝ
  /-- The angle between a slant edge and the plane of the lateral face that does not contain this edge -/
  slant_lateral_angle : ℝ
  /-- The angles are equal -/
  angle_equality : slant_base_angle = slant_lateral_angle

/-- The theorem stating the angle in a regular quadrilateral pyramid -/
theorem regular_quadrilateral_pyramid_angle (pyramid : RegularQuadrilateralPyramid) :
  pyramid.slant_base_angle = Real.arctan (Real.sqrt (3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_regular_quadrilateral_pyramid_angle_l3789_378926


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_implies_a_equals_one_l3789_378973

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) / (x + a)

theorem f_derivative_at_zero_implies_a_equals_one (a : ℝ) :
  (deriv (f a)) 0 = 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_implies_a_equals_one_l3789_378973


namespace NUMINAMATH_CALUDE_bill_take_home_salary_l3789_378942

def take_home_salary (gross_salary property_taxes sales_taxes income_tax_rate : ℝ) : ℝ :=
  gross_salary - (property_taxes + sales_taxes + income_tax_rate * gross_salary)

theorem bill_take_home_salary :
  take_home_salary 50000 2000 3000 0.1 = 40000 := by
  sorry

end NUMINAMATH_CALUDE_bill_take_home_salary_l3789_378942


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3789_378907

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, 2*x + y + 2 = 0 ∧ a*x + 4*y - 2 = 0 → 
    ((-1/2) * (-a/4) = -1)) → 
  a = -8 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3789_378907


namespace NUMINAMATH_CALUDE_intersection_point_d_l3789_378914

/-- The function g(x) = 2x + c -/
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

theorem intersection_point_d (c d : ℤ) :
  g c (-4) = d ∧ g c d = -4 → d = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_d_l3789_378914


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l3789_378968

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l3789_378968


namespace NUMINAMATH_CALUDE_largest_number_on_board_l3789_378906

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 6 = 0 ∧ ends_in_4 n

theorem largest_number_on_board :
  ∃ (m : ℕ), satisfies_conditions m ∧
  ∀ (n : ℕ), satisfies_conditions n → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_on_board_l3789_378906


namespace NUMINAMATH_CALUDE_elimination_failure_l3789_378935

theorem elimination_failure (x y : ℝ) : 
  (2 * x - 3 * y = 5) → 
  (3 * x - 2 * y = 7) → 
  (2 * (2 * x - 3 * y) - (-3) * (3 * x - 2 * y) ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_elimination_failure_l3789_378935


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3789_378980

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + 2 * f (1 - x) = 4 * x^2 + 3

/-- Theorem stating that for any function satisfying the functional equation, f(4) = 11/3 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 4 = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3789_378980


namespace NUMINAMATH_CALUDE_P_subset_Q_l3789_378941

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem P_subset_Q : P ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_P_subset_Q_l3789_378941


namespace NUMINAMATH_CALUDE_scientific_notation_of_218000_l3789_378969

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_218000 :
  toScientificNotation 218000 = ScientificNotation.mk 2.18 5 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_218000_l3789_378969
