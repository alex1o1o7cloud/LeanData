import Mathlib

namespace line_translation_theorem_l2849_284928

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - l.slope * dx + dy }

theorem line_translation_theorem :
  let original_line : Line := { slope := 2, intercept := -3 }
  let translated_line := translate original_line 2 3
  translated_line = { slope := 2, intercept := -4 } := by sorry

end line_translation_theorem_l2849_284928


namespace cubic_function_property_l2849_284967

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 3
  f (-2) = 7 → f 2 = -13 := by
  sorry

end cubic_function_property_l2849_284967


namespace min_bushes_for_pumpkins_l2849_284980

/-- Represents the number of containers of raspberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of raspberries needed to trade for 3 pumpkins -/
def containers_per_trade : ℕ := 6

/-- Represents the number of pumpkins obtained from one trade -/
def pumpkins_per_trade : ℕ := 3

/-- Represents the target number of pumpkins -/
def target_pumpkins : ℕ := 72

/-- 
Proves that the minimum number of bushes needed to obtain at least the target number of pumpkins
is 15, given the defined ratios of containers per bush and pumpkins per trade.
-/
theorem min_bushes_for_pumpkins :
  ∃ (n : ℕ), n * containers_per_bush * pumpkins_per_trade ≥ target_pumpkins * containers_per_trade ∧
  ∀ (m : ℕ), m * containers_per_bush * pumpkins_per_trade ≥ target_pumpkins * containers_per_trade → m ≥ n :=
by sorry

end min_bushes_for_pumpkins_l2849_284980


namespace abs_value_sum_difference_l2849_284986

theorem abs_value_sum_difference (a b : ℝ) : 
  (|a| = 2) → (|b| = 4) → (a + b < 0) → (a - b = 2 ∨ a - b = 6) := by
  sorry

end abs_value_sum_difference_l2849_284986


namespace composite_product_equals_twelve_over_pi_squared_l2849_284920

-- Define the sequence of composite numbers
def composite : ℕ → ℕ
  | 0 => 4  -- First composite number
  | n + 1 => sorry  -- Definition of subsequent composite numbers

-- Define the infinite product
def infinite_product : ℝ := sorry

-- Define the infinite sum of reciprocal squares
def reciprocal_squares_sum : ℝ := sorry

-- Theorem statement
theorem composite_product_equals_twelve_over_pi_squared :
  (reciprocal_squares_sum = Real.pi^2 / 6) →
  infinite_product = 12 / Real.pi^2 := by
  sorry

end composite_product_equals_twelve_over_pi_squared_l2849_284920


namespace root_sum_square_l2849_284964

theorem root_sum_square (a b : ℝ) : 
  a ≠ b →
  (a^2 + 2*a - 2022 = 0) → 
  (b^2 + 2*b - 2022 = 0) → 
  a^2 + 4*a + 2*b = 2018 := by
sorry

end root_sum_square_l2849_284964


namespace closer_to_center_is_enclosed_by_bisectors_l2849_284971

/-- A rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of points closer to the center of a rectangle than to any of its vertices -/
def CloserToCenter (r : Rectangle) : Set Point :=
  { p : Point | p.x^2 + p.y^2 < (p.x - r.a)^2 + (p.y - r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x - r.a)^2 + (p.y + r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x + r.a)^2 + (p.y + r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x + r.a)^2 + (p.y - r.b)^2 }

/-- Theorem stating that the set of points closer to the center is enclosed by perpendicular bisectors -/
theorem closer_to_center_is_enclosed_by_bisectors (r : Rectangle) :
  ∃ (bisectors : Set Point), CloserToCenter r = bisectors :=
sorry

end closer_to_center_is_enclosed_by_bisectors_l2849_284971


namespace f_g_inequality_l2849_284913

/-- The function f(x) = -x³ + x² + x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + x + a

/-- The function g(x) = 2a - x³ -/
def g (a : ℝ) (x : ℝ) : ℝ := 2*a - x^3

/-- Theorem: If g(x) ≥ f(x) for all x ∈ [0, 1], then a ≥ 2 -/
theorem f_g_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g a x ≥ f a x) → a ≥ 2 := by
  sorry

end f_g_inequality_l2849_284913


namespace work_completion_time_B_l2849_284963

theorem work_completion_time_B (a b : ℝ) : 
  (a + b = 1/6) →  -- A and B together complete 1/6 of the work in one day
  (a = 1/11) →     -- A alone completes 1/11 of the work in one day
  (b = 5/66) →     -- B alone completes 5/66 of the work in one day
  (1/b = 66/5) :=  -- The time B takes to complete the work alone is 66/5 days
by sorry

end work_completion_time_B_l2849_284963


namespace prime_combinations_theorem_l2849_284910

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def all_combinations_prime (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → is_prime (10^k * 7 + (10^n - 1) / 9 - 10^k)

theorem prime_combinations_theorem :
  ∀ n : ℕ, (all_combinations_prime n ↔ n = 1 ∨ n = 2) :=
sorry

end prime_combinations_theorem_l2849_284910


namespace inequality_theorem_l2849_284905

theorem inequality_theorem (p q r : ℝ) 
  (h_order : p < q)
  (h_inequality : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) :
  p + 2*q + 3*r = 1 :=
by sorry

end inequality_theorem_l2849_284905


namespace solve_system_l2849_284991

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : 2 * x + 3 * y = 8) : x = 37 / 13 := by
  sorry

end solve_system_l2849_284991


namespace current_speed_l2849_284969

/-- Given a boat's upstream and downstream speeds, calculate the current's speed -/
theorem current_speed (upstream_time : ℝ) (downstream_time : ℝ) :
  upstream_time = 30 →
  downstream_time = 12 →
  let upstream_speed := 60 / upstream_time
  let downstream_speed := 60 / downstream_time
  (downstream_speed - upstream_speed) / 2 = 1.5 := by
  sorry

#check current_speed

end current_speed_l2849_284969


namespace complex_exponential_form_angle_l2849_284952

theorem complex_exponential_form_angle (z : ℂ) : 
  z = 2 - 2 * Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (4 * Real.pi / 3)) :=
by sorry

end complex_exponential_form_angle_l2849_284952


namespace prob_diff_games_l2849_284907

/-- The probability of getting heads on a single coin flip -/
def p_heads : ℚ := 3/5

/-- The probability of getting tails on a single coin flip -/
def p_tails : ℚ := 2/5

/-- The probability of winning Game A -/
def p_win_game_a : ℚ := p_heads^4 + p_tails^4

/-- The probability of winning Game B -/
def p_win_game_b : ℚ := (p_heads^2 + p_tails^2) * (p_heads^3 + p_tails^3)

theorem prob_diff_games : p_win_game_a - p_win_game_b = 6/625 := by
  sorry

end prob_diff_games_l2849_284907


namespace irreducible_fraction_l2849_284935

theorem irreducible_fraction (a b c d : ℤ) (h : a * d - b * c = 1) :
  ¬∃ (m : ℤ), m > 1 ∧ m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d) := by
  sorry

end irreducible_fraction_l2849_284935


namespace division_by_negative_fraction_l2849_284931

theorem division_by_negative_fraction :
  5 / (-1/2 : ℚ) = -10 := by sorry

end division_by_negative_fraction_l2849_284931


namespace simultaneous_equations_solutions_l2849_284956

theorem simultaneous_equations_solutions :
  let eq1 (x y : ℝ) := x^2 + 3*y = 10
  let eq2 (x y : ℝ) := 3 + y = 10/x
  (eq1 (-5) (-5) ∧ eq2 (-5) (-5)) ∧
  (eq1 2 2 ∧ eq2 2 2) ∧
  (eq1 3 (1/3) ∧ eq2 3 (1/3)) :=
by sorry

end simultaneous_equations_solutions_l2849_284956


namespace heesu_has_greatest_sum_l2849_284973

-- Define the card values for each person
def sora_cards : List Nat := [4, 6]
def heesu_cards : List Nat := [7, 5]
def jiyeon_cards : List Nat := [3, 8]

-- Define a function to calculate the sum of cards
def sum_cards (cards : List Nat) : Nat :=
  cards.sum

-- Theorem statement
theorem heesu_has_greatest_sum :
  sum_cards heesu_cards > sum_cards sora_cards ∧
  sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  sorry


end heesu_has_greatest_sum_l2849_284973


namespace road_repair_workers_l2849_284926

/-- Represents the work done by a group of workers -/
structure Work where
  persons : ℕ
  days : ℕ
  hours_per_day : ℕ

/-- Calculates the total work units -/
def total_work (w : Work) : ℕ := w.persons * w.days * w.hours_per_day

theorem road_repair_workers (first_group : Work) (second_group : Work) :
  first_group.days = 12 ∧
  first_group.hours_per_day = 5 ∧
  second_group.persons = 30 ∧
  second_group.days = 17 ∧
  second_group.hours_per_day = 6 ∧
  total_work first_group = total_work second_group →
  first_group.persons = 51 := by
  sorry

end road_repair_workers_l2849_284926


namespace angle_A_measure_l2849_284925

/-- Given a geometric figure with the following properties:
  - Angle B measures 120°
  - A line divides the space opposite angle B on a straight line into two angles
  - One of these angles measures 50°
  - Angle A is vertically opposite to the angle that is not 50°
  Prove that angle A measures 130° -/
theorem angle_A_measure (B : ℝ) (angle1 : ℝ) (angle2 : ℝ) (A : ℝ) 
  (h1 : B = 120)
  (h2 : angle1 + angle2 = 180 - B)
  (h3 : angle1 = 50)
  (h4 : A = 180 - angle2) :
  A = 130 := by sorry

end angle_A_measure_l2849_284925


namespace storks_vs_birds_l2849_284923

theorem storks_vs_birds (initial_birds : ℕ) (additional_birds : ℕ) (storks : ℕ) : 
  initial_birds = 3 → additional_birds = 2 → storks = 6 → 
  storks - (initial_birds + additional_birds) = 1 := by
sorry

end storks_vs_birds_l2849_284923


namespace bart_monday_surveys_l2849_284970

/-- The number of surveys Bart finished on Monday -/
def monday_surveys : ℕ := 3

/-- The amount earned per question in dollars -/
def earnings_per_question : ℚ := 1/5

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart finished on Tuesday -/
def tuesday_surveys : ℕ := 4

/-- The total amount Bart earned over Monday and Tuesday in dollars -/
def total_earnings : ℚ := 14

theorem bart_monday_surveys :
  monday_surveys * questions_per_survey * earnings_per_question +
  tuesday_surveys * questions_per_survey * earnings_per_question =
  total_earnings :=
sorry

end bart_monday_surveys_l2849_284970


namespace weight_difference_l2849_284933

/-- Given weights of five individuals A, B, C, D, and E, prove that E weighs 6 kg more than D
    under specific average weight conditions. -/
theorem weight_difference (W_A W_B W_C W_D W_E : ℝ) : 
  (W_A + W_B + W_C) / 3 = 84 →
  (W_A + W_B + W_C + W_D) / 4 = 80 →
  (W_B + W_C + W_D + W_E) / 4 = 79 →
  W_A = 78 →
  W_E - W_D = 6 := by
sorry

end weight_difference_l2849_284933


namespace g_composition_of_three_l2849_284901

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 5*n - 3

theorem g_composition_of_three : g (g (g 3)) = 232 := by
  sorry

end g_composition_of_three_l2849_284901


namespace exists_large_class_l2849_284974

/-- A club of students -/
structure Club where
  students : Finset Nat
  classes : Nat → Finset Nat
  total_students : students.card = 60
  classmate_property : ∀ s : Finset Nat, s ⊆ students → s.card = 10 →
    ∃ c : Nat, (s ∩ classes c).card ≥ 3

/-- The main theorem -/
theorem exists_large_class (club : Club) :
  ∃ c : Nat, (club.students ∩ club.classes c).card ≥ 15 := by
  sorry

end exists_large_class_l2849_284974


namespace stone_game_ratio_bound_l2849_284937

/-- The stone game process -/
structure StoneGame where
  n : ℕ
  s : ℕ
  t : ℕ
  board : Multiset ℕ

/-- The rules of the stone game -/
def stone_game_step (game : StoneGame) (a b : ℕ) : StoneGame :=
  { n := game.n
  , s := game.s + 1
  , t := game.t + Nat.gcd a b
  , board := game.board - {a, b} + {1, a + b}
  }

/-- The theorem to prove -/
theorem stone_game_ratio_bound (game : StoneGame) (h_n : game.n ≥ 3) 
    (h_init : game.board = Multiset.replicate game.n 1) 
    (h_s_pos : game.s > 0) : 
    1 ≤ (game.t : ℚ) / game.s ∧ (game.t : ℚ) / game.s < game.n - 1 := by
  sorry

end stone_game_ratio_bound_l2849_284937


namespace smallest_integer_with_remainders_l2849_284914

theorem smallest_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 5 = 1 ∧
  b % 4 = 2 ∧
  b % 7 = 3 ∧
  ∀ c : ℕ, c > 0 → c % 5 = 1 → c % 4 = 2 → c % 7 = 3 → b ≤ c :=
by
  use 86
  sorry

end smallest_integer_with_remainders_l2849_284914


namespace sticker_difference_l2849_284919

theorem sticker_difference (initial_stickers : ℕ) : 
  initial_stickers > 0 →
  (initial_stickers - 15 : ℤ) / (initial_stickers + 18 : ℤ) = 2 / 5 →
  (initial_stickers + 18) - (initial_stickers - 15) = 33 := by
  sorry

#check sticker_difference

end sticker_difference_l2849_284919


namespace log_relationship_l2849_284934

theorem log_relationship (a b : ℝ) : 
  a = Real.log 243 / Real.log 5 → b = Real.log 27 / Real.log 3 → a = 5 * b / 3 := by
  sorry

end log_relationship_l2849_284934


namespace probability_same_gender_is_four_ninths_l2849_284930

/-- Represents a school with a specific number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- Calculates the total number of teachers in a school -/
def School.total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- Calculates the number of ways to select two teachers of the same gender -/
def same_gender_selections (s1 s2 : School) : ℕ :=
  s1.male_teachers * s2.male_teachers + s1.female_teachers * s2.female_teachers

/-- Calculates the total number of ways to select one teacher from each school -/
def total_selections (s1 s2 : School) : ℕ :=
  s1.total_teachers * s2.total_teachers

/-- The probability of selecting two teachers of the same gender -/
def probability_same_gender (s1 s2 : School) : ℚ :=
  (same_gender_selections s1 s2 : ℚ) / (total_selections s1 s2 : ℚ)

theorem probability_same_gender_is_four_ninths :
  let school_A : School := { male_teachers := 2, female_teachers := 1 }
  let school_B : School := { male_teachers := 1, female_teachers := 2 }
  probability_same_gender school_A school_B = 4 / 9 := by
  sorry

end probability_same_gender_is_four_ninths_l2849_284930


namespace junk_mail_distribution_l2849_284951

/-- Given a block with houses and junk mail to distribute, calculate the number of pieces per house -/
def junk_mail_per_house (num_houses : ℕ) (total_junk_mail : ℕ) : ℕ :=
  total_junk_mail / num_houses

/-- Theorem: For a block with 6 houses and 24 pieces of junk mail, each house receives 4 pieces -/
theorem junk_mail_distribution :
  junk_mail_per_house 6 24 = 4 := by
  sorry

end junk_mail_distribution_l2849_284951


namespace stickers_bought_from_store_l2849_284999

/-- Calculates the number of stickers Mika bought from the store -/
theorem stickers_bought_from_store 
  (initial : ℝ) 
  (birthday : ℝ) 
  (from_sister : ℝ) 
  (from_mother : ℝ) 
  (total : ℝ) 
  (h1 : initial = 20.0)
  (h2 : birthday = 20.0)
  (h3 : from_sister = 6.0)
  (h4 : from_mother = 58.0)
  (h5 : total = 130.0) :
  total - (initial + birthday + from_sister + from_mother) = 46.0 := by
  sorry

end stickers_bought_from_store_l2849_284999


namespace dinner_cost_calculation_l2849_284995

/-- The total amount paid for a dinner, given the food cost, sales tax rate, and tip rate. -/
def total_dinner_cost (food_cost : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  food_cost + (food_cost * sales_tax_rate) + (food_cost * tip_rate)

/-- Theorem stating that the total dinner cost is $35.85 given the specified conditions. -/
theorem dinner_cost_calculation :
  total_dinner_cost 30 0.095 0.10 = 35.85 := by
  sorry

end dinner_cost_calculation_l2849_284995


namespace range_of_a_l2849_284929

/-- The function g(x) = ax + 2 -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, g a x₁ = f x₀) →
  a ∈ Set.Ioo 0 1 :=
sorry

end range_of_a_l2849_284929


namespace power_of_128_four_sevenths_l2849_284994

theorem power_of_128_four_sevenths : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by
  sorry

end power_of_128_four_sevenths_l2849_284994


namespace integer_sum_of_fractions_l2849_284975

theorem integer_sum_of_fractions (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  ∃ k : ℤ, x^n / ((x-y)*(x-z)) + y^n / ((y-x)*(y-z)) + z^n / ((z-x)*(z-y)) = k := by
  sorry

end integer_sum_of_fractions_l2849_284975


namespace box_surface_area_l2849_284908

/-- Calculates the surface area of the interior of an open box formed from a rectangular cardboard with square corners removed. -/
def interior_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Proves that the surface area of the interior of the specified open box is 731 square units. -/
theorem box_surface_area : interior_surface_area 25 35 6 = 731 := by
  sorry

#eval interior_surface_area 25 35 6

end box_surface_area_l2849_284908


namespace pyramid_volume_l2849_284939

theorem pyramid_volume (base_length base_width height : ℝ) 
  (h1 : base_length = 2/3)
  (h2 : base_width = 1/2)
  (h3 : height = 1) : 
  (1/3 : ℝ) * base_length * base_width * height = 1/9 := by
  sorry

end pyramid_volume_l2849_284939


namespace boat_speed_in_still_water_l2849_284954

/-- Given a boat that travels 13 km/hr downstream and 4 km/hr upstream,
    prove that its speed in still water is 8.5 km/hr. -/
theorem boat_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : downstream_speed = 13)
  (h2 : upstream_speed = 4) :
  (downstream_speed + upstream_speed) / 2 = 8.5 := by
  sorry

end boat_speed_in_still_water_l2849_284954


namespace sheila_weekly_earnings_l2849_284909

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hoursMWF : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hoursTT : ℕ   -- Hours worked on Tuesday, Thursday
  daysLong : ℕ  -- Number of days working long hours (MWF)
  daysShort : ℕ -- Number of days working short hours (TT)
  hourlyRate : ℕ -- Hourly rate in dollars

/-- Calculates weekly earnings based on work schedule --/
def weeklyEarnings (schedule : WorkSchedule) : ℕ :=
  (schedule.hoursMWF * schedule.daysLong + schedule.hoursTT * schedule.daysShort) * schedule.hourlyRate

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  let schedule : WorkSchedule := {
    hoursMWF := 8,
    hoursTT := 6,
    daysLong := 3,
    daysShort := 2,
    hourlyRate := 11
  }
  weeklyEarnings schedule = 396 := by sorry

end sheila_weekly_earnings_l2849_284909


namespace abs_neg_five_l2849_284916

theorem abs_neg_five : |(-5 : ℤ)| = 5 := by
  sorry

end abs_neg_five_l2849_284916


namespace right_angle_clackers_l2849_284917

/-- The number of clackers in a full circle -/
def clackers_in_full_circle : ℕ := 600

/-- The fraction of a full circle that a right angle represents -/
def right_angle_fraction : ℚ := 1/4

/-- The number of clackers in a right angle -/
def clackers_in_right_angle : ℕ := 150

/-- Theorem: The number of clackers in a right angle is 150 -/
theorem right_angle_clackers :
  clackers_in_right_angle = (clackers_in_full_circle : ℚ) * right_angle_fraction := by
  sorry

end right_angle_clackers_l2849_284917


namespace pizza_fraction_l2849_284948

theorem pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) :
  total_slices = 12 →
  whole_slice = 1 →
  shared_slice = 1 / 2 →
  (whole_slice + shared_slice) / total_slices = 1 / 8 := by
  sorry

end pizza_fraction_l2849_284948


namespace target_line_properties_l2849_284979

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def line2 (x y : ℝ) : Prop := x - y + 5 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y = 0
def target_line (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem target_line_properties :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    target_line x y ∧
    ∃ (m1 m2 : ℝ),
      (∀ (x y : ℝ), line3 x y ↔ y = m1 * x) ∧
      (∀ (x y : ℝ), target_line x y ↔ y = m2 * x + (y - m2 * x)) ∧
      perpendicular m1 m2 :=
sorry

end target_line_properties_l2849_284979


namespace parabola_focus_directrix_distance_l2849_284912

/-- A parabola defined by y = ax² where a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- A line with slope 1 -/
structure Line where
  b : ℝ

/-- Intersection points of a parabola and a line -/
structure Intersection (p : Parabola) (l : Line) where
  x₁ : ℝ
  x₂ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = p.a * x₁^2
  eq₂ : y₂ = p.a * x₂^2
  eq₃ : y₁ = x₁ + l.b
  eq₄ : y₂ = x₂ + l.b

/-- The theorem to be proved -/
theorem parabola_focus_directrix_distance 
  (p : Parabola) (l : Line) (i : Intersection p l) :
  (i.x₁ + i.x₂) / 2 = 1 → 1 / (4 * p.a) = 1/4 := by
  sorry

end parabola_focus_directrix_distance_l2849_284912


namespace family_reunion_food_l2849_284962

/-- The amount of food Peter buys for the family reunion -/
def total_food (chicken hamburger hotdog side : ℝ) : ℝ :=
  chicken + hamburger + hotdog + side

theorem family_reunion_food : ∃ (chicken hamburger hotdog side : ℝ),
  chicken = 16 ∧
  hamburger = chicken / 2 ∧
  hotdog = hamburger + 2 ∧
  side = hotdog / 2 ∧
  total_food chicken hamburger hotdog side = 39 := by
  sorry

end family_reunion_food_l2849_284962


namespace student_count_incorrect_l2849_284990

theorem student_count_incorrect : ¬ ∃ k : ℕ, 18 + 17 * k = 2012 := by
  sorry

end student_count_incorrect_l2849_284990


namespace watermelon_problem_l2849_284932

theorem watermelon_problem (initial_watermelons : ℕ) (total_watermelons : ℕ) 
  (h1 : initial_watermelons = 4)
  (h2 : total_watermelons = 7) :
  total_watermelons - initial_watermelons = 3 := by
  sorry

end watermelon_problem_l2849_284932


namespace cabin_rental_duration_l2849_284987

/-- Proves that the number of days for which the cabin is rented is 14, given the specified conditions. -/
theorem cabin_rental_duration :
  let daily_rate : ℚ := 125
  let pet_fee : ℚ := 100
  let service_fee_rate : ℚ := 0.2
  let security_deposit_rate : ℚ := 0.5
  let security_deposit : ℚ := 1110
  ∃ (days : ℕ), 
    security_deposit = security_deposit_rate * (daily_rate * days + pet_fee + service_fee_rate * (daily_rate * days + pet_fee)) ∧
    days = 14 := by
  sorry

end cabin_rental_duration_l2849_284987


namespace quadratic_one_solution_l2849_284961

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) → k = 0 ∨ k = 1 := by
  sorry

end quadratic_one_solution_l2849_284961


namespace arithmetic_sequence_property_l2849_284985

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that a_{n+1}^2 = a_n * a_{n+2} for all n -/
def has_square_middle_property (a : Sequence) : Prop :=
  ∀ n : ℕ, (a (n + 1))^2 = a n * a (n + 2)

theorem arithmetic_sequence_property :
  (∀ a : Sequence, is_arithmetic a → has_square_middle_property a) ∧
  (∃ a : Sequence, has_square_middle_property a ∧ ¬is_arithmetic a) :=
sorry

end arithmetic_sequence_property_l2849_284985


namespace inequality_problem_l2849_284993

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) ∧
  ¬ (∀ b, c * b^2 < a * b^2) :=
by sorry

end inequality_problem_l2849_284993


namespace train_length_calculation_l2849_284958

/-- The length of a train given its speed and time to cross a point -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 360 →
  time_s = 0.9999200063994881 →
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 :=
by
  sorry

end train_length_calculation_l2849_284958


namespace star_two_three_l2849_284936

/-- The star operation defined as a * b = a * b^3 - 2 * b + 2 -/
def star (a b : ℝ) : ℝ := a * b^3 - 2 * b + 2

/-- Theorem: The value of 2 ★ 3 is 50 -/
theorem star_two_three : star 2 3 = 50 := by
  sorry

end star_two_three_l2849_284936


namespace circular_section_area_l2849_284981

theorem circular_section_area (r : ℝ) (d : ℝ) (h : r = 5 ∧ d = 3) :
  let section_radius : ℝ := Real.sqrt (r^2 - d^2)
  π * section_radius^2 = 16 * π :=
by sorry

end circular_section_area_l2849_284981


namespace sum_components_eq_46_l2849_284915

/-- Represents a trapezoid with four sides --/
structure Trapezoid :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)

/-- Represents the sum of areas in the form r₄√n₄ + r₅√n₅ + r₆ --/
structure AreaSum :=
  (r₄ : ℚ) (r₅ : ℚ) (r₆ : ℚ) (n₄ : ℕ) (n₅ : ℕ)

/-- Function to calculate the sum of all possible areas of a trapezoid --/
def sumAreas (t : Trapezoid) : AreaSum :=
  sorry

/-- Theorem stating that the sum of components equals 46 for the given trapezoid --/
theorem sum_components_eq_46 (t : Trapezoid) (a : AreaSum) :
  t.side1 = 4 ∧ t.side2 = 6 ∧ t.side3 = 8 ∧ t.side4 = 10 ∧
  a = sumAreas t →
  a.r₄ + a.r₅ + a.r₆ + a.n₄ + a.n₅ = 46 :=
sorry

end sum_components_eq_46_l2849_284915


namespace edges_after_intersection_l2849_284983

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the result of intersecting a polyhedron with planes -/
def intersect_with_planes (Q : ConvexPolyhedron) (num_planes : ℕ) : ℕ := sorry

/-- Theorem: The number of edges after intersection is 450 -/
theorem edges_after_intersection (Q : ConvexPolyhedron) (h1 : Q.edges = 150) :
  intersect_with_planes Q Q.vertices = 450 := by sorry

end edges_after_intersection_l2849_284983


namespace three_lines_divide_plane_l2849_284953

/-- A line in the plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at different points -/
def intersect_differently (l1 l2 l3 : Line) : Prop :=
  ¬ parallel l1 l2 ∧ ¬ parallel l1 l3 ∧ ¬ parallel l2 l3 ∧
  (l1.a * l2.b - l1.b * l2.a ≠ 0) ∧
  (l1.a * l3.b - l1.b * l3.a ≠ 0) ∧
  (l2.a * l3.b - l2.b * l3.a ≠ 0)

/-- The three given lines -/
def line1 : Line := ⟨1, 2, -1⟩
def line2 : Line := ⟨1, 0, 1⟩
def line3 (k : ℝ) : Line := ⟨1, k, 0⟩

theorem three_lines_divide_plane (k : ℝ) : 
  intersect_differently line1 line2 (line3 k) ↔ (k = 0 ∨ k = 1 ∨ k = 2) :=
sorry

end three_lines_divide_plane_l2849_284953


namespace apps_deleted_l2849_284972

theorem apps_deleted (initial_apps new_apps remaining_apps : ℕ) : 
  initial_apps = 10 →
  new_apps = 11 →
  remaining_apps = 4 →
  initial_apps + new_apps - remaining_apps = 17 :=
by sorry

end apps_deleted_l2849_284972


namespace prism_volume_l2849_284955

/-- 
  Given a right prism with an isosceles triangle base ABC, where:
  - AB = AC
  - ∠BAC = α
  - A line segment of length l from the upper vertex A₁ to the center of 
    the circumscribed circle of ABC makes an angle β with the base plane
  
  The volume of the prism is l³ sin(2β) cos(β) sin(α) cos²(α/2)
-/
theorem prism_volume 
  (α β l : ℝ) 
  (h_α : 0 < α ∧ α < π) 
  (h_β : 0 < β ∧ β < π/2) 
  (h_l : l > 0) : 
  ∃ (V : ℝ), V = l^3 * Real.sin (2*β) * Real.cos β * Real.sin α * (Real.cos (α/2))^2 := by
  sorry

end prism_volume_l2849_284955


namespace difference_sum_of_powers_of_three_l2849_284982

def S : Finset ℕ := Finset.range 9

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (λ i => S.sum (λ j => if i > j then (3^i - 3^j) else 0))

theorem difference_sum_of_powers_of_three : difference_sum S = 68896 := by
  sorry

end difference_sum_of_powers_of_three_l2849_284982


namespace fourth_month_sales_l2849_284949

/-- Calculates the missing sales amount for a month given the sales of other months and the average --/
def calculate_missing_sales (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

theorem fourth_month_sales :
  let sale1 : ℕ := 6400
  let sale2 : ℕ := 7000
  let sale3 : ℕ := 6800
  let sale5 : ℕ := 6500
  let sale6 : ℕ := 5100
  let average : ℕ := 6500
  calculate_missing_sales sale1 sale2 sale3 sale5 sale6 average = 7200 := by
  sorry

end fourth_month_sales_l2849_284949


namespace B_is_top_leftmost_l2849_284965

/-- Represents a rectangle with four sides labeled w, x, y, z --/
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- The set of all rectangles in the arrangement --/
def rectangles : Finset Rectangle := sorry

/-- Rectangle A --/
def A : Rectangle := ⟨5, 2, 8, 11⟩

/-- Rectangle B --/
def B : Rectangle := ⟨2, 1, 4, 7⟩

/-- Rectangle C --/
def C : Rectangle := ⟨4, 9, 6, 3⟩

/-- Rectangle D --/
def D : Rectangle := ⟨8, 6, 5, 9⟩

/-- Rectangle E --/
def E : Rectangle := ⟨10, 3, 9, 1⟩

/-- Rectangle F --/
def F : Rectangle := ⟨11, 4, 10, 2⟩

/-- Predicate to check if a rectangle is in the leftmost position --/
def isLeftmost (r : Rectangle) : Prop :=
  ∀ s ∈ rectangles, r.w ≤ s.w

/-- Predicate to check if a rectangle is in the top row --/
def isTopRow (r : Rectangle) : Prop := sorry

/-- The main theorem stating that B is the top leftmost rectangle --/
theorem B_is_top_leftmost : isLeftmost B ∧ isTopRow B := by sorry

end B_is_top_leftmost_l2849_284965


namespace min_value_xyz_l2849_284922

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  x + 3 * y + 6 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 6 * z₀ = 27 :=
by sorry

end min_value_xyz_l2849_284922


namespace tangent_line_implies_a_and_b_l2849_284945

/-- Given a curve f(x) = ax - b/x, prove that if its tangent line at (2, f(2)) 
    is 7x - 4y - 12 = 0, then a = 1 and b = 3 -/
theorem tangent_line_implies_a_and_b (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x - b / x
  let f' : ℝ → ℝ := λ x => a + b / (x^2)
  let tangent_slope : ℝ := f' 2
  let point_on_curve : ℝ := f 2
  (7 * 2 - 4 * point_on_curve - 12 = 0 ∧ 
   7 - 4 * tangent_slope = 0) →
  a = 1 ∧ b = 3 := by
sorry

end tangent_line_implies_a_and_b_l2849_284945


namespace fraction_equals_98_when_x_is_3_l2849_284927

theorem fraction_equals_98_when_x_is_3 :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64 + 2*x^2) / (x^4 + 8 + x^2) = 98 := by
  sorry

end fraction_equals_98_when_x_is_3_l2849_284927


namespace magnitude_of_vector_sum_l2849_284943

/-- Given vectors a and b, prove that the magnitude of 2a + b is 5√2 -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  a = (3, 2) → b = (-1, 1) → ‖(2 • a) + b‖ = 5 * Real.sqrt 2 := by
  sorry

end magnitude_of_vector_sum_l2849_284943


namespace new_bucket_capacity_l2849_284904

/-- Represents the capacity of a water tank in liters. -/
def TankCapacity : ℝ := 22 * 13.5

/-- Proves that given a tank that can be filled by either 22 buckets of 13.5 liters each
    or 33 buckets of equal capacity, the capacity of each of the 33 buckets is 9 liters. -/
theorem new_bucket_capacity : 
  ∀ (new_capacity : ℝ), 
  (33 * new_capacity = TankCapacity) → 
  new_capacity = 9 := by
sorry

end new_bucket_capacity_l2849_284904


namespace canoe_upstream_speed_l2849_284960

/-- 
Given a canoe that rows downstream at 12 km/hr and a stream with a speed of 2 km/hr,
this theorem proves that the speed of the canoe when rowing upstream is 8 km/hr.
-/
theorem canoe_upstream_speed :
  let downstream_speed : ℝ := 12
  let stream_speed : ℝ := 2
  let canoe_speed : ℝ := downstream_speed - stream_speed
  let upstream_speed : ℝ := canoe_speed - stream_speed
  upstream_speed = 8 := by sorry

end canoe_upstream_speed_l2849_284960


namespace tenth_term_is_37_l2849_284906

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 3 + a 5 = 26) ∧
  (a 1 + a 2 + a 3 + a 4 = 28)

/-- The 10th term of the arithmetic sequence is 37 -/
theorem tenth_term_is_37 (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 10 = 37 := by
  sorry

end tenth_term_is_37_l2849_284906


namespace three_digit_number_theorem_l2849_284997

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧
  n / 100 + n % 10 = 8

theorem three_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n → (n = 810 ∨ n = 840 ∨ n = 870) :=
by sorry

end three_digit_number_theorem_l2849_284997


namespace geometric_sequence_common_ratio_l2849_284946

/-- A geometric sequence of positive real numbers. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_prod : a 2 * a 6 = 16)
  (h_sum : a 4 + a 8 = 8) :
  q = 1 := by
sorry

end geometric_sequence_common_ratio_l2849_284946


namespace triangular_field_yield_l2849_284988

/-- Proves that a triangular field with given dimensions and harvest yields 1 ton per hectare -/
theorem triangular_field_yield (base : ℝ) (height_factor : ℝ) (total_harvest : ℝ) :
  base = 200 →
  height_factor = 1.2 →
  total_harvest = 2.4 →
  let height := height_factor * base
  let area_sq_meters := (1 / 2) * base * height
  let area_hectares := area_sq_meters / 10000
  total_harvest / area_hectares = 1 := by sorry

end triangular_field_yield_l2849_284988


namespace duck_count_l2849_284976

theorem duck_count (total_legs : ℕ) (rabbit_count : ℕ) (rabbit_legs : ℕ) (duck_legs : ℕ) :
  total_legs = 48 →
  rabbit_count = 9 →
  rabbit_legs = 4 →
  duck_legs = 2 →
  (total_legs - rabbit_count * rabbit_legs) / duck_legs = 6 :=
by sorry

end duck_count_l2849_284976


namespace catman_do_whisker_count_l2849_284940

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers (princess_puff_whiskers : ℕ) : ℕ :=
  2 * princess_puff_whiskers - 6

/-- Theorem stating the number of whiskers Catman Do has -/
theorem catman_do_whisker_count :
  catman_do_whiskers 14 = 22 := by
  sorry

end catman_do_whisker_count_l2849_284940


namespace smallest_four_digit_all_different_l2849_284947

/-- A function that checks if a natural number has all digits different --/
def allDigitsDifferent (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

/-- The smallest four-digit number with all digits different --/
def smallestFourDigitAllDifferent : ℕ := 1023

/-- Theorem: 1023 is the smallest four-digit number with all digits different --/
theorem smallest_four_digit_all_different :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ allDigitsDifferent n → smallestFourDigitAllDifferent ≤ n) ∧
  1000 ≤ smallestFourDigitAllDifferent ∧
  smallestFourDigitAllDifferent < 10000 ∧
  allDigitsDifferent smallestFourDigitAllDifferent :=
sorry

end smallest_four_digit_all_different_l2849_284947


namespace ali_total_money_l2849_284911

def five_dollar_bills : ℕ := 7
def ten_dollar_bills : ℕ := 1
def five_dollar_value : ℕ := 5
def ten_dollar_value : ℕ := 10

theorem ali_total_money :
  five_dollar_bills * five_dollar_value + ten_dollar_bills * ten_dollar_value = 45 := by
sorry

end ali_total_money_l2849_284911


namespace walter_age_2005_conditions_hold_l2849_284957

-- Define Walter's age in 2000
def walter_age_2000 : ℚ := 4 / 3

-- Define grandmother's age in 2000
def grandmother_age_2000 : ℚ := 2 * walter_age_2000

-- Define the current year
def current_year : ℕ := 2000

-- Define the target year
def target_year : ℕ := 2005

-- Define the sum of birth years
def sum_birth_years : ℕ := 4004

-- Theorem statement
theorem walter_age_2005 :
  (walter_age_2000 + (target_year - current_year : ℚ)) = 19 / 3 :=
by
  sorry

-- Verify the conditions
theorem conditions_hold :
  (walter_age_2000 = grandmother_age_2000 / 2) ∧
  (current_year - walter_age_2000 + current_year - grandmother_age_2000 = sum_birth_years) :=
by
  sorry

end walter_age_2005_conditions_hold_l2849_284957


namespace borgnine_leg_count_l2849_284978

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs (chimps lions lizards tarantulas : ℕ) : ℕ :=
  2 * chimps + 4 * lions + 4 * lizards + 8 * tarantulas

/-- Theorem stating the total number of legs Borgnine wants to see -/
theorem borgnine_leg_count : total_legs 12 8 5 125 = 1076 := by
  sorry

end borgnine_leg_count_l2849_284978


namespace jellybean_probability_l2849_284938

/-- Represents the number of ways to choose k items from n items --/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable total : ℕ) : ℚ := sorry

theorem jellybean_probability :
  let total_jellybeans : ℕ := 15
  let green_jellybeans : ℕ := 6
  let purple_jellybeans : ℕ := 2
  let yellow_jellybeans : ℕ := 7
  let picked_jellybeans : ℕ := 4

  let total_outcomes : ℕ := binomial total_jellybeans picked_jellybeans
  let yellow_combinations : ℕ := binomial yellow_jellybeans 2
  let non_yellow_combinations : ℕ := binomial (green_jellybeans + purple_jellybeans) 2
  let favorable_outcomes : ℕ := yellow_combinations * non_yellow_combinations

  probability favorable_outcomes total_outcomes = 4 / 9 :=
by sorry

end jellybean_probability_l2849_284938


namespace ursulas_purchases_l2849_284941

theorem ursulas_purchases (tea_price : ℝ) 
  (h1 : tea_price = 10)
  (h2 : tea_price > 0) :
  let cheese_price := tea_price / 2
  let butter_price := 0.8 * cheese_price
  let bread_price := butter_price / 2
  tea_price + cheese_price + butter_price + bread_price = 21 :=
by sorry

end ursulas_purchases_l2849_284941


namespace hallie_tuesday_hours_l2849_284977

/-- Calculates the number of hours Hallie worked on Tuesday given her earnings and tips -/
def hours_worked_tuesday (hourly_rate : ℚ) (monday_hours : ℚ) (monday_tips : ℚ) 
  (tuesday_tips : ℚ) (wednesday_hours : ℚ) (wednesday_tips : ℚ) (total_earnings : ℚ) : ℚ :=
  let monday_earnings := hourly_rate * monday_hours + monday_tips
  let wednesday_earnings := hourly_rate * wednesday_hours + wednesday_tips
  let tuesday_earnings := total_earnings - monday_earnings - wednesday_earnings
  let tuesday_wage_earnings := tuesday_earnings - tuesday_tips
  tuesday_wage_earnings / hourly_rate

theorem hallie_tuesday_hours :
  hours_worked_tuesday 10 7 18 12 7 20 240 = 5 := by
  sorry

end hallie_tuesday_hours_l2849_284977


namespace correct_quadratic_equation_l2849_284992

/-- The correct quadratic equation given erroneous roots -/
theorem correct_quadratic_equation 
  (root1_student1 root2_student1 : ℝ)
  (root1_student2 root2_student2 : ℝ)
  (h1 : root1_student1 = 5 ∧ root2_student1 = 3)
  (h2 : root1_student2 = -12 ∧ root2_student2 = -4) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 8 * x + 48 = 0) :=
sorry

#check correct_quadratic_equation

end correct_quadratic_equation_l2849_284992


namespace function_characterization_l2849_284924

def is_valid_function (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, (n : ℕ)^3 - (n : ℕ)^2 ≤ (f n : ℕ) * (f (f n) : ℕ)^2 ∧ 
             (f n : ℕ) * (f (f n) : ℕ)^2 ≤ (n : ℕ)^3 + (n : ℕ)^2

theorem function_characterization (f : ℕ+ → ℕ+) (h : is_valid_function f) : 
  ∀ n : ℕ+, f n = n - 1 ∨ f n = n ∨ f n = n + 1 :=
sorry

end function_characterization_l2849_284924


namespace jellybean_count_l2849_284902

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- Caleb's number of jellybeans -/
def caleb_jellybeans : ℕ := 3 * dozen

/-- Sophie's number of jellybeans -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The total number of jellybeans Caleb and Sophie have -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans

theorem jellybean_count : total_jellybeans = 54 := by
  sorry

end jellybean_count_l2849_284902


namespace line_intercept_at_10_l2849_284918

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

theorem line_intercept_at_10 : 
  let l : Line := { x₁ := 7, y₁ := 3, x₂ := 3, y₂ := 7 }
  xIntercept l = 10 := by sorry

end line_intercept_at_10_l2849_284918


namespace circumcircle_radius_of_triangle_l2849_284903

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + 1

theorem circumcircle_radius_of_triangle (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x = Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (f C = 2) →
  (a + b = 4) →
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 3) →
  (∃ R : ℝ, R = c / (2 * Real.sin C) ∧ R = 2) :=
by sorry

end circumcircle_radius_of_triangle_l2849_284903


namespace ball_distribution_probability_ratio_l2849_284921

theorem ball_distribution_probability_ratio :
  let total_balls : ℕ := 25
  let total_bins : ℕ := 6
  let p := (Nat.choose total_bins 2 * Nat.choose total_balls 3 * Nat.choose (total_balls - 3) 3 *
            Nat.choose (total_balls - 6) 4 * Nat.choose (total_balls - 10) 4 *
            Nat.choose (total_balls - 14) 4 * Nat.choose (total_balls - 18) 4) / 
           (Nat.factorial 4 * Nat.pow total_bins total_balls)
  let q := (Nat.choose total_bins 1 * Nat.choose total_balls 5 *
            Nat.choose (total_balls - 5) 4 * Nat.choose (total_balls - 9) 4 *
            Nat.choose (total_balls - 13) 4 * Nat.choose (total_balls - 17) 4 *
            Nat.choose (total_balls - 21) 4) / 
           Nat.pow total_bins total_balls
  p / q = 8 := by
sorry

end ball_distribution_probability_ratio_l2849_284921


namespace f_plus_g_is_linear_l2849_284950

/-- Represents a cubic function ax³ + bx² + cx + d -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The function resulting from reflecting and translating a cubic function -/
def reflected_translated (cf : CubicFunction) (x : ℝ) : ℝ :=
  cf.a * (x - 10)^3 + cf.b * (x - 10)^2 + cf.c * (x - 10) + cf.d

/-- The function resulting from reflecting about x-axis, then translating a cubic function -/
def reflected_translated_negative (cf : CubicFunction) (x : ℝ) : ℝ :=
  -cf.a * (x + 10)^3 - cf.b * (x + 10)^2 - cf.c * (x + 10) - cf.d

/-- The sum of the two reflected and translated functions -/
def f_plus_g (cf : CubicFunction) (x : ℝ) : ℝ :=
  reflected_translated cf x + reflected_translated_negative cf x

/-- Theorem stating that f_plus_g is a non-horizontal linear function -/
theorem f_plus_g_is_linear (cf : CubicFunction) :
  ∃ m k, m ≠ 0 ∧ ∀ x, f_plus_g cf x = m * x + k :=
sorry

end f_plus_g_is_linear_l2849_284950


namespace sum_of_x₁_and_x₂_l2849_284900

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def B : Set ℝ := {x | ∃ (x₁ x₂ : ℝ), x₁ ≤ x ∧ x ≤ x₂}

-- Define the conditions for union and intersection
axiom union_condition : A ∪ B = {x | x > -2}
axiom intersection_condition : A ∩ B = {x | 1 < x ∧ x ≤ 3}

-- The theorem to prove
theorem sum_of_x₁_and_x₂ : 
  ∃ (x₁ x₂ : ℝ), (∀ x, x ∈ B ↔ x₁ ≤ x ∧ x ≤ x₂) ∧ x₁ + x₂ = 2 :=
sorry

end sum_of_x₁_and_x₂_l2849_284900


namespace find_number_l2849_284996

theorem find_number (x : ℚ) : (55 + x / 78) * 78 = 4403 ↔ x = 1 := by sorry

end find_number_l2849_284996


namespace water_bottles_needed_l2849_284984

theorem water_bottles_needed (people : ℕ) (trip_hours : ℕ) (bottles_per_person_per_hour : ℚ) : 
  people = 10 → trip_hours = 24 → bottles_per_person_per_hour = 1/2 →
  (people : ℚ) * trip_hours * bottles_per_person_per_hour = 120 :=
by sorry

end water_bottles_needed_l2849_284984


namespace a_months_is_seven_l2849_284944

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  a_oxen : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  b_months : ℕ
  c_months : ℕ
  total_rent : ℕ
  c_share : ℕ

/-- Calculates the number of months a put his oxen for grazing -/
def calculate_a_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, a put his oxen for 7 months -/
theorem a_months_is_seven (rental : PastureRental)
  (h1 : rental.a_oxen = 10)
  (h2 : rental.b_oxen = 12)
  (h3 : rental.c_oxen = 15)
  (h4 : rental.b_months = 5)
  (h5 : rental.c_months = 3)
  (h6 : rental.total_rent = 140)
  (h7 : rental.c_share = 36) :
  calculate_a_months rental = 7 :=
sorry

end a_months_is_seven_l2849_284944


namespace complex_equation_solution_l2849_284966

theorem complex_equation_solution (z : ℂ) : z = -Complex.I / 7 ↔ 3 + 2 * Complex.I * z = 4 - 5 * Complex.I * z := by
  sorry

end complex_equation_solution_l2849_284966


namespace largest_whole_number_less_than_120_over_8_l2849_284989

theorem largest_whole_number_less_than_120_over_8 :
  ∃ (x : ℕ), x = 14 ∧ (∀ y : ℕ, 8 * y < 120 → y ≤ x) :=
by sorry

end largest_whole_number_less_than_120_over_8_l2849_284989


namespace quadratic_equation_properties_l2849_284942

theorem quadratic_equation_properties (m : ℝ) :
  let equation := fun x => x^2 - 2*m*x + m^2 - 4*m - 1
  (∃ x : ℝ, equation x = 0) ↔ m ≥ -1/4
  ∧
  equation 1 = 0 → m = 0 ∨ m = 6 := by
  sorry

end quadratic_equation_properties_l2849_284942


namespace room_number_unit_digit_l2849_284959

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def divisible_by_seven (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def contains_digit_nine (n : ℕ) : Prop := ∃ a b : ℕ, n = 10 * a + 9 * b ∧ b ≤ 1

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def satisfies_three_conditions (n : ℕ) : Prop :=
  (is_prime n ∧ is_even n ∧ divisible_by_seven n) ∨
  (is_prime n ∧ is_even n ∧ contains_digit_nine n) ∨
  (is_prime n ∧ divisible_by_seven n ∧ contains_digit_nine n) ∨
  (is_even n ∧ divisible_by_seven n ∧ contains_digit_nine n)

theorem room_number_unit_digit :
  ∃ n : ℕ, is_two_digit n ∧ satisfies_three_conditions n ∧ n % 10 = 8 :=
sorry

end room_number_unit_digit_l2849_284959


namespace union_equals_universal_l2849_284968

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_equals_universal : M ∪ N = U := by
  sorry

end union_equals_universal_l2849_284968


namespace complex_number_magnitude_squared_l2849_284998

theorem complex_number_magnitude_squared : 
  ∀ z : ℂ, z + Complex.abs z = 5 - 3*I → Complex.abs z^2 = 11.56 := by
  sorry

end complex_number_magnitude_squared_l2849_284998
