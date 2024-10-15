import Mathlib

namespace NUMINAMATH_CALUDE_amount_after_two_years_l2575_257531

/-- Calculate the amount after n years with a given initial value and annual increase rate -/
def amountAfterYears (initialValue : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + rate) ^ years

/-- Theorem: Given an initial amount of 6400 and an annual increase rate of 1/8,
    the amount after 2 years will be 8100 -/
theorem amount_after_two_years :
  let initialValue : ℝ := 6400
  let rate : ℝ := 1/8
  let years : ℕ := 2
  amountAfterYears initialValue rate years = 8100 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l2575_257531


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l2575_257587

theorem alcohol_mixture_percentage 
  (initial_volume : ℝ) 
  (initial_percentage : ℝ) 
  (added_alcohol : ℝ) 
  (h1 : initial_volume = 6) 
  (h2 : initial_percentage = 25) 
  (h3 : added_alcohol = 3) : 
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  final_alcohol / final_volume * 100 = 50 := by
sorry


end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l2575_257587


namespace NUMINAMATH_CALUDE_pictures_on_back_l2575_257554

theorem pictures_on_back (total : ℕ) (front : ℕ) (back : ℕ) 
  (h1 : total = 15) 
  (h2 : front = 6) 
  (h3 : total = front + back) : 
  back = 9 := by
  sorry

end NUMINAMATH_CALUDE_pictures_on_back_l2575_257554


namespace NUMINAMATH_CALUDE_circle_radius_reduction_l2575_257553

/-- Given a circle with an initial radius of 5 cm, if its area is reduced by 36%, the new radius will be 4 cm. -/
theorem circle_radius_reduction (π : ℝ) (h_π_pos : π > 0) : 
  let r₁ : ℝ := 5
  let A₁ : ℝ := π * r₁^2
  let A₂ : ℝ := 0.64 * A₁
  let r₂ : ℝ := Real.sqrt (A₂ / π)
  r₂ = 4 := by sorry

end NUMINAMATH_CALUDE_circle_radius_reduction_l2575_257553


namespace NUMINAMATH_CALUDE_train_length_l2575_257547

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 8 → speed_kmh * (1000 / 3600) * time_s = 160 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2575_257547


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2575_257575

theorem unique_integer_solution (m : ℤ) : 
  (∃! (x : ℤ), |2*x - m| ≤ 1) ∧ 
  (∀ (x : ℤ), |2*x - m| ≤ 1 → x = 2) → 
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2575_257575


namespace NUMINAMATH_CALUDE_quadratic_transformation_sum_l2575_257538

/-- Given a quadratic function y = x^2 - 4x - 12, when transformed into the form y = (x - m)^2 + p,
    the sum of m and p equals -14. -/
theorem quadratic_transformation_sum (x : ℝ) :
  ∃ (m p : ℝ), (∀ x, x^2 - 4*x - 12 = (x - m)^2 + p) ∧ (m + p = -14) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_sum_l2575_257538


namespace NUMINAMATH_CALUDE_no_small_order_of_two_l2575_257540

theorem no_small_order_of_two (p : ℕ) (h1 : Prime p) (h2 : ∃ k : ℕ, p = 4 * k + 1) (h3 : Prime (2 * p + 1)) :
  ¬ ∃ k : ℕ, k < 2 * p ∧ (2 : ZMod (2 * p + 1))^k = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_small_order_of_two_l2575_257540


namespace NUMINAMATH_CALUDE_perfect_cube_pair_l2575_257578

theorem perfect_cube_pair (a b : ℕ+) :
  (∃ (m n : ℕ+), a^3 + 6*a*b + 1 = m^3 ∧ b^3 + 6*a*b + 1 = n^3) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_perfect_cube_pair_l2575_257578


namespace NUMINAMATH_CALUDE_largest_811_double_l2575_257591

/-- Converts a number from base 8 to base 10 --/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 --/
def base10To8 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 11 --/
def base10To11 (n : ℕ) : ℕ := sorry

/-- Checks if a number is an 8-11 double --/
def is811Double (n : ℕ) : Prop :=
  base10To11 (base8To10 (base10To8 n)) = 2 * n

/-- The largest 8-11 double is 504 --/
theorem largest_811_double :
  (∀ m : ℕ, m > 504 → ¬ is811Double m) ∧ is811Double 504 := by sorry

end NUMINAMATH_CALUDE_largest_811_double_l2575_257591


namespace NUMINAMATH_CALUDE_gps_primary_benefit_l2575_257529

/-- Represents the capabilities of GPS technology -/
structure GPSTechnology where
  navigation : Bool
  routeOptimization : Bool
  costReduction : Bool

/-- Represents the uses of GPS in mobile phones -/
structure GPSUses where
  travel : Bool
  tourism : Bool
  exploration : Bool

/-- Represents the primary benefit of GPS technology in daily life -/
def primaryBenefit (tech : GPSTechnology) : Prop :=
  tech.routeOptimization ∧ tech.costReduction

/-- The theorem stating that given GPS is used for travel, tourism, and exploration,
    its primary benefit is route optimization and cost reduction -/
theorem gps_primary_benefit (uses : GPSUses) (tech : GPSTechnology) 
  (h1 : uses.travel = true)
  (h2 : uses.tourism = true)
  (h3 : uses.exploration = true)
  (h4 : tech.navigation = true) :
  primaryBenefit tech :=
sorry

end NUMINAMATH_CALUDE_gps_primary_benefit_l2575_257529


namespace NUMINAMATH_CALUDE_add_3577_minutes_to_start_time_l2575_257589

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startDateTime : DateTime :=
  { year := 2020, month := 12, day := 31, hour := 18, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3577

/-- The resulting DateTime after adding minutes -/
def resultDateTime : DateTime :=
  { year := 2021, month := 1, day := 3, hour := 5, minute := 37 }

theorem add_3577_minutes_to_start_time :
  addMinutes startDateTime minutesToAdd = resultDateTime := by sorry

end NUMINAMATH_CALUDE_add_3577_minutes_to_start_time_l2575_257589


namespace NUMINAMATH_CALUDE_earnings_difference_l2575_257550

def bert_phones : ℕ := 8
def bert_price : ℕ := 18
def tory_guns : ℕ := 7
def tory_price : ℕ := 20

theorem earnings_difference : bert_phones * bert_price - tory_guns * tory_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l2575_257550


namespace NUMINAMATH_CALUDE_mortgage_payment_l2575_257563

theorem mortgage_payment (total : ℝ) (months : ℕ) (ratio : ℝ) (first_payment : ℝ) :
  total = 109300 ∧ 
  months = 7 ∧ 
  ratio = 3 ∧ 
  total = first_payment * (1 - ratio^months) / (1 - ratio) →
  first_payment = 100 := by
sorry

end NUMINAMATH_CALUDE_mortgage_payment_l2575_257563


namespace NUMINAMATH_CALUDE_ladder_length_l2575_257508

theorem ladder_length : ∃ L : ℝ, 
  L > 0 ∧ 
  (4/5 * L)^2 + 4^2 = L^2 ∧ 
  L = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_ladder_length_l2575_257508


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2575_257588

/-- The side length of a rhombus given its area and diagonal ratio -/
theorem rhombus_side_length (K : ℝ) (h : K > 0) : ∃ (s : ℝ),
  s > 0 ∧
  ∃ (d₁ d₂ : ℝ),
    d₁ > 0 ∧ d₂ > 0 ∧
    d₂ = 3 * d₁ ∧
    K = (1/2) * d₁ * d₂ ∧
    s^2 = (d₁/2)^2 + (d₂/2)^2 ∧
    s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2575_257588


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l2575_257566

/-- Represents the composition of a bag of marbles -/
structure BagComposition where
  color1 : ℕ
  color2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def drawProbability (bag : BagComposition) (colorCount : ℕ) : ℚ :=
  colorCount / (bag.color1 + bag.color2)

/-- The main theorem statement -/
theorem yellow_marble_probability
  (bagX : BagComposition)
  (bagY : BagComposition)
  (bagZ : BagComposition)
  (hX : bagX = ⟨5, 3⟩)
  (hY : bagY = ⟨8, 2⟩)
  (hZ : bagZ = ⟨3, 4⟩) :
  let probWhiteX := drawProbability bagX bagX.color1
  let probYellowY := drawProbability bagY bagY.color1
  let probBlackX := drawProbability bagX bagX.color2
  let probYellowZ := drawProbability bagZ bagZ.color1
  probWhiteX * probYellowY + probBlackX * probYellowZ = 37 / 56 :=
sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l2575_257566


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_eq_16_l2575_257511

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  a_4_eq_7 : a 4 = 7
  a_3_plus_a_6_eq_16 : a 3 + a 6 = 16
  exists_n : ∃ n : ℕ, a n = 31

/-- The theorem stating that n = 16 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_eq_16 (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = 31 ∧ n = 16 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_n_eq_16_l2575_257511


namespace NUMINAMATH_CALUDE_passing_percentage_l2575_257595

/-- Given a total of 500 marks, a student who got 150 marks and failed by 50 marks,
    prove that the percentage needed to pass is 40%. -/
theorem passing_percentage (total_marks : ℕ) (obtained_marks : ℕ) (failing_margin : ℕ) :
  total_marks = 500 →
  obtained_marks = 150 →
  failing_margin = 50 →
  (obtained_marks + failing_margin) / total_marks * 100 = 40 := by
  sorry


end NUMINAMATH_CALUDE_passing_percentage_l2575_257595


namespace NUMINAMATH_CALUDE_binomial_coefficient_1300_2_l2575_257585

theorem binomial_coefficient_1300_2 : 
  Nat.choose 1300 2 = 844350 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1300_2_l2575_257585


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2575_257527

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x | x^2 + a*x + b < 0}) : 
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2575_257527


namespace NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l2575_257565

/-- A sequence of positive integers is in harmonic progression if their reciprocals form an arithmetic progression. -/
def IsHarmonicProgression (s : ℕ → ℕ) : Prop :=
  ∃ d : ℚ, ∀ i j : ℕ, (1 : ℚ) / s i - (1 : ℚ) / s j = d * (i - j)

/-- For any natural number N, there exists a strictly increasing sequence of N positive integers in harmonic progression. -/
theorem exists_finite_harmonic_progression (N : ℕ) :
    ∃ (s : ℕ → ℕ), (∀ i < N, s i < s (i + 1)) ∧ IsHarmonicProgression s :=
  sorry

/-- There does not exist a strictly increasing infinite sequence of positive integers in harmonic progression. -/
theorem no_infinite_harmonic_progression :
    ¬∃ (s : ℕ → ℕ), (∀ i : ℕ, s i < s (i + 1)) ∧ IsHarmonicProgression s :=
  sorry

end NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l2575_257565


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l2575_257551

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals 12 = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l2575_257551


namespace NUMINAMATH_CALUDE_vendor_division_l2575_257546

theorem vendor_division (account_balance : Nat) (min_addition : Nat) (num_vendors : Nat) : 
  account_balance = 329864 →
  min_addition = 4 →
  num_vendors = 20 →
  (∀ k < num_vendors, account_balance % k ≠ 0 ∨ (account_balance + min_addition) % k ≠ 0) ∧
  account_balance % num_vendors ≠ 0 ∧
  (account_balance + min_addition) % num_vendors = 0 :=
by sorry

end NUMINAMATH_CALUDE_vendor_division_l2575_257546


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_condition_l2575_257522

/-- Represents the condition for a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  (m + 3) * (2*m + 1) < 0

/-- Represents the condition for an ellipse with foci on y-axis -/
def is_ellipse_y_foci (m : ℝ) : Prop :=
  -(2*m - 1) > m + 2 ∧ m + 2 > 0

/-- The necessary but not sufficient condition -/
def necessary_condition (m : ℝ) : Prop :=
  -2 < m ∧ m < -1/3

theorem hyperbola_ellipse_condition :
  (∀ m, is_hyperbola m ∧ is_ellipse_y_foci m → necessary_condition m) ∧
  ¬(∀ m, necessary_condition m → is_hyperbola m ∧ is_ellipse_y_foci m) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_condition_l2575_257522


namespace NUMINAMATH_CALUDE_five_at_ten_equals_ten_thirds_l2575_257557

-- Define the @ operation for positive integers
def at_operation (a b : ℕ+) : ℚ := (a * b : ℚ) / (a + b : ℚ)

-- State the theorem
theorem five_at_ten_equals_ten_thirds : 
  at_operation 5 10 = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_five_at_ten_equals_ten_thirds_l2575_257557


namespace NUMINAMATH_CALUDE_nina_running_distance_l2575_257526

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℝ := 0.621371

/-- Conversion factor from yards to miles -/
def yard_to_miles : ℝ := 0.000568182

/-- Distance Nina ran in miles for her initial run -/
def initial_run : ℝ := 0.08

/-- Distance Nina ran in kilometers for her second run (done twice) -/
def second_run_km : ℝ := 3

/-- Distance Nina ran in yards for her third run -/
def third_run_yards : ℝ := 1200

/-- Distance Nina ran in kilometers for her final run -/
def final_run_km : ℝ := 6

/-- Total distance Nina ran in miles -/
def total_distance : ℝ := 
  initial_run + 
  2 * (second_run_km * km_to_miles) + 
  (third_run_yards * yard_to_miles) + 
  (final_run_km * km_to_miles)

theorem nina_running_distance : 
  ∃ ε > 0, |total_distance - 8.22| < ε :=
sorry

end NUMINAMATH_CALUDE_nina_running_distance_l2575_257526


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l2575_257500

-- Define the speed of the boat in still water
def boat_speed : ℝ := 10

-- Define the distance traveled against the stream in one hour
def distance_against_stream : ℝ := 5

-- Define the time of travel
def travel_time : ℝ := 1

-- Theorem statement
theorem boat_distance_along_stream :
  let stream_speed := boat_speed - distance_against_stream / travel_time
  (boat_speed + stream_speed) * travel_time = 15 := by sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l2575_257500


namespace NUMINAMATH_CALUDE_triangle_inequality_l2575_257555

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  (c + a - b)^4 / (a * (a + b - c)) +
  (a + b - c)^4 / (b * (b + c - a)) +
  (b + c - a)^4 / (c * (c + a - b)) ≥
  a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2575_257555


namespace NUMINAMATH_CALUDE_geometric_sequence_log_sum_l2575_257523

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The logarithm function (base 10) -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Theorem: In a geometric sequence where a₂ * a₅ * a₈ = 1, lg(a₄) + lg(a₆) = 0 -/
theorem geometric_sequence_log_sum (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_prod : a 2 * a 5 * a 8 = 1) : 
  lg (a 4) + lg (a 6) = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_log_sum_l2575_257523


namespace NUMINAMATH_CALUDE_quotient_invariance_problem_solution_l2575_257574

theorem quotient_invariance (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / b = (c * a) / (c * b) :=
by sorry

theorem problem_solution : (0.75 : ℝ) / 25 = 7.5 / 250 := by
  have h1 : (0.75 : ℝ) / 25 = (10 * 0.75) / (10 * 25) := by
    apply quotient_invariance 0.75 25 10
    norm_num
    norm_num
  have h2 : (10 * 0.75 : ℝ) = 7.5 := by norm_num
  have h3 : (10 * 25 : ℝ) = 250 := by norm_num
  rw [h1, h2, h3]

end NUMINAMATH_CALUDE_quotient_invariance_problem_solution_l2575_257574


namespace NUMINAMATH_CALUDE_connie_red_markers_l2575_257513

/-- The number of red markers Connie has -/
def red_markers : ℕ := 3343 - 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := 3343

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

theorem connie_red_markers :
  red_markers = 2315 ∧ total_markers = red_markers + blue_markers :=
sorry

end NUMINAMATH_CALUDE_connie_red_markers_l2575_257513


namespace NUMINAMATH_CALUDE_max_at_neg_two_l2575_257503

/-- The function f(x) that we're analyzing -/
def f (m : ℝ) (x : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 + 2*x*(x - m)

theorem max_at_neg_two (m : ℝ) :
  (∀ x : ℝ, f m x ≤ f m (-2)) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_max_at_neg_two_l2575_257503


namespace NUMINAMATH_CALUDE_x_value_when_y_72_l2575_257534

/-- Given positive numbers x and y, where x^2 * y is constant, y = 8 when x = 3,
    and x^2 has increased by a factor of 4, prove that x = 1 when y = 72 -/
theorem x_value_when_y_72 (x y : ℝ) (z : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : ∃ k : ℝ, ∀ x y, x^2 * y = k)
  (h4 : 3^2 * 8 = 8 * 3^2)
  (h5 : z = 4)
  (h6 : y = 72)
  (h7 : x^2 = 3^2 * z) :
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_72_l2575_257534


namespace NUMINAMATH_CALUDE_conner_needs_27_rocks_l2575_257535

/-- Calculates the number of rocks Conner needs to collect on day 3 to at least tie with Sydney -/
def rocks_conner_needs_day3 (sydney_initial : ℕ) (conner_initial : ℕ) 
  (sydney_day1 : ℕ) (conner_day1_multiplier : ℕ) 
  (sydney_day2 : ℕ) (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ) : ℕ :=
  let conner_day1 := sydney_day1 * conner_day1_multiplier
  let sydney_day3 := conner_day1 * sydney_day3_multiplier
  let sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
  let conner_before_day3 := conner_initial + conner_day1 + conner_day2
  sydney_total - conner_before_day3

/-- Theorem stating that Conner needs to collect 27 rocks on day 3 to at least tie with Sydney -/
theorem conner_needs_27_rocks : 
  rocks_conner_needs_day3 837 723 4 8 0 123 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_conner_needs_27_rocks_l2575_257535


namespace NUMINAMATH_CALUDE_downstream_distance_l2575_257570

/-- Prove that given the conditions of a boat rowing upstream and downstream,
    the distance rowed downstream is 200 km. -/
theorem downstream_distance
  (boat_speed : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (downstream_time : ℝ)
  (h1 : boat_speed = 14)
  (h2 : upstream_distance = 96)
  (h3 : upstream_time = 12)
  (h4 : downstream_time = 10)
  (h5 : upstream_distance / upstream_time = boat_speed - (boat_speed - upstream_distance / upstream_time)) :
  (boat_speed + (boat_speed - upstream_distance / upstream_time)) * downstream_time = 200 := by
sorry

end NUMINAMATH_CALUDE_downstream_distance_l2575_257570


namespace NUMINAMATH_CALUDE_first_digit_base_9_of_21221122211112211111_base_3_l2575_257505

def base_3_num : List Nat := [2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]

def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 3^(digits.length - 1 - i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0 else
  let log_9_n := (Nat.log n 9)
  n / 9^log_9_n

theorem first_digit_base_9_of_21221122211112211111_base_3 :
  first_digit_base_9 (to_base_10 base_3_num) = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base_9_of_21221122211112211111_base_3_l2575_257505


namespace NUMINAMATH_CALUDE_angle_D_measure_l2575_257519

-- Define the geometric figure
def geometric_figure (B C D E F : Real) : Prop :=
  -- Angle B measures 120°
  B = 120 ∧
  -- Angle B and C form a linear pair
  B + C = 180 ∧
  -- In triangle DEF, angle E = 45°
  E = 45 ∧
  -- Angle F is vertically opposite to angle C
  F = C ∧
  -- Triangle DEF sum of angles
  D + E + F = 180

-- Theorem statement
theorem angle_D_measure (B C D E F : Real) :
  geometric_figure B C D E F → D = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2575_257519


namespace NUMINAMATH_CALUDE_image_of_3_4_preimage_of_1_neg6_l2575_257593

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (3, 4)
theorem image_of_3_4 : f (3, 4) = (7, 12) := by sorry

-- Definition of preimage
def preimage (f : ℝ × ℝ → ℝ × ℝ) (y : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | f x = y}

-- Theorem for the preimage of (1, -6)
theorem preimage_of_1_neg6 : preimage f (1, -6) = {(-2, 3), (3, -2)} := by sorry

end NUMINAMATH_CALUDE_image_of_3_4_preimage_of_1_neg6_l2575_257593


namespace NUMINAMATH_CALUDE_orange_orchard_composition_l2575_257577

/-- Represents an orange orchard with flat and hilly areas. -/
structure Orchard :=
  (total_acres : ℕ)
  (sampled_acres : ℕ)
  (flat_sampled : ℕ)
  (hilly_sampled : ℕ)

/-- Checks if the sampling method is valid for the given orchard. -/
def valid_sampling (o : Orchard) : Prop :=
  o.hilly_sampled = 2 * o.flat_sampled + 1 ∧
  o.flat_sampled + o.hilly_sampled = o.sampled_acres

/-- Calculates the number of flat acres in the orchard based on the sampling. -/
def flat_acres (o : Orchard) : ℕ :=
  o.flat_sampled * (o.total_acres / o.sampled_acres)

/-- Calculates the number of hilly acres in the orchard based on the sampling. -/
def hilly_acres (o : Orchard) : ℕ :=
  o.hilly_sampled * (o.total_acres / o.sampled_acres)

/-- Theorem stating the composition of the orange orchard. -/
theorem orange_orchard_composition (o : Orchard) 
  (h1 : o.total_acres = 120)
  (h2 : o.sampled_acres = 10)
  (h3 : valid_sampling o) :
  flat_acres o = 36 ∧ hilly_acres o = 84 :=
sorry

end NUMINAMATH_CALUDE_orange_orchard_composition_l2575_257577


namespace NUMINAMATH_CALUDE_quadratic_completion_sum_l2575_257510

theorem quadratic_completion_sum (x : ℝ) : ∃ (m n : ℝ), 
  (x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_sum_l2575_257510


namespace NUMINAMATH_CALUDE_circle_center_l2575_257533

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 1 = 0, its center is (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 1 = 0) → (∃ r : ℝ, (x - 1)^2 + (y + 2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l2575_257533


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l2575_257592

theorem min_sum_with_constraint (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (h : a / x + b / y = 2) :
  x + y ≥ (a + b) / 2 + Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l2575_257592


namespace NUMINAMATH_CALUDE_simple_interest_proof_l2575_257573

/-- Given a principal amount where the compound interest for 2 years at 5% per annum is 41,
    prove that the simple interest for the same principal, rate, and time is 40. -/
theorem simple_interest_proof (P : ℝ) : 
  P * (1 + 0.05)^2 - P = 41 → P * 0.05 * 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_proof_l2575_257573


namespace NUMINAMATH_CALUDE_max_ab_value_l2575_257502

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a^2 + b^2 - 6*a = 0) :
  ∃ (max_ab : ℝ), max_ab = (27 * Real.sqrt 3) / 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + y^2 - 6*x = 0 → x*y ≤ max_ab :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2575_257502


namespace NUMINAMATH_CALUDE_max_q_plus_r_for_1057_l2575_257568

theorem max_q_plus_r_for_1057 :
  ∃ (q r : ℕ+), 1057 = 23 * q + r ∧ ∀ (q' r' : ℕ+), 1057 = 23 * q' + r' → q + r ≥ q' + r' :=
by sorry

end NUMINAMATH_CALUDE_max_q_plus_r_for_1057_l2575_257568


namespace NUMINAMATH_CALUDE_profit_share_ratio_l2575_257560

theorem profit_share_ratio (total_profit : ℕ) (difference : ℕ) : 
  total_profit = 700 → difference = 140 → 
  ∃ (x y : ℕ), x + y = total_profit ∧ x - y = difference ∧ 
  (y : ℚ) / total_profit = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l2575_257560


namespace NUMINAMATH_CALUDE_white_balls_count_l2575_257584

theorem white_balls_count (total : ℕ) (p_yellow : ℚ) (h_total : total = 32) (h_p_yellow : p_yellow = 1/4) :
  total - (total * p_yellow).floor = 24 :=
sorry

end NUMINAMATH_CALUDE_white_balls_count_l2575_257584


namespace NUMINAMATH_CALUDE_age_sum_proof_l2575_257598

theorem age_sum_proof (child_age mother_age : ℕ) : 
  child_age = 10 →
  mother_age = 3 * child_age →
  child_age + mother_age = 40 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2575_257598


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2575_257581

theorem no_solution_absolute_value_equation :
  ¬ ∃ x : ℝ, |(-2 * x)| + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2575_257581


namespace NUMINAMATH_CALUDE_violet_percentage_l2575_257530

/-- Represents a flower bouquet with yellow and purple flowers -/
structure Bouquet where
  total : ℕ
  yellow : ℕ
  purple : ℕ
  yellow_daisies : ℕ
  purple_violets : ℕ

/-- Conditions for the flower bouquet -/
def bouquet_conditions (b : Bouquet) : Prop :=
  b.total > 0 ∧
  b.yellow + b.purple = b.total ∧
  b.yellow = b.total / 2 ∧
  b.yellow_daisies = b.yellow / 5 ∧
  b.purple_violets = b.purple / 2

/-- Theorem: The percentage of violets in the bouquet is 25% -/
theorem violet_percentage (b : Bouquet) (h : bouquet_conditions b) :
  (b.purple_violets : ℚ) / b.total = 1/4 := by
  sorry

#check violet_percentage

end NUMINAMATH_CALUDE_violet_percentage_l2575_257530


namespace NUMINAMATH_CALUDE_wire_cutting_l2575_257561

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 49 →
  ratio = 2 / 5 →
  shorter_piece + ratio⁻¹ * shorter_piece = total_length →
  shorter_piece = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l2575_257561


namespace NUMINAMATH_CALUDE_library_books_l2575_257517

theorem library_books (initial_books : ℕ) : 
  (initial_books : ℚ) * (2 / 6) = 3300 → initial_books = 9900 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l2575_257517


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l2575_257596

theorem certain_fraction_proof : 
  ∃ (x y : ℚ), (x / y) / (6 / 7) = (7 / 15) / (2 / 3) ∧ x / y = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l2575_257596


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2575_257544

/-- Sam's current age -/
def s : ℕ := by sorry

/-- Anna's current age -/
def a : ℕ := by sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := by sorry

theorem age_ratio_problem :
  (s - 3 = 4 * (a - 3)) ∧ 
  (s - 5 = 6 * (a - 5)) →
  (x = 22 ∧ (s + x) * 2 = (a + x) * 3) := by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2575_257544


namespace NUMINAMATH_CALUDE_sara_initial_savings_l2575_257599

/-- Sara's initial savings -/
def S : ℕ := sorry

/-- Number of weeks -/
def weeks : ℕ := 820

/-- Sara's weekly savings -/
def sara_weekly : ℕ := 10

/-- Jim's weekly savings -/
def jim_weekly : ℕ := 15

/-- Theorem stating that Sara's initial savings is $4100 -/
theorem sara_initial_savings :
  S = 4100 ∧
  S + sara_weekly * weeks = jim_weekly * weeks :=
sorry

end NUMINAMATH_CALUDE_sara_initial_savings_l2575_257599


namespace NUMINAMATH_CALUDE_intersection_point_sum_l2575_257542

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  sorry

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (a b c : Point) : ℚ :=
  sorry

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Represents the intersection point of a line with CD -/
structure IntersectionPoint where
  p : ℕ
  q : ℕ
  r : ℕ
  s : ℕ

/-- The main theorem -/
theorem intersection_point_sum (a b c d : Point) (l : Line) (i : IntersectionPoint) :
  a = Point.mk 0 0 →
  b = Point.mk 2 4 →
  c = Point.mk 6 6 →
  d = Point.mk 8 0 →
  l.p1 = a →
  isPointOnLine (Point.mk (i.p / i.q) (i.r / i.s)) l →
  isPointOnLine (Point.mk (i.p / i.q) (i.r / i.s)) (Line.mk c d) →
  triangleArea a (Point.mk (i.p / i.q) (i.r / i.s)) d = (1/3) * quadrilateralArea a b c d →
  i.p + i.q + i.r + i.s = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l2575_257542


namespace NUMINAMATH_CALUDE_mod_inverse_sum_17_l2575_257543

theorem mod_inverse_sum_17 :
  ∃ (a b : ℤ), (2 * a) % 17 = 1 ∧ (4 * b) % 17 = 1 ∧ (a + b) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_17_l2575_257543


namespace NUMINAMATH_CALUDE_smallest_number_l2575_257590

theorem smallest_number (a b c d : ℝ) (h1 : a = 0) (h2 : b = -1) (h3 : c = -Real.sqrt 3) (h4 : d = 3) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2575_257590


namespace NUMINAMATH_CALUDE_twelve_hour_clock_chimes_90_l2575_257583

/-- Calculates the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a clock that chimes on the hour and half-hour -/
structure ChimingClock where
  hours : ℕ
  chimes_on_hour : ℕ → ℕ
  chimes_on_half_hour : ℕ

/-- Calculates the total number of chimes for a ChimingClock over its set hours -/
def total_chimes (clock : ChimingClock) : ℕ :=
  sum_to_n clock.hours + clock.hours * clock.chimes_on_half_hour

/-- Theorem stating that a clock chiming the hour count on the hour and once on the half-hour,
    over 12 hours, will chime 90 times in total -/
theorem twelve_hour_clock_chimes_90 :
  ∃ (clock : ChimingClock),
    clock.hours = 12 ∧
    clock.chimes_on_hour = id ∧
    clock.chimes_on_half_hour = 1 ∧
    total_chimes clock = 90 := by
  sorry

end NUMINAMATH_CALUDE_twelve_hour_clock_chimes_90_l2575_257583


namespace NUMINAMATH_CALUDE_cider_production_l2575_257506

theorem cider_production (golden_per_pint pink_per_pint : ℕ)
  (num_farmhands work_hours : ℕ) (total_pints : ℕ) :
  golden_per_pint = 20 →
  pink_per_pint = 40 →
  num_farmhands = 6 →
  work_hours = 5 →
  total_pints = 120 →
  (∃ (apples_per_hour : ℕ),
    apples_per_hour * num_farmhands * work_hours = 
      (golden_per_pint + pink_per_pint) * total_pints ∧
    3 * (golden_per_pint * total_pints) = 
      (golden_per_pint + pink_per_pint) * total_pints ∧
    apples_per_hour = 240) :=
by sorry

end NUMINAMATH_CALUDE_cider_production_l2575_257506


namespace NUMINAMATH_CALUDE_floor_paving_cost_l2575_257541

-- Define the room dimensions and cost per square meter
def room_length : ℝ := 5.5
def room_width : ℝ := 3.75
def cost_per_sq_meter : ℝ := 700

-- Define the function to calculate the total cost
def total_cost (length width cost_per_unit : ℝ) : ℝ :=
  length * width * cost_per_unit

-- Theorem statement
theorem floor_paving_cost :
  total_cost room_length room_width cost_per_sq_meter = 14437.50 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l2575_257541


namespace NUMINAMATH_CALUDE_triangle_with_same_color_and_unit_area_l2575_257552

-- Define a color type
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point in the plane
def colorFunction : Point → Color := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem triangle_with_same_color_and_unit_area :
  ∃ (p1 p2 p3 : Point),
    colorFunction p1 = colorFunction p2 ∧
    colorFunction p2 = colorFunction p3 ∧
    triangleArea p1 p2 p3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_same_color_and_unit_area_l2575_257552


namespace NUMINAMATH_CALUDE_salary_comparison_l2575_257507

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l2575_257507


namespace NUMINAMATH_CALUDE_find_t_l2575_257518

/-- The number of hours I worked -/
def my_hours (t : ℝ) : ℝ := 2*t + 2

/-- My hourly rate in dollars -/
def my_rate (t : ℝ) : ℝ := 4*t - 4

/-- The number of hours Emily worked -/
def emily_hours (t : ℝ) : ℝ := 4*t - 2

/-- Emily's hourly rate in dollars -/
def emily_rate (t : ℝ) : ℝ := t + 3

/-- My total earnings -/
def my_earnings (t : ℝ) : ℝ := my_hours t * my_rate t

/-- Emily's total earnings -/
def emily_earnings (t : ℝ) : ℝ := emily_hours t * emily_rate t

theorem find_t : ∃ t : ℝ, t > 0 ∧ my_earnings t = emily_earnings t + 6 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l2575_257518


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l2575_257597

theorem sum_of_reciprocal_roots (x₁ x₂ : ℝ) : 
  x₁^2 + 2*x₁ - 3 = 0 → x₂^2 + 2*x₂ - 3 = 0 → x₁ ≠ x₂ → 
  (1/x₁ + 1/x₂ : ℝ) = 2/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l2575_257597


namespace NUMINAMATH_CALUDE_carolyn_sum_is_24_l2575_257512

def game_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_removable (n : Nat) (l : List Nat) : Bool :=
  ∃ m ∈ l, m ≠ n ∧ n % m = 0

def remove_divisors (n : Nat) (l : List Nat) : List Nat :=
  l.filter (fun m => m = n ∨ n % m ≠ 0)

def carolyn_moves (l : List Nat) : List Nat :=
  let after_first_move := l.filter (· ≠ 8)
  let after_paul_first := remove_divisors 8 after_first_move
  let second_move := after_paul_first.filter (· ≠ 10)
  let after_paul_second := remove_divisors 10 second_move
  let third_move := after_paul_second.filter (· ≠ 6)
  third_move

theorem carolyn_sum_is_24 :
  let carolyn_removed := [8, 10, 6]
  carolyn_removed.sum = 24 ∧
  (∀ n ∈ carolyn_moves game_list, ¬is_removable n (carolyn_moves game_list)) := by
  sorry

end NUMINAMATH_CALUDE_carolyn_sum_is_24_l2575_257512


namespace NUMINAMATH_CALUDE_age_difference_l2575_257514

/-- Given that the sum of X and Y is 15 years greater than the sum of Y and Z,
    prove that Z is 1.5 decades younger than X. -/
theorem age_difference (X Y Z : ℕ) (h : X + Y = Y + Z + 15) :
  (X - Z : ℚ) / 10 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2575_257514


namespace NUMINAMATH_CALUDE_common_number_in_list_l2575_257559

theorem common_number_in_list (list : List ℝ) : 
  list.length = 7 →
  (list.take 4).sum / 4 = 7 →
  (list.drop 3).sum / 4 = 9 →
  list.sum / 7 = 8 →
  ∃ x ∈ list.take 4 ∩ list.drop 3, x = 8 := by
sorry

end NUMINAMATH_CALUDE_common_number_in_list_l2575_257559


namespace NUMINAMATH_CALUDE_subset_X_l2575_257525

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_subset_X_l2575_257525


namespace NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2575_257549

theorem solution_set_linear_inequalities :
  let S := {x : ℝ | x - 2 > 1 ∧ x < 4}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2575_257549


namespace NUMINAMATH_CALUDE_ratio_problem_l2575_257545

/-- Given three positive real numbers A, B, and C with specified ratios,
    prove the fraction of C to A and the ratio of A to C. -/
theorem ratio_problem (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hAB : A / B = 7 / 3) (hBC : B / C = 6 / 5) :
  C / A = 5 / 14 ∧ A / C = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2575_257545


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_geq_two_l2575_257548

theorem absolute_value_inequality_implies_a_geq_two :
  (∀ x : ℝ, |x + 3| - |x + 1| - 2*a + 2 < 0) → a ≥ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_geq_two_l2575_257548


namespace NUMINAMATH_CALUDE_seven_at_eight_equals_nineteen_thirds_l2575_257586

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := (5 * a - 2 * b) / 3

-- Theorem statement
theorem seven_at_eight_equals_nineteen_thirds :
  at_op 7 8 = 19 / 3 := by sorry

end NUMINAMATH_CALUDE_seven_at_eight_equals_nineteen_thirds_l2575_257586


namespace NUMINAMATH_CALUDE_factorization_equality_l2575_257558

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2575_257558


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2575_257524

/-- 
For a quadratic equation x^2 + kx + 1 = 0 to have two equal real roots,
k must equal ±2.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + k*y + 1 = 0 → y = x) ↔ 
  k = 2 ∨ k = -2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2575_257524


namespace NUMINAMATH_CALUDE_ellipse_axis_lengths_l2575_257520

/-- Given an ellipse with equation x²/16 + y²/25 = 1, prove that its major axis length is 10 and its minor axis length is 8 -/
theorem ellipse_axis_lengths :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/25 = 1}
  ∃ (major_axis minor_axis : ℝ),
    major_axis = 10 ∧
    minor_axis = 8 ∧
    (∀ (p : ℝ × ℝ), p ∈ ellipse →
      (p.1^2 + p.2^2 ≤ (major_axis/2)^2 ∧
       p.1^2 + p.2^2 ≥ (minor_axis/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_lengths_l2575_257520


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2575_257539

theorem purely_imaginary_condition (m : ℝ) : 
  (((2 * m ^ 2 - 3 * m - 2) : ℂ) + (m ^ 2 - 3 * m + 2) * Complex.I).im ≠ 0 ∧ 
  (((2 * m ^ 2 - 3 * m - 2) : ℂ) + (m ^ 2 - 3 * m + 2) * Complex.I).re = 0 ↔ 
  m = -1/2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2575_257539


namespace NUMINAMATH_CALUDE_projection_orthogonal_vectors_l2575_257516

/-- Given two orthogonal vectors a and b in ℝ², prove that if the projection of (4, -2) onto a
    is (1/2, 1), then the projection of (4, -2) onto b is (7/2, -3). -/
theorem projection_orthogonal_vectors (a b : ℝ × ℝ) : 
  a.1 * b.1 + a.2 * b.2 = 0 →  -- a and b are orthogonal
  (∃ k : ℝ, k • a = (1/2, 1) ∧ k * (a.1 * 4 + a.2 * (-2)) = a.1^2 + a.2^2) →  -- proj_a (4, -2) = (1/2, 1)
  (∃ m : ℝ, m • b = (7/2, -3) ∧ m * (b.1 * 4 + b.2 * (-2)) = b.1^2 + b.2^2)  -- proj_b (4, -2) = (7/2, -3)
  := by sorry

end NUMINAMATH_CALUDE_projection_orthogonal_vectors_l2575_257516


namespace NUMINAMATH_CALUDE_hannah_remaining_money_l2575_257569

def county_fair_expenses (initial_amount : ℚ) (ride_percentage : ℚ) (game_percentage : ℚ)
  (dessert_cost : ℚ) (cotton_candy_cost : ℚ) (hotdog_cost : ℚ)
  (keychain_cost : ℚ) (poster_cost : ℚ) (attraction_cost : ℚ) : ℚ :=
  let ride_expense := initial_amount * ride_percentage
  let game_expense := initial_amount * game_percentage
  let food_souvenir_expense := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + attraction_cost
  initial_amount - (ride_expense + game_expense + food_souvenir_expense)

theorem hannah_remaining_money :
  county_fair_expenses 120 0.4 0.15 8 5 6 7 10 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_remaining_money_l2575_257569


namespace NUMINAMATH_CALUDE_pascal_ratio_row_l2575_257576

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Three consecutive entries in Pascal's Triangle are in ratio 3:4:5 -/
def ratio_condition (n : ℕ) (r : ℕ) : Prop :=
  ∃ (a b c : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a * (pascal n (r+1)) = b * (pascal n r) ∧
    b * (pascal n (r+2)) = c * (pascal n (r+1)) ∧
    3 * b = 4 * a ∧ 4 * c = 5 * b

theorem pascal_ratio_row :
  ∃ (n : ℕ), n = 62 ∧ ∃ (r : ℕ), ratio_condition n r :=
sorry

end NUMINAMATH_CALUDE_pascal_ratio_row_l2575_257576


namespace NUMINAMATH_CALUDE_work_speed_ratio_is_two_to_one_l2575_257504

def work_speed_ratio (a b : ℚ) : Prop :=
  b = 1 / 12 ∧ a + b = 1 / 4 → a / b = 2

theorem work_speed_ratio_is_two_to_one :
  ∃ a b : ℚ, work_speed_ratio a b :=
by
  sorry

end NUMINAMATH_CALUDE_work_speed_ratio_is_two_to_one_l2575_257504


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l2575_257537

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def graded_worksheets : ℕ := 5

theorem problems_left_to_grade :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 := by
  sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l2575_257537


namespace NUMINAMATH_CALUDE_integral_x_plus_sqrt_one_minus_x_squared_l2575_257528

open Set
open MeasureTheory
open Interval
open Real

theorem integral_x_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1 : ℝ)..1, (x + Real.sqrt (1 - x^2)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_sqrt_one_minus_x_squared_l2575_257528


namespace NUMINAMATH_CALUDE_positive_number_and_square_sum_l2575_257594

theorem positive_number_and_square_sum : ∃ (n : ℝ), n > 0 ∧ n^2 + n = 210 ∧ n = 14 ∧ n^3 = 2744 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_and_square_sum_l2575_257594


namespace NUMINAMATH_CALUDE_work_completion_time_l2575_257509

/-- Worker rates and work completion time -/
theorem work_completion_time
  (rate_a rate_b rate_c rate_d : ℝ)
  (total_work : ℝ)
  (h1 : rate_a = 1.5 * rate_b)
  (h2 : rate_a * 30 = total_work)
  (h3 : rate_c = 2 * rate_b)
  (h4 : rate_d = 0.5 * rate_a)
  : ∃ (days : ℕ), days = 12 ∧ 
    (1.25 * rate_b + 2.75 * rate_b) * (days : ℝ) ≥ total_work ∧
    (1.25 * rate_b + 2.75 * rate_b) * ((days - 1) : ℝ) < total_work :=
by sorry


end NUMINAMATH_CALUDE_work_completion_time_l2575_257509


namespace NUMINAMATH_CALUDE_league_games_l2575_257582

theorem league_games (n : ℕ) (h : n = 10) : (n.choose 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l2575_257582


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_l2575_257562

/-- The minimum distance sum for a point on the parabola x = (1/4)y^2 -/
theorem min_distance_sum_parabola :
  let parabola := {P : ℝ × ℝ | P.1 = (1/4) * P.2^2}
  let dist_to_A (P : ℝ × ℝ) := Real.sqrt ((P.1 - 0)^2 + (P.2 - 1)^2)
  let dist_to_y_axis (P : ℝ × ℝ) := |P.1|
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 - 1 ∧
    ∀ P ∈ parabola, dist_to_A P + dist_to_y_axis P ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_parabola_l2575_257562


namespace NUMINAMATH_CALUDE_cosine_range_in_geometric_progression_triangle_l2575_257532

theorem cosine_range_in_geometric_progression_triangle (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (hacute : 0 < a ^ 2 + b ^ 2 - c ^ 2 ∧ 0 < b ^ 2 + c ^ 2 - a ^ 2 ∧ 0 < c ^ 2 + a ^ 2 - b ^ 2)
  (hgeo : b ^ 2 = a * c) : 1 / 2 ≤ (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) ∧ (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) < 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_range_in_geometric_progression_triangle_l2575_257532


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2575_257536

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2575_257536


namespace NUMINAMATH_CALUDE_f_at_one_l2575_257501

/-- Given a polynomial g(x) with three distinct roots, where each root is also a root of f(x),
    prove that f(1) = -217 -/
theorem f_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    (∀ x : ℝ, x^3 + a*x^2 + x + 20 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  (∀ x : ℝ, x^3 + a*x^2 + x + 20 = 0 → x^4 + x^3 + b*x^2 + 50*x + c = 0) →
  (1 : ℝ)^4 + (1 : ℝ)^3 + b*(1 : ℝ)^2 + 50*(1 : ℝ) + c = -217 :=
by sorry

end NUMINAMATH_CALUDE_f_at_one_l2575_257501


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l2575_257579

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l2575_257579


namespace NUMINAMATH_CALUDE_four_by_four_min_cuts_five_by_five_min_cuts_l2575_257556

/-- Represents a square grid of size n x n -/
structure Square (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Minimum number of cuts required to divide a square into unit squares -/
def min_cuts (s : Square n) : ℕ :=
  sorry

/-- Pieces can be overlapped during cutting -/
axiom overlap_allowed : ∀ (n : ℕ) (s : Square n), True

theorem four_by_four_min_cuts :
  ∀ (s : Square 4), min_cuts s = 4 :=
sorry

theorem five_by_five_min_cuts :
  ∀ (s : Square 5), min_cuts s = 6 :=
sorry

end NUMINAMATH_CALUDE_four_by_four_min_cuts_five_by_five_min_cuts_l2575_257556


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2575_257521

theorem quadratic_roots_sum_of_squares (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ 
    x₂^2 - m*x₂ + 2*m - 1 = 0 ∧
    x₁^2 + x₂^2 = 7) → 
  m = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2575_257521


namespace NUMINAMATH_CALUDE_completing_square_transformation_l2575_257571

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l2575_257571


namespace NUMINAMATH_CALUDE_product_of_solutions_l2575_257567

theorem product_of_solutions (x : ℝ) : 
  (|18 / x - 4| = 3) → (∃ y : ℝ, (|18 / y - 4| = 3) ∧ (x * y = 324 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2575_257567


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2575_257580

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ f (-1)) ∧  -- f(-1) is the minimum value
  f (-1) = -3 ∧          -- f(-1) = -3
  f 1 = 5                -- f(1) = 5
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2575_257580


namespace NUMINAMATH_CALUDE_chord_intersection_probability_for_1996_points_chord_intersection_probability_general_l2575_257515

/-- The number of points on the circle -/
def n : ℕ := 1996

/-- The probability that two chords formed by four randomly selected points intersect -/
def chord_intersection_probability (n : ℕ) : ℚ :=
  if n ≥ 4 then 1 / 4 else 0

/-- Theorem stating that the probability of chord intersection is 1/4 for 1996 points -/
theorem chord_intersection_probability_for_1996_points :
  chord_intersection_probability n = 1 / 4 := by
  sorry

/-- Theorem stating that the probability of chord intersection is always 1/4 for n ≥ 4 -/
theorem chord_intersection_probability_general (n : ℕ) (h : n ≥ 4) :
  chord_intersection_probability n = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_for_1996_points_chord_intersection_probability_general_l2575_257515


namespace NUMINAMATH_CALUDE_a_range_l2575_257564

theorem a_range (a : ℝ) : 
  (a + 1)^(-1/4 : ℝ) < (3 - 2*a)^(-1/4 : ℝ) → 2/3 < a ∧ a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_a_range_l2575_257564


namespace NUMINAMATH_CALUDE_solution_theorem_l2575_257572

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 - i)^2 * z = 3 + 2*i

-- State the theorem
theorem solution_theorem :
  ∃ (z : ℂ), given_equation z ∧ z = -1 + (3/2) * i :=
sorry

end NUMINAMATH_CALUDE_solution_theorem_l2575_257572
