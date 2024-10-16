import Mathlib

namespace NUMINAMATH_CALUDE_fuse_probability_l1933_193330

/-- The probability of the union of two events -/
def prob_union (prob_A prob_B prob_A_and_B : ℝ) : ℝ :=
  prob_A + prob_B - prob_A_and_B

theorem fuse_probability (prob_A prob_B prob_A_and_B : ℝ) 
  (h1 : prob_A = 0.085)
  (h2 : prob_B = 0.074)
  (h3 : prob_A_and_B = 0.063) :
  prob_union prob_A prob_B prob_A_and_B = 0.096 := by
sorry

end NUMINAMATH_CALUDE_fuse_probability_l1933_193330


namespace NUMINAMATH_CALUDE_vector_magnitude_l1933_193396

/-- Given two planar vectors a and b, prove that |2a - b| = 2√3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 3/5 ∧ a.2 = -4/5) →  -- Vector a = (3/5, -4/5)
  (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1) →  -- |a| = 1
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 2) →  -- |b| = 2
  (a.1 * b.1 + a.2 * b.2 = -1) →  -- a · b = -1 (dot product for 120° angle)
  Real.sqrt (((2 * a.1 - b.1) ^ 2) + ((2 * a.2 - b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1933_193396


namespace NUMINAMATH_CALUDE_quotient_problem_l1933_193333

theorem quotient_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ n : ℤ, a / b = n) (h4 : a / b = a / 2 ∨ a / b = 6 * b) : 
  a / b = 12 := by
sorry

end NUMINAMATH_CALUDE_quotient_problem_l1933_193333


namespace NUMINAMATH_CALUDE_converse_abs_inequality_l1933_193302

theorem converse_abs_inequality (x y : ℝ) : x > |y| → x > y := by sorry

end NUMINAMATH_CALUDE_converse_abs_inequality_l1933_193302


namespace NUMINAMATH_CALUDE_triangle_proof_l1933_193318

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    and vectors m and n, prove that C = π/3 and if a^2 = 2b^2 + c^2, then tan(A) = -3√3 --/
theorem triangle_proof (a b c A B C : Real) (m n : Real × Real) :
  let m_x := 2 * Real.cos (C / 2)
  let m_y := -Real.sin C
  let n_x := Real.cos (C / 2)
  let n_y := 2 * Real.sin C
  m = (m_x, m_y) →
  n = (n_x, n_y) →
  m.1 * n.1 + m.2 * n.2 = 0 →  -- m ⊥ n
  (C = Real.pi / 3 ∧ (a^2 = 2*b^2 + c^2 → Real.tan A = -3 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l1933_193318


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l1933_193388

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l1933_193388


namespace NUMINAMATH_CALUDE_theater_seat_count_l1933_193368

/-- The number of seats in a theater -/
def theater_seats (people_watching : ℕ) (empty_seats : ℕ) : ℕ :=
  people_watching + empty_seats

/-- Theorem: The theater has 750 seats -/
theorem theater_seat_count : theater_seats 532 218 = 750 := by
  sorry

end NUMINAMATH_CALUDE_theater_seat_count_l1933_193368


namespace NUMINAMATH_CALUDE_expression_simplification_l1933_193319

theorem expression_simplification (a : ℝ) (h : a^2 - a - 2 = 0) :
  (1 + 1/a) / ((a^2 - 1)/a) - (2*a - 2)/(a^2 - 2*a + 1) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1933_193319


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_49_l1933_193354

theorem quadratic_expression_equals_49 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_49_l1933_193354


namespace NUMINAMATH_CALUDE_train_length_l1933_193371

/-- The length of a train given its speed and time to pass a fixed point --/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 ∧ time = 28 → speed * time * (1000 / 3600) = 280 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1933_193371


namespace NUMINAMATH_CALUDE_special_triangle_st_length_l1933_193399

/-- Triangle with given side lengths and a line segment parallel to one side passing through the incenter --/
structure SpecialTriangle where
  -- Side lengths of the triangle
  pq : ℝ
  pr : ℝ
  qr : ℝ
  -- Points S and T on sides PQ and PR respectively
  s : ℝ  -- distance PS
  t : ℝ  -- distance PT
  -- Conditions
  pq_positive : pq > 0
  pr_positive : pr > 0
  qr_positive : qr > 0
  s_on_pq : 0 < s ∧ s < pq
  t_on_pr : 0 < t ∧ t < pr
  st_parallel_qr : True  -- We can't directly express this geometric condition
  st_contains_incenter : True  -- We can't directly express this geometric condition

/-- The theorem stating that in the special triangle, ST has a specific value --/
theorem special_triangle_st_length (tri : SpecialTriangle) 
    (h_pq : tri.pq = 26) 
    (h_pr : tri.pr = 28) 
    (h_qr : tri.qr = 30) : 
  (tri.s - 0) / tri.pq + (tri.t - 0) / tri.pr = 135 / 7 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_st_length_l1933_193399


namespace NUMINAMATH_CALUDE_marble_bag_size_l1933_193350

/-- Represents a bag of marbles with blue, red, and white colors. -/
structure MarbleBag where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ

/-- The probability of selecting a red or white marble from the bag. -/
def redOrWhiteProbability (bag : MarbleBag) : ℚ :=
  (bag.red + bag.white : ℚ) / bag.total

theorem marble_bag_size :
  ∃ (bag : MarbleBag),
    bag.blue = 5 ∧
    bag.red = 7 ∧
    redOrWhiteProbability bag = 3/4 ∧
    bag.total = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_bag_size_l1933_193350


namespace NUMINAMATH_CALUDE_conference_trip_distance_l1933_193343

/-- Conference Trip Problem -/
theorem conference_trip_distance :
  ∀ (d : ℝ) (t : ℝ),
    -- Initial speed
    let v₁ : ℝ := 40
    -- Speed increase
    let v₂ : ℝ := 20
    -- Time late if continued at initial speed
    let t_late : ℝ := 0.75
    -- Time early with speed increase
    let t_early : ℝ := 0.25
    -- Distance equation at initial speed
    d = v₁ * (t + t_late) →
    -- Distance equation with speed increase
    d - v₁ = (v₁ + v₂) * (t - 1 - t_early) →
    -- Conclusion: distance is 160 miles
    d = 160 := by
  sorry

end NUMINAMATH_CALUDE_conference_trip_distance_l1933_193343


namespace NUMINAMATH_CALUDE_smallest_divisible_by_20_and_36_l1933_193332

theorem smallest_divisible_by_20_and_36 : ∃ n : ℕ, n > 0 ∧ 20 ∣ n ∧ 36 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 20 ∣ m ∧ 36 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_20_and_36_l1933_193332


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1933_193374

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > n → ¬((m + 12) ∣ (m^3 + 160))) ∧ 
  ((n + 12) ∣ (n^3 + 160)) ∧ n = 1748 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1933_193374


namespace NUMINAMATH_CALUDE_service_cost_calculation_l1933_193316

/-- The service cost per vehicle at a fuel station -/
def service_cost_per_vehicle : ℝ := 2.20

/-- The cost of fuel per liter -/
def fuel_cost_per_liter : ℝ := 0.70

/-- The capacity of a mini-van's fuel tank in liters -/
def minivan_tank_capacity : ℝ := 65

/-- The capacity of a truck's fuel tank in liters -/
def truck_tank_capacity : ℝ := minivan_tank_capacity * 2.2

/-- The number of mini-vans filled up -/
def num_minivans : ℕ := 3

/-- The number of trucks filled up -/
def num_trucks : ℕ := 2

/-- The total cost for filling up all vehicles -/
def total_cost : ℝ := 347.7

/-- Theorem stating that the service cost per vehicle is correct given the problem conditions -/
theorem service_cost_calculation :
  service_cost_per_vehicle * (num_minivans + num_trucks : ℝ) +
  fuel_cost_per_liter * (num_minivans * minivan_tank_capacity + num_trucks * truck_tank_capacity) =
  total_cost :=
by sorry

end NUMINAMATH_CALUDE_service_cost_calculation_l1933_193316


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1933_193397

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * Real.log x

theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 + 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1933_193397


namespace NUMINAMATH_CALUDE_binomial_60_3_l1933_193345

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1933_193345


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l1933_193349

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (previousFine : ℚ) : ℚ :=
  min (previousFine + 0.3) (previousFine * 2)

/-- Calculates the fine for a given number of days overdue -/
def fineFordaysOverdue (days : ℕ) : ℚ :=
  match days with
  | 0 => 0
  | 1 => 0.05
  | n + 1 => nextDayFine (fineFordaysOverdue n)

theorem fine_on_fifth_day :
  fineFordaysOverdue 5 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l1933_193349


namespace NUMINAMATH_CALUDE_bounded_by_one_l1933_193373

/-- A function from integers to reals satisfying certain properties -/
def IntToRealFunction (f : ℤ → ℝ) : Prop :=
  (∀ n, f n ≥ 0) ∧ 
  (∀ m n, f (m * n) = f m * f n) ∧ 
  (∀ m n, f (m + n) ≤ max (f m) (f n))

/-- Theorem stating that any function satisfying IntToRealFunction is bounded above by 1 -/
theorem bounded_by_one (f : ℤ → ℝ) (hf : IntToRealFunction f) : 
  ∀ n, f n ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bounded_by_one_l1933_193373


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1933_193356

theorem imaginary_part_of_complex_product : Complex.im ((3 - 2 * Complex.I) * (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1933_193356


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1933_193362

def number_of_people : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_pair_together (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem seating_arrangements_with_restriction :
  total_arrangements number_of_people - arrangements_with_pair_together number_of_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1933_193362


namespace NUMINAMATH_CALUDE_sqrt_equation_l1933_193307

theorem sqrt_equation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l1933_193307


namespace NUMINAMATH_CALUDE_candy_problem_l1933_193369

theorem candy_problem :
  ∀ (N : ℕ) (S : ℕ),
    N > 0 →
    (∀ i : Fin N, ∃ (a : ℕ), a > 1 ∧ a = S - (N - 1) * a - 7) →
    S = 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l1933_193369


namespace NUMINAMATH_CALUDE_range_of_a_l1933_193323

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.cos x₀ ∧ f a (f a y₀) = y₀) →
  2 ≤ a ∧ a ≤ Real.exp 1 + 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1933_193323


namespace NUMINAMATH_CALUDE_soccer_team_strikers_l1933_193315

theorem soccer_team_strikers 
  (total_players : ℕ) 
  (goalies : ℕ) 
  (defenders : ℕ) 
  (midfielders : ℕ) 
  (strikers : ℕ) :
  total_players = 40 →
  goalies = 3 →
  defenders = 10 →
  midfielders = 2 * defenders →
  total_players = goalies + defenders + midfielders + strikers →
  strikers = 7 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_strikers_l1933_193315


namespace NUMINAMATH_CALUDE_mean_variance_preserved_l1933_193310

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
def new_set : List Int := [-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5]

def mean (s : List Int) : ℚ :=
  (s.sum : ℚ) / s.length

def variance (s : List Int) : ℚ :=
  let m := mean s
  (s.map (fun x => ((x : ℚ) - m) ^ 2)).sum / s.length

theorem mean_variance_preserved :
  mean initial_set = mean new_set ∧
  variance initial_set = variance new_set := by
  sorry

#eval mean initial_set
#eval mean new_set
#eval variance initial_set
#eval variance new_set

end NUMINAMATH_CALUDE_mean_variance_preserved_l1933_193310


namespace NUMINAMATH_CALUDE_password_decryption_probability_l1933_193344

theorem password_decryption_probability 
  (p : ℝ) 
  (hp : p = 1 / 4) 
  (n : ℕ) 
  (hn : n = 3) :
  (n.choose 2 : ℝ) * p^2 * (1 - p) = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l1933_193344


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_root_l1933_193335

theorem quadratic_equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_root_l1933_193335


namespace NUMINAMATH_CALUDE_jason_commute_distance_l1933_193326

/-- Represents Jason's commute with convenience stores and a detour --/
structure JasonCommute where
  distance_house_to_first : ℝ
  distance_first_to_second : ℝ
  distance_second_to_third : ℝ
  distance_third_to_work : ℝ
  detour_distance : ℝ

/-- Calculates the total commute distance with detour --/
def total_commute_with_detour (j : JasonCommute) : ℝ :=
  j.distance_house_to_first + j.distance_first_to_second + 
  (j.distance_second_to_third + j.detour_distance) + j.distance_third_to_work

/-- Theorem stating Jason's commute distance with detour --/
theorem jason_commute_distance :
  ∀ j : JasonCommute,
  j.distance_house_to_first = 4 →
  j.distance_first_to_second = 6 →
  j.distance_second_to_third = j.distance_first_to_second + (2/3 * j.distance_first_to_second) →
  j.distance_third_to_work = j.distance_house_to_first →
  j.detour_distance = 3 →
  total_commute_with_detour j = 27 := by
  sorry

end NUMINAMATH_CALUDE_jason_commute_distance_l1933_193326


namespace NUMINAMATH_CALUDE_fraction_transformation_l1933_193363

theorem fraction_transformation (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 2 / 5 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1933_193363


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1933_193320

theorem rationalize_denominator : 
  (35 - Real.sqrt 35) / Real.sqrt 35 = Real.sqrt 35 - 1 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1933_193320


namespace NUMINAMATH_CALUDE_gcd_876543_765432_l1933_193317

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_876543_765432_l1933_193317


namespace NUMINAMATH_CALUDE_root_difference_squared_l1933_193376

theorem root_difference_squared (f g : ℝ) : 
  (6 * f^2 + 13 * f - 28 = 0) → 
  (6 * g^2 + 13 * g - 28 = 0) → 
  (f - g)^2 = 169 / 9 := by
sorry

end NUMINAMATH_CALUDE_root_difference_squared_l1933_193376


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1933_193327

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1933_193327


namespace NUMINAMATH_CALUDE_divisible_integers_count_l1933_193365

-- Define the range of integers
def lower_bound : ℕ := 2000
def upper_bound : ℕ := 3000

-- Define the factors
def factor1 : ℕ := 30
def factor2 : ℕ := 45
def factor3 : ℕ := 75

-- Function to count integers in the range divisible by all factors
def count_divisible_integers : ℕ := sorry

-- Theorem statement
theorem divisible_integers_count : count_divisible_integers = 2 := by sorry

end NUMINAMATH_CALUDE_divisible_integers_count_l1933_193365


namespace NUMINAMATH_CALUDE_star_value_l1933_193382

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

/-- Theorem: If a + b = 15 and a * b = 36, then a * b = 5/12 -/
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum : a + b = 15) (product : a * b = 36) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1933_193382


namespace NUMINAMATH_CALUDE_man_walking_time_l1933_193367

theorem man_walking_time (usual_time : ℝ) (reduced_time : ℝ) : 
  reduced_time = usual_time + 24 →
  (1 : ℝ) / 0.4 = reduced_time / usual_time →
  usual_time = 16 := by
sorry

end NUMINAMATH_CALUDE_man_walking_time_l1933_193367


namespace NUMINAMATH_CALUDE_joans_remaining_apples_l1933_193377

/-- Given that Joan picked a certain number of apples and gave some away,
    this theorem proves how many apples Joan has left. -/
theorem joans_remaining_apples 
  (apples_picked : ℕ) 
  (apples_given_away : ℕ) 
  (h1 : apples_picked = 43)
  (h2 : apples_given_away = 27) :
  apples_picked - apples_given_away = 16 := by
sorry

end NUMINAMATH_CALUDE_joans_remaining_apples_l1933_193377


namespace NUMINAMATH_CALUDE_mabels_garden_petal_count_l1933_193358

/-- The number of petals remaining in Mabel's garden after a series of events -/
def final_petal_count (initial_daisies : ℕ) (initial_petals_per_daisy : ℕ) 
  (daisies_given_away : ℕ) (new_daisies : ℕ) (new_petals_per_daisy : ℕ) 
  (petals_lost_new_daisies : ℕ) (petals_lost_original_daisies : ℕ) : ℕ :=
  let initial_petals := initial_daisies * initial_petals_per_daisy
  let remaining_petals := initial_petals - (daisies_given_away * initial_petals_per_daisy)
  let new_petals := new_daisies * new_petals_per_daisy
  let total_petals := remaining_petals + new_petals
  total_petals - (petals_lost_new_daisies + petals_lost_original_daisies)

/-- Theorem stating that the final petal count in Mabel's garden is 39 -/
theorem mabels_garden_petal_count :
  final_petal_count 5 8 2 3 7 4 2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_mabels_garden_petal_count_l1933_193358


namespace NUMINAMATH_CALUDE_secret_number_probability_l1933_193387

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  Odd (tens_digit n) ∧
  Even (units_digit n) ∧
  (units_digit n) % 3 = 0 ∧
  n > 75

theorem secret_number_probability :
  ∃! (valid_numbers : Finset ℕ),
    (∀ n, n ∈ valid_numbers ↔ satisfies_conditions n) ∧
    valid_numbers.card = 3 :=
sorry

end NUMINAMATH_CALUDE_secret_number_probability_l1933_193387


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_specific_line_equation_l1933_193375

/-- The equation of a line passing through a given point with a given inclination angle -/
theorem line_equation_through_point_with_angle (x₀ y₀ : ℝ) (θ : ℝ) :
  (x₀ = Real.sqrt 3) →
  (y₀ = -2 * Real.sqrt 3) →
  (θ = 135 * π / 180) →
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧
                 ∀ (x y : ℝ), a * x + b * y + c = 0 ↔
                               y - y₀ = Real.tan θ * (x - x₀) :=
by sorry

/-- The specific equation of the line in the problem -/
theorem specific_line_equation :
  ∃ (x y : ℝ), x + y + Real.sqrt 3 = 0 ↔
                y - (-2 * Real.sqrt 3) = Real.tan (135 * π / 180) * (x - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_specific_line_equation_l1933_193375


namespace NUMINAMATH_CALUDE_orange_probability_l1933_193300

theorem orange_probability (total : ℕ) (large : ℕ) (small : ℕ) (choose : ℕ) :
  total = 8 →
  large = 5 →
  small = 3 →
  choose = 3 →
  (Nat.choose small choose : ℚ) / (Nat.choose total choose : ℚ) = 1 / 56 :=
by sorry

end NUMINAMATH_CALUDE_orange_probability_l1933_193300


namespace NUMINAMATH_CALUDE_last_three_digits_of_product_cubed_l1933_193346

theorem last_three_digits_of_product_cubed : 
  (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 3 % 1000 = 976 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_product_cubed_l1933_193346


namespace NUMINAMATH_CALUDE_toy_store_fraction_l1933_193380

theorem toy_store_fraction (weekly_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_store_amount : ℚ) :
  weekly_allowance = 3 →
  arcade_fraction = 2/5 →
  candy_store_amount = 6/5 →
  let remaining_after_arcade := weekly_allowance - arcade_fraction * weekly_allowance
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
sorry

end NUMINAMATH_CALUDE_toy_store_fraction_l1933_193380


namespace NUMINAMATH_CALUDE_max_length_sum_xy_l1933_193386

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- Given the constraints, the maximum sum of lengths of x and y is 15. -/
theorem max_length_sum_xy : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x + 3*y < 920 ∧ 
  ∀ (a b : ℕ), a > 1 → b > 1 → a + 3*b < 920 → 
  length x + length y ≥ length a + length b ∧
  length x + length y = 15 := by
sorry

end NUMINAMATH_CALUDE_max_length_sum_xy_l1933_193386


namespace NUMINAMATH_CALUDE_units_digit_of_15_to_15_l1933_193370

theorem units_digit_of_15_to_15 : ∃ n : ℕ, 15^15 ≡ 5 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_15_to_15_l1933_193370


namespace NUMINAMATH_CALUDE_slower_train_speed_l1933_193394

/-- Calculates the speed of the slower train given the conditions of two trains moving in the same direction. -/
theorem slower_train_speed
  (faster_train_speed : ℝ)
  (faster_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_train_speed = 72)
  (h2 : faster_train_length = 70)
  (h3 : crossing_time = 7)
  : ∃ (slower_train_speed : ℝ), slower_train_speed = 36 :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l1933_193394


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l1933_193331

theorem unique_sequence_existence : ∃! a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))) :=
by sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l1933_193331


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1933_193324

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

theorem parallel_vectors_m_value :
  (∃ (k : ℝ), k ≠ 0 ∧ b m = k • a) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1933_193324


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1933_193361

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^12 - 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1933_193361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l1933_193309

/-- For an arithmetic sequence with common difference d ≠ 0, 
    if a_3 + a_9 = a_10 - a_8, then a_n = 0 when n = 5 -/
theorem arithmetic_sequence_zero_term 
  (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_d_neq_0 : d ≠ 0) 
  (h_eq : a 3 + a 9 = a 10 - a 8) :
  ∃ n, a n = 0 ∧ n = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l1933_193309


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1933_193347

theorem sufficient_not_necessary (a b : ℝ) :
  (a > b ∧ b > 0) → (1 / a < 1 / b) ∧
  ¬ ((1 / a < 1 / b) → (a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1933_193347


namespace NUMINAMATH_CALUDE_roger_donated_66_coins_l1933_193372

/-- Represents the number of coins Roger donated -/
def coins_donated (pennies nickels dimes coins_left : ℕ) : ℕ :=
  pennies + nickels + dimes - coins_left

/-- Proves that Roger donated 66 coins given the initial counts and remaining coins -/
theorem roger_donated_66_coins (h1 : coins_donated 42 36 15 27 = 66) : 
  coins_donated 42 36 15 27 = 66 := by
  sorry

end NUMINAMATH_CALUDE_roger_donated_66_coins_l1933_193372


namespace NUMINAMATH_CALUDE_units_digit_base_6_l1933_193308

theorem units_digit_base_6 : ∃ (n : ℕ), (67^2 * 324) = 6 * n :=
sorry

end NUMINAMATH_CALUDE_units_digit_base_6_l1933_193308


namespace NUMINAMATH_CALUDE_total_jump_sequences_l1933_193311

-- Define a regular hexagon
structure RegularHexagon :=
  (vertices : Fin 6 → Point)

-- Define a frog's jump
inductive Jump
| clockwise
| counterclockwise

-- Define a sequence of jumps
def JumpSequence := List Jump

-- Define the result of a jump sequence
inductive JumpResult
| reachedD
| notReachedD

-- Function to determine the result of a jump sequence
def jumpSequenceResult (h : RegularHexagon) (js : JumpSequence) : JumpResult :=
  sorry

-- Function to count valid jump sequences
def countValidJumpSequences (h : RegularHexagon) : Nat :=
  sorry

-- The main theorem
theorem total_jump_sequences (h : RegularHexagon) :
  countValidJumpSequences h = 26 :=
sorry

end NUMINAMATH_CALUDE_total_jump_sequences_l1933_193311


namespace NUMINAMATH_CALUDE_transformation_result_l1933_193312

def rotate_180_degrees (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

def reflect_y_equals_x (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, point.1)

theorem transformation_result (Q : ℝ × ℝ) :
  let rotated := rotate_180_degrees (2, 3) Q
  let reflected := reflect_y_equals_x rotated
  reflected = (4, -1) → Q.1 - Q.2 = 3 := by
sorry

end NUMINAMATH_CALUDE_transformation_result_l1933_193312


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_perimeter_l1933_193321

theorem rectangle_length_from_square_perimeter (square_side : ℝ) (rect_width : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  4 * square_side = 2 * (rect_width + (18 : ℝ)) :=
by
  sorry

#check rectangle_length_from_square_perimeter

end NUMINAMATH_CALUDE_rectangle_length_from_square_perimeter_l1933_193321


namespace NUMINAMATH_CALUDE_circumcenter_rational_coords_l1933_193329

/-- If the coordinates of the vertices of a triangle are rational, 
    then the coordinates of the center of its circumscribed circle are also rational. -/
theorem circumcenter_rational_coords 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) : ∃ x y : ℚ, 
  (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
  (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 := by
  sorry

end NUMINAMATH_CALUDE_circumcenter_rational_coords_l1933_193329


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l1933_193364

/-- Represents a multiple choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- The number of ways to complete a test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question,
    there is exactly 1 way to complete it with all questions unanswered -/
theorem unanswered_test_completion_ways 
  (test : MultipleChoiceTest)
  (h1 : test.num_questions = 4)
  (h2 : test.choices_per_question = 5) :
  ways_to_complete_unanswered test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l1933_193364


namespace NUMINAMATH_CALUDE_vector_colinearity_l1933_193395

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 0)
def c : ℝ × ℝ := (2, 1)

theorem vector_colinearity (k : ℝ) :
  (∃ t : ℝ, t ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = (t * c.1, t * c.2)) →
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_colinearity_l1933_193395


namespace NUMINAMATH_CALUDE_basketball_surface_area_l1933_193383

/-- The surface area of a sphere with diameter 24 centimeters is 576π square centimeters. -/
theorem basketball_surface_area : 
  let diameter : ℝ := 24
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 576 * Real.pi := by sorry

end NUMINAMATH_CALUDE_basketball_surface_area_l1933_193383


namespace NUMINAMATH_CALUDE_log_system_solution_l1933_193334

theorem log_system_solution :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) →
  (x^2 - 5*y^2 + 4 = 0) →
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_log_system_solution_l1933_193334


namespace NUMINAMATH_CALUDE_equation_equivalence_l1933_193366

theorem equation_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : 2*b - a ≠ 0) :
  (a + 2*b) / a = b / (2*b - a) ↔ 
  (a = -b * ((1 + Real.sqrt 17) / 2) ∨ a = -b * ((1 - Real.sqrt 17) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1933_193366


namespace NUMINAMATH_CALUDE_total_stars_l1933_193389

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) : 
  num_students = 124 → stars_per_student = 3 → num_students * stars_per_student = 372 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_l1933_193389


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l1933_193339

variable (a : ℝ)

def f (x : ℝ) := (x - 1)^2 + 2*a*x + 1

theorem decreasing_function_implies_a_bound :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 4 → f a x₁ > f a x₂) →
  a ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l1933_193339


namespace NUMINAMATH_CALUDE_message_clearing_time_l1933_193390

/-- The number of days needed to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  (initial_messages + new_per_day - 1) / (read_per_day - new_per_day)

/-- Theorem stating that it takes 88 days to clear messages under given conditions -/
theorem message_clearing_time : days_to_clear_messages 350 22 18 = 88 := by
  sorry

end NUMINAMATH_CALUDE_message_clearing_time_l1933_193390


namespace NUMINAMATH_CALUDE_sequence_properties_l1933_193357

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a n + 1

def c (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

def T (n : ℕ) : ℚ := n / (6 * n + 9)

theorem sequence_properties :
  (∀ n : ℕ, a n + 1 = 2^(n + 1)) ∧
  (∀ n : ℕ, a n = 2^(n + 1) - 1) ∧
  (∀ n : ℕ, T n = n / (6 * n + 9)) ∧
  (∀ n : ℕ+, T n > 1 / a 5) ∧
  (∀ m : ℕ+, m < 5 → ∃ n : ℕ+, T n ≤ 1 / a m) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1933_193357


namespace NUMINAMATH_CALUDE_squares_ending_in_nine_l1933_193341

theorem squares_ending_in_nine (x : ℤ) :
  (x ^ 2) % 10 = 9 ↔ ∃ a : ℤ, (x = 10 * a + 3 ∨ x = 10 * a + 7) :=
by sorry

end NUMINAMATH_CALUDE_squares_ending_in_nine_l1933_193341


namespace NUMINAMATH_CALUDE_smallest_equivalent_angle_proof_l1933_193303

/-- The smallest positive angle in [0°, 360°) with the same terminal side as 2011° -/
def smallest_equivalent_angle : ℝ := 211

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem smallest_equivalent_angle_proof :
  same_terminal_side smallest_equivalent_angle 2011 ∧
  smallest_equivalent_angle ≥ 0 ∧
  smallest_equivalent_angle < 360 ∧
  ∀ θ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ 2011 → θ ≥ smallest_equivalent_angle := by
  sorry


end NUMINAMATH_CALUDE_smallest_equivalent_angle_proof_l1933_193303


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1933_193381

/-- Two arithmetic sequences and their sums -/
structure ArithmeticSequences where
  a : ℕ → ℚ
  b : ℕ → ℚ
  S : ℕ → ℚ
  T : ℕ → ℚ

/-- The main theorem -/
theorem arithmetic_sequences_ratio 
  (seq : ArithmeticSequences)
  (h : ∀ n, seq.S n / seq.T n = (3 * n - 1) / (2 * n + 3)) :
  seq.a 7 / seq.b 7 = 38 / 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1933_193381


namespace NUMINAMATH_CALUDE_factorization_of_4a_minus_a_cubed_l1933_193340

theorem factorization_of_4a_minus_a_cubed (a : ℝ) : 4*a - a^3 = a*(2-a)*(2+a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4a_minus_a_cubed_l1933_193340


namespace NUMINAMATH_CALUDE_honey_nights_l1933_193393

/-- Represents the number of servings of honey per cup of tea -/
def servings_per_cup : ℕ := 1

/-- Represents the number of cups of tea Tabitha drinks before bed each night -/
def cups_per_night : ℕ := 2

/-- Represents the size of the honey container in ounces -/
def container_size : ℕ := 16

/-- Represents the number of servings of honey per ounce -/
def servings_per_ounce : ℕ := 6

/-- Theorem stating how many nights Tabitha can enjoy honey in her tea -/
theorem honey_nights : 
  (container_size * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 := by
  sorry

end NUMINAMATH_CALUDE_honey_nights_l1933_193393


namespace NUMINAMATH_CALUDE_additional_land_cost_l1933_193359

/-- Calculates the cost of additional land purchased by Carlson -/
theorem additional_land_cost (initial_area : ℝ) (final_area : ℝ) (cost_per_sqm : ℝ) :
  initial_area = 300 →
  final_area = 900 →
  cost_per_sqm = 20 →
  (final_area - initial_area) * cost_per_sqm = 12000 := by
  sorry

#check additional_land_cost

end NUMINAMATH_CALUDE_additional_land_cost_l1933_193359


namespace NUMINAMATH_CALUDE_second_to_last_digit_even_l1933_193314

theorem second_to_last_digit_even (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, (3^n / 10) % 10 = 2 * k :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_even_l1933_193314


namespace NUMINAMATH_CALUDE_root_conditions_imply_m_range_l1933_193336

/-- A quadratic function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + (2 * m + 1)

/-- The theorem stating the range of m given the root conditions -/
theorem root_conditions_imply_m_range :
  ∀ m : ℝ,
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f m r₁ = 0 ∧ f m r₂ = 0) →
  (∃ r₁ : ℝ, -1 < r₁ ∧ r₁ < 0 ∧ f m r₁ = 0) →
  (∃ r₂ : ℝ, 1 < r₂ ∧ r₂ < 2 ∧ f m r₂ = 0) →
  1/4 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_m_range_l1933_193336


namespace NUMINAMATH_CALUDE_f_passes_through_quadrants_234_l1933_193338

/-- A linear function f(x) = kx + b passes through the second, third, and fourth quadrants if and only if k < 0 and b < 0 -/
def passes_through_quadrants_234 (k b : ℝ) : Prop :=
  k < 0 ∧ b < 0

/-- The specific linear function f(x) = -2x - 1 -/
def f (x : ℝ) : ℝ := -2 * x - 1

/-- Theorem stating that f(x) = -2x - 1 passes through the second, third, and fourth quadrants -/
theorem f_passes_through_quadrants_234 :
  passes_through_quadrants_234 (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_f_passes_through_quadrants_234_l1933_193338


namespace NUMINAMATH_CALUDE_total_yards_run_l1933_193392

/-- Calculates the total yards run by three athletes given their individual performances -/
theorem total_yards_run (athlete1_yards athlete2_yards athlete3_avg_yards : ℕ) 
  (games : ℕ) (h1 : games = 4) (h2 : athlete1_yards = 18) (h3 : athlete2_yards = 22) 
  (h4 : athlete3_avg_yards = 11) : 
  athlete1_yards * games + athlete2_yards * games + athlete3_avg_yards * games = 204 :=
by sorry

end NUMINAMATH_CALUDE_total_yards_run_l1933_193392


namespace NUMINAMATH_CALUDE_chord_slope_of_ellipse_l1933_193391

/-- Given an ellipse and a chord bisected by a point, prove the slope of the chord -/
theorem chord_slope_of_ellipse (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 4) →         -- Midpoint x-coordinate is 4
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  (y₁ - y₂) / (x₁ - x₂) = -1/2  -- Slope of the chord is -1/2
:= by sorry

end NUMINAMATH_CALUDE_chord_slope_of_ellipse_l1933_193391


namespace NUMINAMATH_CALUDE_tangent_angle_cosine_at_e_l1933_193353

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_angle_cosine_at_e :
  let θ := Real.arctan (deriv f e)
  Real.cos θ = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_tangent_angle_cosine_at_e_l1933_193353


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_implies_fourth_plus_reciprocal_fourth_l1933_193360

theorem square_plus_reciprocal_square_implies_fourth_plus_reciprocal_fourth
  (x : ℝ) (h : x^2 + 1/x^2 = 2) : x^4 + 1/x^4 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_implies_fourth_plus_reciprocal_fourth_l1933_193360


namespace NUMINAMATH_CALUDE_selection_probability_l1933_193348

/-- Represents the probability of a student being chosen in the selection process -/
def probability_of_selection (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) : ℚ :=
  (total_students - eliminated : ℚ) / total_students * selected / (total_students - eliminated)

/-- Theorem stating that the probability of each student being chosen is 4/43 -/
theorem selection_probability :
  let total_students : ℕ := 86
  let eliminated : ℕ := 6
  let selected : ℕ := 8
  probability_of_selection total_students eliminated selected = 4 / 43 := by
sorry

end NUMINAMATH_CALUDE_selection_probability_l1933_193348


namespace NUMINAMATH_CALUDE_pink_cookies_l1933_193385

theorem pink_cookies (total : ℕ) (red : ℕ) (h1 : total = 86) (h2 : red = 36) :
  total - red = 50 := by
  sorry

end NUMINAMATH_CALUDE_pink_cookies_l1933_193385


namespace NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l1933_193352

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem eighth_fibonacci_is_21 : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l1933_193352


namespace NUMINAMATH_CALUDE_determinant_value_trig_expression_value_l1933_193398

-- Define the determinant function for 2x2 matrices
def det2 (a11 a12 a21 a22 : ℝ) : ℝ := a11 * a22 - a12 * a21

-- Problem 1
theorem determinant_value : 
  det2 (Real.cos (π/4)) 1 1 (Real.cos (π/3)) = (Real.sqrt 2 - 2) / 4 := by
  sorry

-- Problem 2
theorem trig_expression_value (a : ℝ) (h : Real.tan (π/4 + a) = -1/2) :
  (Real.sin (2*a) - 2 * (Real.cos a)^2) / (1 + Real.tan a) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_determinant_value_trig_expression_value_l1933_193398


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1933_193328

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_product (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a (n + 1) > a n) →
  a 6 * a 7 = 15 →
  a 1 = 2 →
  a 4 * a 9 = 234 / 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1933_193328


namespace NUMINAMATH_CALUDE_uncle_dave_nieces_l1933_193379

theorem uncle_dave_nieces (total_sandwiches : ℕ) (sandwiches_per_niece : ℕ) (h1 : total_sandwiches = 143) (h2 : sandwiches_per_niece = 13) :
  total_sandwiches / sandwiches_per_niece = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_nieces_l1933_193379


namespace NUMINAMATH_CALUDE_vector_decomposition_l1933_193304

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![11, -1, 4]
def p : Fin 3 → ℝ := ![1, -1, 2]
def q : Fin 3 → ℝ := ![3, 2, 0]
def r : Fin 3 → ℝ := ![-1, 1, 1]

/-- Theorem stating the decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = λ i => 3 * p i + 2 * q i - 2 * r i := by
  sorry


end NUMINAMATH_CALUDE_vector_decomposition_l1933_193304


namespace NUMINAMATH_CALUDE_total_fruits_eaten_l1933_193351

/-- Prove that the total number of fruits eaten by three dogs is 240 given the specified conditions -/
theorem total_fruits_eaten (dog1_apples dog2_blueberries dog3_bonnies : ℕ) : 
  dog3_bonnies = 60 →
  dog2_blueberries = (3 * dog3_bonnies) / 4 →
  dog1_apples = 3 * dog2_blueberries →
  dog1_apples + dog2_blueberries + dog3_bonnies = 240 := by
  sorry

#check total_fruits_eaten

end NUMINAMATH_CALUDE_total_fruits_eaten_l1933_193351


namespace NUMINAMATH_CALUDE_blue_balls_count_l1933_193305

theorem blue_balls_count (total : ℕ) (p : ℚ) (h_total : total = 12) (h_prob : p = 1 / 22) :
  ∃ b : ℕ, b ≤ total ∧ 
    (b : ℚ) * (b - 1) / (total * (total - 1)) = p ∧
    b = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1933_193305


namespace NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l1933_193355

theorem shortest_side_of_special_triangle :
  ∀ (a b c : ℕ),
    a = 15 →
    a + b + c = 40 →
    (∃ A : ℕ, A^2 = (20 * (20 - a) * (20 - b) * (20 - c))) →
    a + b > c ∧ a + c > b ∧ b + c > a →
    b ≥ 8 ∧ c ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l1933_193355


namespace NUMINAMATH_CALUDE_proposition_a_is_true_l1933_193325

theorem proposition_a_is_true : ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0 := by
  sorry

#check proposition_a_is_true

end NUMINAMATH_CALUDE_proposition_a_is_true_l1933_193325


namespace NUMINAMATH_CALUDE_cube_expansion_seven_plus_one_l1933_193337

theorem cube_expansion_seven_plus_one : 7^3 + 3*(7^2) + 3*7 + 1 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_seven_plus_one_l1933_193337


namespace NUMINAMATH_CALUDE_burrito_combinations_l1933_193306

def number_of_ways_to_make_burritos : ℕ :=
  let max_beef := 4
  let max_chicken := 3
  let total_wraps := 5
  (Nat.choose total_wraps 3) + (Nat.choose total_wraps 2) + (Nat.choose total_wraps 1)

theorem burrito_combinations : number_of_ways_to_make_burritos = 25 := by
  sorry

end NUMINAMATH_CALUDE_burrito_combinations_l1933_193306


namespace NUMINAMATH_CALUDE_third_element_in_tenth_bracket_l1933_193342

/-- The number of elements in the nth bracket -/
def bracket_size (n : ℕ) : ℕ := n

/-- The sum of elements in the first n brackets -/
def sum_bracket_sizes (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last element in the nth bracket -/
def last_element_in_bracket (n : ℕ) : ℕ := sum_bracket_sizes n

theorem third_element_in_tenth_bracket :
  ∃ (k : ℕ), k = last_element_in_bracket 9 + 3 ∧ k = 48 :=
sorry

end NUMINAMATH_CALUDE_third_element_in_tenth_bracket_l1933_193342


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1933_193322

theorem arithmetic_simplification : 180 * (180 - 12) - (180 * 180 - 12) = -2148 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1933_193322


namespace NUMINAMATH_CALUDE_band_encore_problem_l1933_193378

def band_encore_songs (total_songs : ℕ) (first_set : ℕ) (second_set : ℕ) (avg_third_fourth : ℕ) : ℕ :=
  total_songs - (first_set + second_set + 2 * avg_third_fourth)

theorem band_encore_problem :
  band_encore_songs 30 5 7 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_band_encore_problem_l1933_193378


namespace NUMINAMATH_CALUDE_cow_count_l1933_193313

/-- Represents the number of dairy cows owned by a breeder. -/
def number_of_cows : ℕ := sorry

/-- The amount of milk (in oz) produced by one cow per day. -/
def milk_per_cow_per_day : ℕ := 1000

/-- The total amount of milk (in oz) produced in a week. -/
def total_milk_per_week : ℕ := 364000

/-- The number of days in a week. -/
def days_in_week : ℕ := 7

/-- Theorem stating that the number of cows is 52, given the milk production conditions. -/
theorem cow_count : number_of_cows = 52 := by sorry

end NUMINAMATH_CALUDE_cow_count_l1933_193313


namespace NUMINAMATH_CALUDE_foreign_trade_analysis_l1933_193301

-- Define the data points
def x : List ℝ := [1.8, 2.2, 2.6, 3.0]
def y : List ℝ := [2.0, 2.8, 3.2, 4.0]

-- Define the linear correlation function
def linear_correlation (b : ℝ) (x : ℝ) : ℝ := b * x - 0.84

-- Theorem statement
theorem foreign_trade_analysis :
  let x_mean := (List.sum x) / (List.length x : ℝ)
  let y_mean := (List.sum y) / (List.length y : ℝ)
  let b_hat := (y_mean + 0.84) / x_mean
  ∀ (ε : ℝ), ε > 0 →
    (abs (b_hat - 1.6) < ε) ∧
    (abs ((linear_correlation b_hat⁻¹ 6 + 0.84) / b_hat - 4.275) < ε) :=
by sorry

end NUMINAMATH_CALUDE_foreign_trade_analysis_l1933_193301


namespace NUMINAMATH_CALUDE_lee_annual_salary_l1933_193384

/-- Lee's annual salary calculation --/
theorem lee_annual_salary (monthly_savings : ℕ) (saving_months : ℕ) : 
  monthly_savings = 1000 →
  saving_months = 10 →
  (monthly_savings * saving_months : ℕ) = (2 * (60000 / 12) : ℕ) →
  60000 = (monthly_savings * saving_months * 6 : ℕ) := by
  sorry

#check lee_annual_salary

end NUMINAMATH_CALUDE_lee_annual_salary_l1933_193384
