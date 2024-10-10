import Mathlib

namespace sum_of_roots_l1208_120820

theorem sum_of_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3*x^3 - 7*x^2 + 2*x
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 7/3) :=
by sorry

end sum_of_roots_l1208_120820


namespace somu_age_problem_l1208_120867

theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 5 = (father_age - 5) / 5 →
  somu_age = 10 := by
sorry

end somu_age_problem_l1208_120867


namespace quadratic_function_sum_l1208_120891

theorem quadratic_function_sum (a b : ℝ) : 
  a > 0 → 
  (∀ x ∈ Set.Icc 2 3, (a * x^2 - 2 * a * x + 1 + b) ≤ 4) →
  (∀ x ∈ Set.Icc 2 3, (a * x^2 - 2 * a * x + 1 + b) ≥ 1) →
  (∃ x ∈ Set.Icc 2 3, a * x^2 - 2 * a * x + 1 + b = 4) →
  (∃ x ∈ Set.Icc 2 3, a * x^2 - 2 * a * x + 1 + b = 1) →
  a + b = 6 :=
by sorry

end quadratic_function_sum_l1208_120891


namespace words_with_A_count_l1208_120837

/-- The number of letters in our alphabet -/
def n : ℕ := 4

/-- The length of words we're considering -/
def k : ℕ := 3

/-- The number of letters in our alphabet excluding 'A' -/
def m : ℕ := 3

/-- The number of 3-letter words that can be made from the letters A, B, C, and D, 
    with at least one A being used and allowing repetition of letters -/
def words_with_A : ℕ := n^k - m^k

theorem words_with_A_count : words_with_A = 37 := by sorry

end words_with_A_count_l1208_120837


namespace friday_temperature_l1208_120860

theorem friday_temperature
  (temp_mon : ℝ)
  (temp_tue : ℝ)
  (temp_wed : ℝ)
  (temp_thu : ℝ)
  (temp_fri : ℝ)
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 46)
  (h3 : temp_mon = 43)
  : temp_fri = 35 := by
  sorry

end friday_temperature_l1208_120860


namespace intersection_of_A_and_B_l1208_120877

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 2*x + 5)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 2 := by sorry

end intersection_of_A_and_B_l1208_120877


namespace number_difference_l1208_120836

/-- 
Theorem: Given a three-digit number x and an even two-digit number y, 
if their difference is 3, then x = 101 and y = 98.
-/
theorem number_difference (x y : ℕ) : 
  (100 ≤ x ∧ x ≤ 999) →  -- x is a three-digit number
  (10 ≤ y ∧ y ≤ 98) →    -- y is a two-digit number
  Even y →               -- y is even
  x - y = 3 →            -- difference is 3
  x = 101 ∧ y = 98 :=
by sorry

end number_difference_l1208_120836


namespace probability_sum_seven_is_one_sixth_l1208_120866

/-- The number of possible outcomes for each die -/
def dice_outcomes : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := dice_outcomes * dice_outcomes

/-- The number of ways to get a sum of 7 with two dice -/
def favorable_outcomes : ℕ := 6

/-- The probability of getting a sum of 7 when throwing two fair dice -/
def probability_sum_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_seven_is_one_sixth :
  probability_sum_seven = 1 / 6 := by sorry

end probability_sum_seven_is_one_sixth_l1208_120866


namespace sin_pi_6_minus_2alpha_l1208_120814

theorem sin_pi_6_minus_2alpha (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.cos (α - π/6), 1/2))
  (hb : b = (1, -2 * Real.sin α))
  (hab : a.1 * b.1 + a.2 * b.2 = 1/3) :
  Real.sin (π/6 - 2*α) = -7/9 := by
sorry

end sin_pi_6_minus_2alpha_l1208_120814


namespace train_length_l1208_120816

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * (5/18) → 
  platform_length = 250 → 
  crossing_time = 24 → 
  speed * crossing_time - platform_length = 230 :=
by sorry

end train_length_l1208_120816


namespace ball_picking_probabilities_l1208_120804

/-- The probability of picking ball 3 using method one -/
def P₁ : ℚ := 1/3

/-- The probability of picking ball 3 using method two -/
def P₂ : ℚ := 1/2

/-- The probability of picking ball 3 using method three -/
def P₃ : ℚ := 2/3

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem ball_picking_probabilities :
  (P₁ < P₂) ∧ (P₁ < P₃) ∧ (P₂ ≠ P₃) ∧ (2 * P₁ = P₃) := by
  sorry

end ball_picking_probabilities_l1208_120804


namespace intersection_vector_sum_l1208_120865

noncomputable def f (x : ℝ) : ℝ := (2 * Real.cos x ^ 2 + 1) / Real.log ((2 + x) / (2 - x))

theorem intersection_vector_sum (a : ℝ) (h_a : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), 
    (∀ x : ℝ, a * x - (f x) = 0 → x = A.1 ∨ x = B.1) →
    (A ≠ B) →
    (∀ m n : ℝ, 
      (A.1 - m, A.2 - n) + (B.1 - m, B.2 - n) = (m - 6, n) →
      m + n = 2) :=
by sorry

end intersection_vector_sum_l1208_120865


namespace square_of_102_l1208_120810

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end square_of_102_l1208_120810


namespace sample_size_is_80_l1208_120857

/-- Represents the ratio of product models A, B, and C -/
def productRatio : Fin 3 → ℕ
  | 0 => 2  -- Model A
  | 1 => 3  -- Model B
  | 2 => 5  -- Model C
  | _ => 0  -- This case should never occur due to Fin 3

/-- Calculates the total ratio sum -/
def totalRatio : ℕ := (productRatio 0) + (productRatio 1) + (productRatio 2)

/-- Represents the number of units of model A in the sample -/
def modelAUnits : ℕ := 16

/-- Theorem stating that the sample size is 80 given the conditions -/
theorem sample_size_is_80 :
  ∃ (n : ℕ), n * (productRatio 0) / totalRatio = modelAUnits ∧ n = 80 :=
sorry

end sample_size_is_80_l1208_120857


namespace vector_parallel_proof_l1208_120818

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, -1]
def c (m n : ℝ) : Fin 2 → ℝ := ![m - 2, -n]

theorem vector_parallel_proof (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_parallel : ∃ (k : ℝ), ∀ i, (a - b) i = k * c m n i) :
  (2 * m + n = 4) ∧ (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 4 → x * y ≤ 2) := by
  sorry

end vector_parallel_proof_l1208_120818


namespace smallest_multiple_37_congruent_7_mod_76_l1208_120874

theorem smallest_multiple_37_congruent_7_mod_76 : ∃ (n : ℕ), n > 0 ∧ 37 ∣ n ∧ n ≡ 7 [MOD 76] ∧ ∀ (m : ℕ), m > 0 ∧ 37 ∣ m ∧ m ≡ 7 [MOD 76] → n ≤ m :=
by sorry

end smallest_multiple_37_congruent_7_mod_76_l1208_120874


namespace f_value_at_3_l1208_120888

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = 4) : f a b 3 = -12 := by
  sorry

end f_value_at_3_l1208_120888


namespace seconds_in_week_l1208_120840

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: The product of seconds per minute, minutes per hour, hours per day, and days per week
    equals the number of seconds in a week -/
theorem seconds_in_week :
  seconds_per_minute * minutes_per_hour * hours_per_day * days_per_week =
  (seconds_per_minute * minutes_per_hour * hours_per_day) * days_per_week :=
by sorry

end seconds_in_week_l1208_120840


namespace fifth_bush_berries_l1208_120835

def berry_sequence : ℕ → ℕ
  | 0 => 3
  | 1 => 4
  | 2 => 7
  | 3 => 12
  | n + 4 => berry_sequence (n + 3) + (berry_sequence (n + 3) - berry_sequence (n + 2) + 2)

theorem fifth_bush_berries : berry_sequence 4 = 19 := by
  sorry

end fifth_bush_berries_l1208_120835


namespace range_of_a_l1208_120801

-- Define the function g
def g (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g x < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by sorry

end range_of_a_l1208_120801


namespace binary_1101001101_equals_base4_311310_l1208_120886

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_1101001101_equals_base4_311310 :
  let binary : List Bool := [true, true, false, true, false, false, true, true, false, true]
  decimal_to_base4 (binary_to_decimal binary) = [3, 1, 1, 3, 1, 0] := by
  sorry

end binary_1101001101_equals_base4_311310_l1208_120886


namespace derivative_cos_times_exp_sin_l1208_120873

/-- The derivative of f(x) = cos(x) * e^(sin(x)) -/
theorem derivative_cos_times_exp_sin (x : ℝ) :
  deriv (fun x => Real.cos x * Real.exp (Real.sin x)) x =
  (Real.cos x ^ 2 - Real.sin x) * Real.exp (Real.sin x) := by
sorry

end derivative_cos_times_exp_sin_l1208_120873


namespace hiker_speed_difference_l1208_120870

/-- A hiker's journey over three days -/
def hiker_journey (v : ℝ) : Prop :=
  let day1_distance : ℝ := 18
  let day1_speed : ℝ := 3
  let day1_hours : ℝ := day1_distance / day1_speed
  let day2_hours : ℝ := day1_hours - 1
  let day2_distance : ℝ := day2_hours * v
  let day3_distance : ℝ := 5 * 3
  day1_distance + day2_distance + day3_distance = 53

theorem hiker_speed_difference : ∃ v : ℝ, hiker_journey v ∧ v - 3 = 1 := by
  sorry

#check hiker_speed_difference

end hiker_speed_difference_l1208_120870


namespace specific_frustum_small_cone_altitude_l1208_120880

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude

/-- Theorem stating that for a specific frustum, the altitude of the small cone is 18 cm -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := {
    altitude := 18,
    lower_base_area := 144 * Real.pi,
    upper_base_area := 36 * Real.pi
  }
  small_cone_altitude f = 18 := by sorry

end specific_frustum_small_cone_altitude_l1208_120880


namespace value_of_two_over_x_l1208_120897

theorem value_of_two_over_x (x : ℂ) (h : 1 - 5 / x + 9 / x^2 = 0) :
  2 / x = Complex.ofReal (5 / 9) - Complex.I * Complex.ofReal (Real.sqrt 11 / 9) ∨
  2 / x = Complex.ofReal (5 / 9) + Complex.I * Complex.ofReal (Real.sqrt 11 / 9) := by
sorry

end value_of_two_over_x_l1208_120897


namespace weight_of_new_person_l1208_120898

theorem weight_of_new_person 
  (n : ℕ) 
  (initial_average : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) : 
  n = 8 → 
  weight_increase = 5 → 
  replaced_weight = 35 → 
  (n * initial_average + (n * weight_increase + replaced_weight)) / n = 
    initial_average + weight_increase → 
  n * weight_increase + replaced_weight = 75 := by
sorry

end weight_of_new_person_l1208_120898


namespace jessica_cut_thirteen_roses_l1208_120883

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses vase_roses garden_roses : ℕ) : ℕ :=
  vase_roses - initial_roses

/-- Theorem stating that Jessica cut 13 roses -/
theorem jessica_cut_thirteen_roses :
  ∃ (initial_roses vase_roses garden_roses : ℕ),
    initial_roses = 7 ∧
    garden_roses = 59 ∧
    vase_roses = 20 ∧
    roses_cut initial_roses vase_roses garden_roses = 13 :=
by
  sorry

end jessica_cut_thirteen_roses_l1208_120883


namespace no_valid_stacking_l1208_120849

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of crates -/
def numCrates : ℕ := 12

/-- The dimensions of each crate -/
def crateDims : CrateDimensions := ⟨3, 4, 5⟩

/-- The target total height -/
def targetHeight : ℕ := 50

/-- Theorem stating that there are no valid ways to stack the crates to reach the target height -/
theorem no_valid_stacking :
  ¬∃ (a b c : ℕ), a + b + c = numCrates ∧
                  a * crateDims.length + b * crateDims.width + c * crateDims.height = targetHeight :=
by sorry

end no_valid_stacking_l1208_120849


namespace impossible_three_shell_piles_l1208_120884

/-- Represents the number of seashells at step n -/
def S (n : ℕ) : ℤ := 637 - n

/-- Represents the number of piles at step n -/
def P (n : ℕ) : ℤ := 1 + n

/-- Theorem stating that it's impossible to end up with only piles of exactly three seashells -/
theorem impossible_three_shell_piles : ¬ ∃ n : ℕ, S n = 3 * P n ∧ S n > 0 := by
  sorry

end impossible_three_shell_piles_l1208_120884


namespace students_doing_homework_l1208_120838

theorem students_doing_homework (total : ℕ) (silent_reading : ℚ) (board_games : ℚ) 
  (h1 : total = 60)
  (h2 : silent_reading = 3/8)
  (h3 : board_games = 1/4) :
  total - (Int.floor (silent_reading * total) + Int.floor (board_games * total)) = 22 :=
by sorry

end students_doing_homework_l1208_120838


namespace largest_non_representable_integer_l1208_120858

theorem largest_non_representable_integer 
  (a b c : ℕ+) 
  (h1 : Nat.gcd a b = 1) 
  (h2 : Nat.gcd b c = 1) 
  (h3 : Nat.gcd c a = 1) :
  ∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
  ∃ (x y z : ℕ), n = b*c*x + c*a*y + a*b*z ∧
  ¬∃ (x y z : ℕ), 2*a*b*c - a*b - b*c - c*a = b*c*x + c*a*y + a*b*z :=
sorry

end largest_non_representable_integer_l1208_120858


namespace compacted_cans_space_l1208_120882

/-- The space occupied by compacted cans -/
def space_occupied (num_cans : ℕ) (initial_space : ℝ) (compaction_ratio : ℝ) : ℝ :=
  (num_cans : ℝ) * initial_space * compaction_ratio

/-- Theorem: 60 cans, each initially 30 sq inches, compacted to 20%, occupy 360 sq inches -/
theorem compacted_cans_space :
  space_occupied 60 30 0.2 = 360 := by
  sorry

end compacted_cans_space_l1208_120882


namespace line_intersection_theorem_l1208_120868

/-- A line y = mx + b intersecting a circle and a hyperbola -/
structure LineIntersection where
  m : ℝ
  b : ℝ
  h_m : |m| < 1
  h_b : |b| < 1

/-- Points of intersection -/
structure IntersectionPoints (l : LineIntersection) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_circle_P : P.1^2 + P.2^2 = 1 ∧ P.2 = l.m * P.1 + l.b
  h_circle_Q : Q.1^2 + Q.2^2 = 1 ∧ Q.2 = l.m * Q.1 + l.b
  h_hyperbola_R : R.1^2 - R.2^2 = 1 ∧ R.2 = l.m * R.1 + l.b
  h_hyperbola_S : S.1^2 - S.2^2 = 1 ∧ S.2 = l.m * S.1 + l.b
  h_trisect : dist P R = dist P Q ∧ dist Q S = dist P Q

/-- The main theorem -/
theorem line_intersection_theorem (l : LineIntersection) (p : IntersectionPoints l) :
  (l.m = 0 ∧ l.b^2 = 4/5) ∨ (l.b = 0 ∧ l.m^2 = 4/5) := by
  sorry

end line_intersection_theorem_l1208_120868


namespace percentage_difference_l1208_120817

theorem percentage_difference : 
  (67.5 / 100 * 250) - (52.3 / 100 * 180) = 74.61 := by
  sorry

end percentage_difference_l1208_120817


namespace coprime_pairs_count_l1208_120822

def count_coprime_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    1 ≤ a ∧ a ≤ b ∧ b ≤ 5 ∧ Nat.gcd a b = 1) 
    (Finset.product (Finset.range 6) (Finset.range 6))).card

theorem coprime_pairs_count : count_coprime_pairs = 10 := by
  sorry

end coprime_pairs_count_l1208_120822


namespace divisible_by_nine_l1208_120854

theorem divisible_by_nine (a b : ℤ) : ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by
  sorry

end divisible_by_nine_l1208_120854


namespace x_eleven_percent_greater_than_90_l1208_120827

theorem x_eleven_percent_greater_than_90 :
  ∀ x : ℝ, x = 90 * (1 + 11 / 100) → x = 99.9 := by
  sorry

end x_eleven_percent_greater_than_90_l1208_120827


namespace cone_height_l1208_120878

theorem cone_height (r : Real) (h : Real) :
  (3 : Real) * (2 * Real.pi / 3) = 2 * Real.pi * r →
  h ^ 2 + r ^ 2 = 3 ^ 2 →
  h = 2 * Real.sqrt 2 := by
  sorry

end cone_height_l1208_120878


namespace units_digit_of_six_to_fourth_l1208_120859

theorem units_digit_of_six_to_fourth (n : ℕ) : n = 6^4 → n % 10 = 6 := by
  sorry

end units_digit_of_six_to_fourth_l1208_120859


namespace angle_c_not_five_sixths_pi_l1208_120805

theorem angle_c_not_five_sixths_pi (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_eq1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h_eq2 : 3 * Real.cos A + 4 * Real.sin B = 1) : 
  C ≠ 5 * π / 6 := by
  sorry

end angle_c_not_five_sixths_pi_l1208_120805


namespace combined_work_time_l1208_120821

def worker_a_time : ℝ := 10
def worker_b_time : ℝ := 15

theorem combined_work_time : 
  let combined_rate := (1 / worker_a_time) + (1 / worker_b_time)
  1 / combined_rate = 6 := by sorry

end combined_work_time_l1208_120821


namespace y_intercept_of_line_l1208_120844

/-- The y-intercept of the line 3x - 5y = 10 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 3*x - 5*y = 10 → x = 0 → y = -2 := by
  sorry

end y_intercept_of_line_l1208_120844


namespace units_digit_of_27_cubed_minus_17_cubed_units_digit_is_zero_l1208_120851

theorem units_digit_of_27_cubed_minus_17_cubed : ℕ → Prop :=
  fun d => (27^3 - 17^3) % 10 = d

theorem units_digit_is_zero :
  units_digit_of_27_cubed_minus_17_cubed 0 := by
  sorry

end units_digit_of_27_cubed_minus_17_cubed_units_digit_is_zero_l1208_120851


namespace det_trig_matrix_zero_l1208_120869

/-- The determinant of the matrix
    [1, cos(a-b), sin(a);
     cos(a-b), 1, sin(b);
     sin(a), sin(b), 1]
    is equal to 0 for any real numbers a and b. -/
theorem det_trig_matrix_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1, Real.cos (a - b), Real.sin a;
                                        Real.cos (a - b), 1, Real.sin b;
                                        Real.sin a, Real.sin b, 1]
  Matrix.det M = 0 := by sorry

end det_trig_matrix_zero_l1208_120869


namespace point_P_coordinates_l1208_120834

def C (x : ℝ) : ℝ := x^3 - 10*x + 3

theorem point_P_coordinates :
  ∃! (x y : ℝ), 
    y = C x ∧ 
    x < 0 ∧ 
    y > 0 ∧ 
    (3 * x^2 - 10 = 2) ∧ 
    x = -2 ∧ 
    y = 15 := by
  sorry

end point_P_coordinates_l1208_120834


namespace quadratic_roots_range_l1208_120823

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_roots_range (a b c : ℝ) (ha : a ≠ 0) :
  f a b c (-1) = -2 →
  f a b c (-1/2) = -1/4 →
  f a b c 0 = 1 →
  f a b c (1/2) = 7/4 →
  f a b c 1 = 2 →
  f a b c (3/2) = 7/4 →
  f a b c 2 = 1 →
  f a b c (5/2) = -1/4 →
  f a b c 3 = -2 →
  ∃ x₁ x₂ : ℝ, f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ 
    -1/2 < x₁ ∧ x₁ < 0 ∧ 2 < x₂ ∧ x₂ < 5/2 :=
by sorry

end quadratic_roots_range_l1208_120823


namespace inscribed_squares_ratio_l1208_120843

/-- Given two isosceles right triangles with leg lengths 1, let x be the side length of a square
    inscribed in the first triangle with one vertex at the right angle, and y be the side length
    of a square inscribed in the second triangle with one side on the hypotenuse. -/
theorem inscribed_squares_ratio (x y : ℝ) 
  (hx : x = (1 : ℝ) / 2)  -- x is the side length of the square in the first triangle
  (hy : y = Real.sqrt 2 / 2) -- y is the side length of the square in the second triangle
  : x / y = Real.sqrt 2 := by
  sorry

#check inscribed_squares_ratio

end inscribed_squares_ratio_l1208_120843


namespace no_solution_lcm_equation_l1208_120812

theorem no_solution_lcm_equation :
  ¬∃ (n m : ℕ), Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019 := by
  sorry

end no_solution_lcm_equation_l1208_120812


namespace unique_solution_is_two_l1208_120824

theorem unique_solution_is_two : 
  ∃! n : ℕ+, 
    (n : ℕ) ∣ (Nat.totient n)^(Nat.divisors n).card + 1 ∧ 
    ¬((Nat.divisors n).card^5 ∣ (n : ℕ)^(Nat.totient n) - 1) ∧
    n = 2 := by
  sorry

end unique_solution_is_two_l1208_120824


namespace intersection_of_A_and_B_l1208_120825

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {-1, 0, 1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := by sorry

end intersection_of_A_and_B_l1208_120825


namespace different_color_pairs_count_l1208_120894

def white_socks : ℕ := 4
def brown_socks : ℕ := 4
def blue_socks : ℕ := 2
def gray_socks : ℕ := 5

def total_socks : ℕ := white_socks + brown_socks + blue_socks + gray_socks

def different_color_pairs : ℕ := 
  white_socks * brown_socks + 
  white_socks * blue_socks + 
  white_socks * gray_socks + 
  brown_socks * blue_socks + 
  brown_socks * gray_socks + 
  blue_socks * gray_socks

theorem different_color_pairs_count : different_color_pairs = 82 := by
  sorry

end different_color_pairs_count_l1208_120894


namespace factoring_quadratic_l1208_120807

theorem factoring_quadratic (x : ℝ) : 60 * x + 90 - 15 * x^2 = 15 * (-x^2 + 4 * x + 6) := by
  sorry

end factoring_quadratic_l1208_120807


namespace system_two_solutions_l1208_120876

/-- The system of equations has exactly two solutions iff a = 4 or a = 100 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (x y : ℝ), |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ∧
  (∃! (x' y' : ℝ), (x', y') ≠ (x, y) ∧ |x' - 6 - y'| + |x' - 6 + y'| = 12 ∧ (|x'| - 6)^2 + (|y'| - 8)^2 = a) ↔
  (a = 4 ∨ a = 100) :=
sorry

end system_two_solutions_l1208_120876


namespace max_value_of_expression_l1208_120806

theorem max_value_of_expression (p q r s : ℕ) : 
  p ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  q ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  r ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  s ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
  p^q + r^s ≤ 83 :=
by sorry

end max_value_of_expression_l1208_120806


namespace divisors_of_420_l1208_120899

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of divisors function -/
def num_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the sum and number of divisors for 420 -/
theorem divisors_of_420 : 
  sum_of_divisors 420 = 1344 ∧ num_of_divisors 420 = 24 := by sorry

end divisors_of_420_l1208_120899


namespace scalene_triangles_count_l1208_120808

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c < 20 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem scalene_triangles_count :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S → is_valid_scalene_triangle t.1 t.2.1 t.2.2) ∧
    S.card > 7 :=
sorry

end scalene_triangles_count_l1208_120808


namespace solution_to_equation_one_no_solution_to_equation_two_l1208_120800

-- Problem 1
theorem solution_to_equation_one (x : ℝ) : 
  (3 / x) - (2 / (x - 2)) = 0 ↔ x = 6 :=
sorry

-- Problem 2
theorem no_solution_to_equation_two :
  ¬∃ (x : ℝ), (3 / (4 - x)) + 2 = ((1 - x) / (x - 4)) :=
sorry

end solution_to_equation_one_no_solution_to_equation_two_l1208_120800


namespace solution_set_implies_m_value_l1208_120896

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end solution_set_implies_m_value_l1208_120896


namespace tan_thirty_degrees_l1208_120861

theorem tan_thirty_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_thirty_degrees_l1208_120861


namespace one_true_proposition_l1208_120815

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1

-- Define the inverse of the proposition
def inverse_proposition (a b : ℝ) : Prop :=
  a + b ≠ 1 → a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0

-- Define the negation of the proposition
def negation_proposition (a b : ℝ) : Prop :=
  a^2 + 2*a*b + b^2 + a + b - 2 = 0 → a + b = 1

-- Define the contrapositive of the proposition
def contrapositive_proposition (a b : ℝ) : Prop :=
  a + b = 1 → a^2 + 2*a*b + b^2 + a + b - 2 = 0

-- Theorem statement
theorem one_true_proposition :
  ∃! p : (ℝ → ℝ → Prop), 
    (p = inverse_proposition ∨ p = negation_proposition ∨ p = contrapositive_proposition) ∧
    (∀ a b : ℝ, p a b) :=
  sorry

end one_true_proposition_l1208_120815


namespace unique_solution_fraction_equation_l1208_120879

theorem unique_solution_fraction_equation :
  ∃! x : ℝ, (x ≠ 3 ∧ x ≠ 4) ∧ (3 / (x - 3) = 4 / (x - 4)) ∧ x = 0 := by
sorry

end unique_solution_fraction_equation_l1208_120879


namespace primitive_existence_l1208_120831

open Set

theorem primitive_existence (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : a < c) (h2 : c < b)
  (h3 : ContinuousAt f c)
  (h4 : ∃ F1 : ℝ → ℝ, ∀ x ∈ Icc a c, HasDerivAt F1 (f x) x)
  (h5 : ∃ F2 : ℝ → ℝ, ∀ x ∈ Ico c b, HasDerivAt F2 (f x) x) :
  ∃ F : ℝ → ℝ, ∀ x ∈ Icc a b, HasDerivAt F (f x) x :=
sorry

end primitive_existence_l1208_120831


namespace dogs_wearing_neither_l1208_120826

theorem dogs_wearing_neither (total : ℕ) (tags : ℕ) (collars : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : tags = 45)
  (h3 : collars = 40)
  (h4 : both = 6) :
  total - (tags + collars - both) = 1 := by
  sorry

end dogs_wearing_neither_l1208_120826


namespace post_office_distance_l1208_120890

/-- The distance from the village to the post office satisfies the given conditions -/
theorem post_office_distance (D : ℝ) : D > 0 →
  (D / 25 + D / 4 = 5.8) → D = 20 := by sorry

end post_office_distance_l1208_120890


namespace triangle_ratio_l1208_120832

/-- Given a triangle ABC with the following properties:
  - M is the midpoint of BC
  - AB = 15
  - AC = 20
  - E is on AC
  - F is on AB
  - G is the intersection of EF and AM
  - AE = 3AF
  Prove that EG/GF = 2/3 -/
theorem triangle_ratio (A B C M E F G : ℝ × ℝ) : 
  (M = (B + C) / 2) →
  (dist A B = 15) →
  (dist A C = 20) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C) →
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (1 - s) • A + s • B) →
  (∃ r : ℝ, G = (1 - r) • E + r • F) →
  (∃ q : ℝ, G = (1 - q) • A + q • M) →
  (dist A E = 3 * dist A F) →
  (dist E G) / (dist G F) = 2 / 3 := by
  sorry

end triangle_ratio_l1208_120832


namespace modular_multiplication_l1208_120842

theorem modular_multiplication (m : ℕ) : 
  0 ≤ m ∧ m < 25 ∧ m ≡ (66 * 77 * 88) [ZMOD 25] → m = 16 := by
  sorry

end modular_multiplication_l1208_120842


namespace smallest_d_value_l1208_120830

theorem smallest_d_value (d : ℝ) : 
  (4 * Real.sqrt 3)^2 + (d - 2)^2 = (4 * d)^2 → d ≥ 2.006 := by
  sorry

end smallest_d_value_l1208_120830


namespace students_above_120_l1208_120881

/-- Normal distribution parameters -/
structure NormalDist where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for normal distribution -/
noncomputable def prob (nd : NormalDist) (a b : ℝ) : ℝ := sorry

/-- Theorem: Number of students scoring above 120 -/
theorem students_above_120 (nd : NormalDist) (total_students : ℕ) :
  nd.μ = 90 →
  prob nd 60 120 = 0.8 →
  total_students = 780 →
  ⌊(1 - prob nd 60 120) / 2 * total_students⌋ = 78 := by sorry

end students_above_120_l1208_120881


namespace intersection_of_A_and_B_l1208_120850

-- Define the sets A and B
variable (A B : Set ℤ)

-- Define the function f
def f (x : ℤ) : ℤ := x^2

-- State the theorem
theorem intersection_of_A_and_B :
  (∀ x ∈ A, f x ∈ B) →
  B = {1, 2} →
  (A ∩ B = ∅ ∨ A ∩ B = {1}) :=
by sorry

end intersection_of_A_and_B_l1208_120850


namespace cube_root_problem_l1208_120803

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end cube_root_problem_l1208_120803


namespace minimum_n_for_inequality_l1208_120853

theorem minimum_n_for_inequality :
  (∃ (n : ℕ), ∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < 3 → ∃ (x y z : ℝ), (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) :=
by sorry

end minimum_n_for_inequality_l1208_120853


namespace hazel_walk_l1208_120863

/-- Hazel's walk problem -/
theorem hazel_walk (first_hour_distance : ℝ) (h1 : first_hour_distance = 2) :
  let second_hour_distance := 2 * first_hour_distance
  first_hour_distance + second_hour_distance = 6 := by
  sorry

end hazel_walk_l1208_120863


namespace left_handed_women_percentage_l1208_120852

theorem left_handed_women_percentage
  (total : ℕ)
  (right_handed : ℕ)
  (left_handed : ℕ)
  (men : ℕ)
  (women : ℕ)
  (h1 : right_handed = 3 * left_handed)
  (h2 : men = 3 * women / 2)
  (h3 : total = right_handed + left_handed)
  (h4 : total = men + women)
  (h5 : right_handed ≥ men)
  (h6 : right_handed = men) :
  women = left_handed ∧ left_handed * 100 / total = 25 :=
sorry

end left_handed_women_percentage_l1208_120852


namespace complex_absolute_value_l1208_120848

theorem complex_absolute_value (z : ℂ) : z = 2 / (1 - Complex.I * Real.sqrt 3) → Complex.abs z = 1 := by
  sorry

end complex_absolute_value_l1208_120848


namespace product_reciprocal_sum_l1208_120802

theorem product_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 16) (h_recip : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end product_reciprocal_sum_l1208_120802


namespace total_shingles_needed_l1208_120839

/-- The number of shingles needed to cover a given area of roof --/
def shingles_per_square_foot : ℕ := 8

/-- The number of roofs to be shingled --/
def number_of_roofs : ℕ := 3

/-- The length of each rectangular side of a roof in feet --/
def roof_side_length : ℕ := 40

/-- The width of each rectangular side of a roof in feet --/
def roof_side_width : ℕ := 20

/-- The number of rectangular sides per roof --/
def sides_per_roof : ℕ := 2

/-- Theorem stating the total number of shingles needed --/
theorem total_shingles_needed :
  (number_of_roofs * sides_per_roof * roof_side_length * roof_side_width * shingles_per_square_foot) = 38400 := by
  sorry

end total_shingles_needed_l1208_120839


namespace arithmetic_sequence_problem_l1208_120887

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₃ = 6 and a₆ = 3, prove a₉ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 6)
  (h_6 : a 6 = 3) :
  a 9 = 0 := by
  sorry

end arithmetic_sequence_problem_l1208_120887


namespace sequence_convergence_and_general_term_l1208_120847

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | n + 2 => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

/-- The general term formula for a_n when n ≥ 2 -/
noncomputable def a_general_term (x y : ℝ) (n : ℕ) : ℝ :=
  let num := 2 * ((y - 1) / (y + 1)) ^ (fib (n - 1)) * ((x - 1) / (x + 1)) ^ (fib (n - 2))
  let den := 1 - ((y - 1) / (y + 1)) ^ (fib (n - 1)) * ((x - 1) / (x + 1)) ^ (fib (n - 2))
  num / den - 1

theorem sequence_convergence_and_general_term (x y : ℝ) :
  (∃ n₀ : ℕ+, ∀ n ≥ n₀, a x y n = 1 ∨ a x y n = -1) ↔
    ((x = 1 ∧ y ≠ -1) ∨ (x = -1 ∧ y ≠ 1) ∨ (y = 1 ∧ x ≠ -1) ∨ (y = -1 ∧ x ≠ 1)) ∧
  ∀ n ≥ 2, a x y n = a_general_term x y n :=
by sorry

end sequence_convergence_and_general_term_l1208_120847


namespace smallest_side_difference_l1208_120828

theorem smallest_side_difference (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2021 →
  PQ < QR →
  QR ≤ PR →
  (∀ PQ' QR' PR' : ℕ, 
    PQ' + QR' + PR' = 2021 → 
    PQ' < QR' → 
    QR' ≤ PR' → 
    QR' - PQ' ≥ 1) →
  QR - PQ = 1 :=
by sorry

end smallest_side_difference_l1208_120828


namespace probability_two_blue_marbles_l1208_120829

def total_marbles : ℕ := 3 + 4 + 8 + 5

def blue_marbles : ℕ := 8

theorem probability_two_blue_marbles :
  (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) = 14 / 95 := by
  sorry

end probability_two_blue_marbles_l1208_120829


namespace louis_lemon_heads_l1208_120811

/-- The number of Lemon Heads in a package -/
def lemon_heads_per_package : ℕ := 6

/-- The number of whole boxes Louis ate -/
def boxes_eaten : ℕ := 9

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := lemon_heads_per_package * boxes_eaten

theorem louis_lemon_heads : total_lemon_heads = 54 := by
  sorry

end louis_lemon_heads_l1208_120811


namespace sum_of_squares_implies_sum_l1208_120864

theorem sum_of_squares_implies_sum : ∀ (a b c : ℝ), 
  (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0 → a + 2*b + 3*c = 18 := by
  sorry

end sum_of_squares_implies_sum_l1208_120864


namespace matchsticks_left_l1208_120885

/-- Calculates the number of matchsticks left in a box after Elvis and Ralph make their squares -/
theorem matchsticks_left (initial_count : ℕ) (elvis_square_size elvis_squares : ℕ) (ralph_square_size ralph_squares : ℕ) : 
  initial_count = 50 → 
  elvis_square_size = 4 → 
  ralph_square_size = 8 → 
  elvis_squares = 5 → 
  ralph_squares = 3 → 
  initial_count - (elvis_square_size * elvis_squares + ralph_square_size * ralph_squares) = 6 := by
sorry

end matchsticks_left_l1208_120885


namespace unique_integer_sum_l1208_120895

theorem unique_integer_sum (C y M A : ℕ) : 
  C > 0 ∧ y > 0 ∧ M > 0 ∧ A > 0 →
  C ≠ y ∧ C ≠ M ∧ C ≠ A ∧ y ≠ M ∧ y ≠ A ∧ M ≠ A →
  C + y + M + M + A = 11 →
  M = 1 := by
sorry

end unique_integer_sum_l1208_120895


namespace matrix_equation_proof_l1208_120889

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-21, 19; 15, -13]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -5; 0.5, 3.5]
  M * A = B := by sorry

end matrix_equation_proof_l1208_120889


namespace factor_expression_l1208_120875

theorem factor_expression (x : ℝ) : 60 * x^5 - 180 * x^9 = 60 * x^5 * (1 - 3 * x^4) := by
  sorry

end factor_expression_l1208_120875


namespace arithmetic_to_geometric_sequence_l1208_120892

theorem arithmetic_to_geometric_sequence :
  ∀ (a b c : ℝ),
  (∃ (x : ℝ), a = 3*x ∧ b = 4*x ∧ c = 5*x) →
  (b - a = c - b) →
  ((a + 1) * c = b^2) →
  (a = 15 ∧ b = 20 ∧ c = 25) :=
by
  sorry

end arithmetic_to_geometric_sequence_l1208_120892


namespace f_plus_g_at_2_l1208_120893

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_at_2 (hf : is_even f) (hg : is_odd g) 
  (h : ∀ x, f x - g x = x^3 + 2^(-x)) : 
  f 2 + g 2 = -4 := by
  sorry

end f_plus_g_at_2_l1208_120893


namespace product_xy_is_60_l1208_120841

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem product_xy_is_60 (x y : ℝ) :
  line_k x 6 ∧ line_k 10 y → x * y = 60 := by
  sorry

end product_xy_is_60_l1208_120841


namespace joes_test_scores_l1208_120833

theorem joes_test_scores (scores : Fin 4 → ℝ) 
  (avg_before : (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 35)
  (avg_after : ∃ i, (scores 0 + scores 1 + scores 2 + scores 3 - scores i) / 3 = 40)
  (lowest : ∃ i, ∀ j, scores i ≤ scores j) :
  ∃ i, scores i = 20 ∧ ∀ j, scores i ≤ scores j :=
by sorry

end joes_test_scores_l1208_120833


namespace triangle_area_l1208_120846

theorem triangle_area (t : ℝ) : 
  let base : ℝ := 2 * t
  let height : ℝ := 3 * t + 1
  (1 / 2) * base * height = 3 * t^2 + t :=
by sorry

end triangle_area_l1208_120846


namespace triangle_special_x_values_l1208_120809

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle (a b c R : ℝ) : Prop where
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  circumradius : R = (a * b * c) / (4 * area)
  area_positive : 0 < area

/-- The main theorem -/
theorem triangle_special_x_values
  (a b c : ℝ)
  (h_triangle : Triangle a b c 2)
  (h_angle : a^2 + c^2 ≤ b^2)
  (h_polynomial : ∃ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 = 0) :
  ∃ x : ℝ, x = -1/2 * (Real.sqrt 6 + Real.sqrt 2) ∨ x = -1/2 * (Real.sqrt 6 - Real.sqrt 2) :=
sorry

end triangle_special_x_values_l1208_120809


namespace break_room_tables_l1208_120856

/-- The number of people each table can seat -/
def seating_capacity_per_table : ℕ := 8

/-- The total seating capacity of the break room -/
def total_seating_capacity : ℕ := 32

/-- The number of tables in the break room -/
def number_of_tables : ℕ := total_seating_capacity / seating_capacity_per_table

theorem break_room_tables : number_of_tables = 4 := by
  sorry

end break_room_tables_l1208_120856


namespace product_equals_243_l1208_120862

theorem product_equals_243 : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 = 243 := by
  sorry

end product_equals_243_l1208_120862


namespace sum_x_y_equals_six_l1208_120813

theorem sum_x_y_equals_six (x y : ℝ) 
  (h1 : x^2 + y^2 = 8*x + 4*y - 20) 
  (h2 : x + y = 6) : 
  x + y = 6 := by
sorry

end sum_x_y_equals_six_l1208_120813


namespace subset_implies_bound_l1208_120872

theorem subset_implies_bound (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | 1 < x ∧ x < 2} →
  B = {x : ℝ | x < a} →
  A ⊆ B →
  a ≥ 2 := by
sorry

end subset_implies_bound_l1208_120872


namespace smallest_divisible_ones_l1208_120845

/-- A number composed of n ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- A number composed of n threes -/
def threes (n : ℕ) : ℕ := 3 * ones n

theorem smallest_divisible_ones (n : ℕ) : 
  (∀ k < n, ¬ (threes 100 ∣ ones k)) ∧ (threes 100 ∣ ones n) → n = 300 := by
  sorry

end smallest_divisible_ones_l1208_120845


namespace fraction_square_sum_l1208_120819

theorem fraction_square_sum (a b c d : ℚ) (h : a / b + c / d = 1) :
  (a / b)^2 + c / d = (c / d)^2 + a / b := by sorry

end fraction_square_sum_l1208_120819


namespace office_age_problem_l1208_120871

/-- Given information about the ages of people in an office, prove that the average age of a specific group is 14 years. -/
theorem office_age_problem (total_people : ℕ) (avg_age_all : ℕ) (group1_size : ℕ) (group1_avg_age : ℕ) (group2_size : ℕ) (person15_age : ℕ) :
  total_people = 17 →
  avg_age_all = 15 →
  group1_size = 9 →
  group1_avg_age = 16 →
  group2_size = 5 →
  person15_age = 41 →
  (total_people * avg_age_all - group1_size * group1_avg_age - person15_age) / group2_size = 14 :=
by sorry

end office_age_problem_l1208_120871


namespace power_product_equals_two_l1208_120855

theorem power_product_equals_two :
  (-1/2)^2022 * 2^2023 = 2 := by sorry

end power_product_equals_two_l1208_120855
