import Mathlib

namespace max_trailing_zeros_consecutive_two_digit_numbers_l1055_105553

/-- Two-digit number type -/
def TwoDigitNumber := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

/-- Function to count trailing zeros of a natural number -/
def countTrailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of consecutive zeros at the end of the product 
    of two consecutive two-digit numbers is 2 -/
theorem max_trailing_zeros_consecutive_two_digit_numbers : 
  ∃ (a : TwoDigitNumber), 
    let b : TwoDigitNumber := ⟨a.val + 1, sorry⟩
    countTrailingZeros (a.val * b.val) = 2 ∧ 
    ∀ (x : TwoDigitNumber), 
      let y : TwoDigitNumber := ⟨x.val + 1, sorry⟩
      countTrailingZeros (x.val * y.val) ≤ 2 := by
  sorry

end max_trailing_zeros_consecutive_two_digit_numbers_l1055_105553


namespace thabo_owns_280_books_l1055_105504

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabos_books : BookCollection where
  hardcover_nonfiction := 55
  paperback_nonfiction := 55 + 20
  paperback_fiction := 2 * (55 + 20)

/-- The total number of books in a collection -/
def total_books (bc : BookCollection) : ℕ :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction

/-- Theorem stating that Thabo owns 280 books in total -/
theorem thabo_owns_280_books : total_books thabos_books = 280 := by
  sorry

end thabo_owns_280_books_l1055_105504


namespace bankers_gain_calculation_l1055_105565

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (present_worth : ℝ) (interest_rate : ℝ) (time_period : ℕ) : 
  present_worth = 400 →
  interest_rate = 0.1 →
  time_period = 3 →
  (present_worth * (1 + interest_rate) ^ time_period - present_worth) = 132.4 := by
  sorry

#check bankers_gain_calculation

end bankers_gain_calculation_l1055_105565


namespace right_triangle_hypotenuse_and_perimeter_l1055_105521

theorem right_triangle_hypotenuse_and_perimeter :
  ∀ (a b c : ℝ),
  a = 60 →
  b = 80 →
  c^2 = a^2 + b^2 →
  c = 100 ∧ (a + b + c = 240) :=
by
  sorry

end right_triangle_hypotenuse_and_perimeter_l1055_105521


namespace base_10_to_base_7_l1055_105520

theorem base_10_to_base_7 : 
  ∃ (a b c : Nat), 
    234 = a * 7^2 + b * 7^1 + c * 7^0 ∧ 
    a < 7 ∧ b < 7 ∧ c < 7 ∧
    a = 4 ∧ b = 5 ∧ c = 3 :=
by sorry

end base_10_to_base_7_l1055_105520


namespace pentagon_perimeter_l1055_105539

/-- The perimeter of an irregular pentagon with given side lengths is 52.9 cm -/
theorem pentagon_perimeter (s1 s2 s3 s4 s5 : ℝ) 
  (h1 : s1 = 5.2) (h2 : s2 = 10.3) (h3 : s3 = 15.8) (h4 : s4 = 8.7) (h5 : s5 = 12.9) :
  s1 + s2 + s3 + s4 + s5 = 52.9 := by
  sorry

end pentagon_perimeter_l1055_105539


namespace abs_negative_two_l1055_105570

theorem abs_negative_two : |(-2 : ℤ)| = 2 := by sorry

end abs_negative_two_l1055_105570


namespace subtract_equations_l1055_105575

theorem subtract_equations (x y : ℝ) :
  (4 * x - 3 * y = 2) ∧ (4 * x + y = 10) → 4 * y = 8 := by
  sorry

end subtract_equations_l1055_105575


namespace harriet_round_trip_l1055_105517

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_round_trip 
  (d : ℝ) -- distance between A-ville and B-town in km
  (speed_to_b : ℝ) -- speed from A-ville to B-town in km/h
  (time_to_b : ℝ) -- time taken from A-ville to B-town in hours
  (total_time : ℝ) -- total round trip time in hours
  (h1 : d = speed_to_b * time_to_b) -- distance = speed * time for A-ville to B-town
  (h2 : speed_to_b = 100) -- speed from A-ville to B-town is 100 km/h
  (h3 : time_to_b = 3) -- time taken from A-ville to B-town is 3 hours
  (h4 : total_time = 5) -- total round trip time is 5 hours
  : d / (total_time - time_to_b) = 150 := by
  sorry

end harriet_round_trip_l1055_105517


namespace not_adjacent_probability_l1055_105503

theorem not_adjacent_probability (n : ℕ) (h : n = 10) : 
  (n.choose 2 - (n - 1)) / n.choose 2 = 4 / 5 := by
  sorry

end not_adjacent_probability_l1055_105503


namespace equal_distance_point_exists_l1055_105596

-- Define the plane
variable (Plane : Type)

-- Define points on the plane
variable (P Q O A : Plane)

-- Define the speed
variable (v : ℝ)

-- Define the distance function
variable (dist : Plane → Plane → ℝ)

-- Define the time
variable (t : ℝ)

-- Define the lines as functions of time
variable (line_P line_Q : ℝ → Plane)

-- State the theorem
theorem equal_distance_point_exists :
  (∀ t, dist O (line_P t) = v * t) →  -- P moves with constant speed v
  (∀ t, dist O (line_Q t) = v * t) →  -- Q moves with constant speed v
  (∃ t₀, line_P t₀ = O ∧ line_Q t₀ = O) →  -- The lines intersect at O
  ∃ A : Plane, ∀ t, dist A (line_P t) = dist A (line_Q t) :=
by sorry

end equal_distance_point_exists_l1055_105596


namespace a_value_m_range_l1055_105516

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Theorem 1: Prove that a = 1
theorem a_value (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Theorem 2: Prove that the minimum value of m is 4
theorem m_range : 
  ∃ m : ℝ, (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) ∧
  (∀ m' : ℝ, (∃ n : ℝ, f 1 n ≤ m' - f 1 (-n)) → m' ≥ m) ∧
  m = 4 := by sorry

end a_value_m_range_l1055_105516


namespace train_station_problem_l1055_105536

theorem train_station_problem :
  ∀ (x v : ℕ),
  v > 3 →
  x = (2 * v) / (v - 3) →
  x - 5 > 0 →
  x / v - (x - 5) / 3 = 1 →
  (x = 8 ∧ v = 4) :=
by
  sorry

end train_station_problem_l1055_105536


namespace mindmaster_codes_l1055_105518

theorem mindmaster_codes (num_slots : ℕ) (num_colors : ℕ) : 
  num_slots = 5 → num_colors = 7 → num_colors ^ num_slots = 16807 := by
  sorry

end mindmaster_codes_l1055_105518


namespace attendance_difference_l1055_105561

/-- Proves that given the initial ratio of boys to girls to adults as 9.5:6.25:4.75,
    with 30% of attendees being girls, and after 15% of girls and 20% of adults leave,
    the percentage difference between boys and the combined number of remaining girls
    and adults is approximately 2.304%. -/
theorem attendance_difference (boys girls adults : ℝ) 
    (h_ratio : boys = 9.5 ∧ girls = 6.25 ∧ adults = 4.75)
    (h_girls_percent : girls / (boys + girls + adults) = 0.3)
    (h_girls_left : ℝ) (h_adults_left : ℝ)
    (h_girls_left_percent : h_girls_left = 0.15)
    (h_adults_left_percent : h_adults_left = 0.2) :
    let total := boys + girls + adults
    let boys_percent := boys / total
    let girls_adjusted := girls * (1 - h_girls_left)
    let adults_adjusted := adults * (1 - h_adults_left)
    let girls_adults_adjusted_percent := (girls_adjusted + adults_adjusted) / total
    abs (boys_percent - girls_adults_adjusted_percent - 0.02304) < 0.00001 := by
  sorry

end attendance_difference_l1055_105561


namespace right_triangle_area_l1055_105568

theorem right_triangle_area (a b c : ℝ) (ha : a^2 = 100) (hb : b^2 = 64) (hc : c^2 = 121)
  (h_right : a^2 + b^2 = c^2) : (1/2) * a * b = 40 := by
  sorry

end right_triangle_area_l1055_105568


namespace profit_increase_l1055_105528

theorem profit_increase (x : ℝ) : 
  (1 + x / 100) * 0.8 * 1.5 = 1.68 → x = 40 := by
  sorry

end profit_increase_l1055_105528


namespace decagon_diagonal_intersections_count_l1055_105550

/-- The number of intersection points of diagonals in a regular decagon -/
def decagon_diagonal_intersections : ℕ :=
  Nat.choose 10 4

/-- Theorem stating that the number of interior intersection points of diagonals
    in a regular decagon is equal to the number of ways to choose 4 vertices from 10 -/
theorem decagon_diagonal_intersections_count :
  decagon_diagonal_intersections = 210 := by
  sorry

#eval decagon_diagonal_intersections

end decagon_diagonal_intersections_count_l1055_105550


namespace set_operation_equality_l1055_105557

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define sets A and B
def A : Set Nat := {0, 3, 4}
def B : Set Nat := {1, 3}

-- State the theorem
theorem set_operation_equality :
  (Aᶜ ∪ A) ∪ B = {1, 2, 3} :=
by sorry

end set_operation_equality_l1055_105557


namespace f_f_10_equals_1_l1055_105574

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 10^(x-1) else Real.log x / Real.log 10

-- State the theorem
theorem f_f_10_equals_1 : f (f 10) = 1 := by sorry

end f_f_10_equals_1_l1055_105574


namespace doubled_average_l1055_105587

theorem doubled_average (n : ℕ) (original_avg : ℚ) (h1 : n = 12) (h2 : original_avg = 50) :
  let total_marks := n * original_avg
  let doubled_marks := 2 * total_marks
  let new_avg := doubled_marks / n
  new_avg = 100 := by sorry

end doubled_average_l1055_105587


namespace trig_identity_proof_l1055_105502

theorem trig_identity_proof :
  Real.sin (50 * π / 180) * Real.cos (20 * π / 180) -
  Real.cos (50 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
  sorry

end trig_identity_proof_l1055_105502


namespace circle_diameter_l1055_105580

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 100 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 20 := by
  sorry

end circle_diameter_l1055_105580


namespace fifteen_ways_to_select_l1055_105566

/-- Represents the number of ways to select performers for singing and dancing -/
def select_performers (total : ℕ) (dancers : ℕ) (singers : ℕ) : ℕ :=
  let both := dancers + singers - total
  let pure_singers := singers - both
  let pure_dancers := dancers - both
  both * pure_dancers + pure_singers * both

/-- Theorem stating that there are 15 ways to select performers from a group of 8,
    where 6 can dance and 5 can sing -/
theorem fifteen_ways_to_select :
  select_performers 8 6 5 = 15 := by
  sorry

#eval select_performers 8 6 5

end fifteen_ways_to_select_l1055_105566


namespace part_one_part_two_l1055_105577

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b

def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem for part (1)
theorem part_one (a b : ℝ) : 2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := by
  sorry

-- Theorem for part (2)
theorem part_two : (∀ a b : ℝ, ∃ c : ℝ, 2 * A a b - B a b = c) → (∀ a : ℝ, 2 * A a 2 - B a 2 = 2 * A 0 2 - B 0 2) := by
  sorry

end part_one_part_two_l1055_105577


namespace tv_price_change_l1055_105583

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.8 * 1.45) = P * 1.16 := by
  sorry

end tv_price_change_l1055_105583


namespace greatest_multiple_of_four_under_cube_root_2000_l1055_105501

theorem greatest_multiple_of_four_under_cube_root_2000 :
  ∀ x : ℕ, 
    x > 0 → 
    x % 4 = 0 → 
    x^3 < 2000 → 
    x ≤ 12 ∧ 
    ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^3 < 2000 ∧ y = 12 :=
by sorry

end greatest_multiple_of_four_under_cube_root_2000_l1055_105501


namespace tim_kittens_count_tim_final_kitten_count_l1055_105573

theorem tim_kittens_count : ℕ → ℕ → ℕ → ℕ
  | initial_kittens, sara_kittens, adoption_rate =>
    let kittens_after_jessica := initial_kittens - initial_kittens / 3
    let kittens_before_adoption := kittens_after_jessica + sara_kittens
    let adopted_kittens := sara_kittens * adoption_rate / 100
    kittens_before_adoption - adopted_kittens

theorem tim_final_kitten_count :
  tim_kittens_count 12 14 50 = 15 := by
  sorry

end tim_kittens_count_tim_final_kitten_count_l1055_105573


namespace nickels_remaining_l1055_105527

def initial_nickels : ℕ := 87
def borrowed_nickels : ℕ := 75

theorem nickels_remaining (initial : ℕ) (borrowed : ℕ) :
  initial ≥ borrowed → initial - borrowed = initial_nickels - borrowed_nickels :=
by sorry

end nickels_remaining_l1055_105527


namespace club_members_proof_l1055_105537

theorem club_members_proof (total : Nat) (left_handed : Nat) (rock_fans : Nat) (right_handed_non_rock : Nat) 
  (h1 : total = 30)
  (h2 : left_handed = 12)
  (h3 : rock_fans = 20)
  (h4 : right_handed_non_rock = 3)
  (h5 : ∀ x : Nat, x ≤ total → x = (left_handed + (total - left_handed)))
  : ∃ x : Nat, x = 5 ∧ 
    x + (left_handed - x) + (rock_fans - x) + right_handed_non_rock = total := by
  sorry


end club_members_proof_l1055_105537


namespace words_with_e_count_l1055_105591

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding E -/
def alphabet_size_without_e : ℕ := 4

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words without E -/
def words_without_e : ℕ := alphabet_size_without_e ^ word_length

/-- The number of words with at least one E -/
def words_with_e : ℕ := total_words - words_without_e

theorem words_with_e_count : words_with_e = 369 := by
  sorry

end words_with_e_count_l1055_105591


namespace smallest_divisible_term_l1055_105563

/-- Geometric sequence with first term a and common ratio r -/
def geometricSequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

/-- The common ratio of the geometric sequence -/
def commonRatio : ℚ := 25 / (5/6)

/-- The nth term of the specific geometric sequence -/
def nthTerm (n : ℕ) : ℚ := geometricSequence (5/6) commonRatio n

/-- Predicate to check if a rational number is divisible by 2,000,000 -/
def divisibleByTwoMillion (q : ℚ) : Prop := ∃ (k : ℤ), q = (2000000 : ℚ) * k

/-- Statement: 8 is the smallest positive integer n such that the nth term 
    of the geometric sequence is divisible by 2,000,000 -/
theorem smallest_divisible_term : 
  (∀ m : ℕ, m < 8 → ¬(divisibleByTwoMillion (nthTerm m))) ∧ 
  (divisibleByTwoMillion (nthTerm 8)) := by sorry

end smallest_divisible_term_l1055_105563


namespace quadratic_roots_sum_product_l1055_105542

theorem quadratic_roots_sum_product (k p : ℝ) : 
  (∃ α β : ℝ, 3 * α^2 - k * α + p = 0 ∧ 3 * β^2 - k * β + p = 0) →
  (∃ α β : ℝ, α + β = 9 ∧ α * β = 10) →
  k + p = 57 := by
sorry

end quadratic_roots_sum_product_l1055_105542


namespace cos_2x_plus_pi_third_equiv_sin_2x_shifted_l1055_105515

theorem cos_2x_plus_pi_third_equiv_sin_2x_shifted (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) := by
  sorry

end cos_2x_plus_pi_third_equiv_sin_2x_shifted_l1055_105515


namespace imaginary_part_of_z_l1055_105535

theorem imaginary_part_of_z (z : ℂ) : z * (1 - 2*I) = Complex.abs (3 + 4*I) → Complex.im z = 2 := by
  sorry

end imaginary_part_of_z_l1055_105535


namespace banknote_sum_divisibility_l1055_105585

theorem banknote_sum_divisibility
  (a b : ℕ)
  (h_distinct : a % 101 ≠ b % 101)
  (h_total : ℕ)
  (h_count : h_total = 100) :
  ∃ (m n : ℕ), m + n ≤ h_total ∧ (m * a + n * b) % 101 = 0 :=
sorry

end banknote_sum_divisibility_l1055_105585


namespace rongcheng_sample_points_l1055_105541

/-- Represents the number of observation points in each county -/
structure ObservationPoints where
  xiongxian : ℕ
  rongcheng : ℕ
  anxin : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Checks if three numbers form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Calculates the number of data points for stratified sampling -/
def stratified_sample (total_samples : ℕ) (points : ObservationPoints) (county : ℕ) : ℕ :=
  (county * total_samples) / (points.xiongxian + points.rongcheng + points.anxin)

theorem rongcheng_sample_points :
  ∀ (points : ObservationPoints),
    points.xiongxian = 6 →
    is_arithmetic_sequence points.xiongxian points.rongcheng points.anxin →
    is_geometric_sequence points.xiongxian points.rongcheng (points.anxin + 6) →
    stratified_sample 12 points points.rongcheng = 4 :=
by sorry

end rongcheng_sample_points_l1055_105541


namespace inequality_equivalence_l1055_105545

theorem inequality_equivalence (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 3) := by
  sorry

end inequality_equivalence_l1055_105545


namespace cape_may_has_24_sightings_l1055_105578

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := 16

/-- The total number of shark sightings in Cape May and Daytona Beach -/
def total_sightings : ℕ := 40

/-- Theorem stating that Cape May has 24 shark sightings given the conditions -/
theorem cape_may_has_24_sightings :
  cape_may_sightings = 24 ∧
  cape_may_sightings + daytona_beach_sightings = total_sightings ∧
  cape_may_sightings = 2 * daytona_beach_sightings - 8 :=
by sorry

end cape_may_has_24_sightings_l1055_105578


namespace tan_45_degrees_l1055_105547

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l1055_105547


namespace intersecting_line_circle_isosceles_right_triangle_l1055_105507

/-- Given a line and a circle that intersect at two points forming an isosceles right triangle with a third point, prove the value of the parameter a. -/
theorem intersecting_line_circle_isosceles_right_triangle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 + a * A.2 - 1 = 0 ∧ (A.1 + a)^2 + (A.2 - 1)^2 = 1) ∧
    (B.1 + a * B.2 - 1 = 0 ∧ (B.1 + a)^2 + (B.2 - 1)^2 = 1) ∧
    A ≠ B) →
  (∃ C : ℝ × ℝ, 
    (C.1 + a * C.2 - 1 ≠ 0 ∨ (C.1 + a)^2 + (C.2 - 1)^2 ≠ 1) ∧
    (dist A C = dist B C ∧ dist A B = dist A C * Real.sqrt 2)) →
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry


end intersecting_line_circle_isosceles_right_triangle_l1055_105507


namespace trapezoid_properties_l1055_105500

structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  FH : ℝ
  height : ℝ

def perimeter (t : Trapezoid) : ℝ := t.EF + t.GH + t.EG + t.FH

theorem trapezoid_properties (t : Trapezoid) 
  (h1 : t.EF = 60)
  (h2 : t.GH = 30)
  (h3 : t.EG = 40)
  (h4 : t.FH = 50)
  (h5 : t.height = 24) :
  perimeter t = 191 ∧ t.EG = 51 := by
  sorry

end trapezoid_properties_l1055_105500


namespace ellipse_intersection_slope_l1055_105558

/-- Given an ellipse ax^2 + by^2 = 1 intersecting with the line y = 1 - x,
    if the slope of the line passing through the origin and the midpoint
    of the intersection points is √3/2, then a/b = √3/2. -/
theorem ellipse_intersection_slope (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 + b * (1 - x₁)^2 = 1 ∧
                a * x₂^2 + b * (1 - x₂)^2 = 1 ∧
                x₁ ≠ x₂) →
  ((b / (a + b)) / (a / (a + b)) = Real.sqrt 3 / 2) →
  a / b = Real.sqrt 3 / 2 :=
by sorry

end ellipse_intersection_slope_l1055_105558


namespace arithmetic_mean_problem_l1055_105598

theorem arithmetic_mean_problem (x y : ℚ) :
  (((3 * x + 12) + (2 * y + 18) + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) →
  (x = 2 * y) →
  (x = 254 / 15 ∧ y = 127 / 15) := by
sorry

end arithmetic_mean_problem_l1055_105598


namespace b_range_l1055_105524

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem b_range (a b : ℝ) (h : f a = g b) : 
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by sorry

end b_range_l1055_105524


namespace wednesday_bags_is_nine_l1055_105584

/-- Represents the leaf raking business scenario -/
structure LeafRakingBusiness where
  charge_per_bag : ℕ
  monday_bags : ℕ
  tuesday_bags : ℕ
  total_earnings : ℕ

/-- Calculates the number of bags raked on Wednesday -/
def bags_on_wednesday (business : LeafRakingBusiness) : ℕ :=
  (business.total_earnings - business.charge_per_bag * (business.monday_bags + business.tuesday_bags)) / business.charge_per_bag

/-- Theorem stating that the number of bags raked on Wednesday is 9 -/
theorem wednesday_bags_is_nine (business : LeafRakingBusiness)
  (h1 : business.charge_per_bag = 4)
  (h2 : business.monday_bags = 5)
  (h3 : business.tuesday_bags = 3)
  (h4 : business.total_earnings = 68) :
  bags_on_wednesday business = 9 := by
  sorry

#eval bags_on_wednesday { charge_per_bag := 4, monday_bags := 5, tuesday_bags := 3, total_earnings := 68 }

end wednesday_bags_is_nine_l1055_105584


namespace little_john_money_l1055_105554

/-- Little John's money problem -/
theorem little_john_money (sweet_cost : ℚ) (friend_gift : ℚ) (num_friends : ℕ) (money_left : ℚ) 
  (h1 : sweet_cost = 105/100)
  (h2 : friend_gift = 1)
  (h3 : num_friends = 2)
  (h4 : money_left = 205/100) :
  sweet_cost + num_friends * friend_gift + money_left = 51/10 := by
  sorry

end little_john_money_l1055_105554


namespace union_of_sets_l1055_105544

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {m : ℕ | m = 1 ∨ m = 4 ∨ m = 7}

theorem union_of_sets (h : A ∩ B = {1, 4}) : A ∪ B = {1, 2, 3, 4, 7} := by
  sorry

end union_of_sets_l1055_105544


namespace speedster_fraction_l1055_105564

/-- Represents the inventory of vehicles -/
structure Inventory where
  speedsters : ℕ
  nonSpeedsters : ℕ

/-- The fraction of Speedsters that are convertibles -/
def convertibleFraction : ℚ := 3/5

/-- The number of Speedster convertibles -/
def speedsterConvertibles : ℕ := 54

/-- The number of non-Speedster vehicles -/
def nonSpeedsterCount : ℕ := 30

/-- Theorem: The fraction of Speedsters in the inventory is 3/4 -/
theorem speedster_fraction (inv : Inventory) 
  (h1 : inv.speedsters * convertibleFraction = speedsterConvertibles)
  (h2 : inv.nonSpeedsters = nonSpeedsterCount) :
  (inv.speedsters : ℚ) / (inv.speedsters + inv.nonSpeedsters) = 3/4 := by
  sorry

end speedster_fraction_l1055_105564


namespace percentage_passed_both_subjects_l1055_105519

theorem percentage_passed_both_subjects (total_students : ℕ) 
  (failed_hindi : ℕ) (failed_english : ℕ) (failed_both : ℕ) :
  failed_hindi = (35 * total_students) / 100 →
  failed_english = (45 * total_students) / 100 →
  failed_both = (20 * total_students) / 100 →
  total_students > 0 →
  ((total_students - (failed_hindi + failed_english - failed_both)) * 100) / total_students = 40 := by
sorry

end percentage_passed_both_subjects_l1055_105519


namespace triangleCount_is_sixteen_l1055_105559

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid :=
  (rows : ℕ)

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows * (grid.rows + 1)) / 2

/-- Counts the number of medium triangles in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows - 1) * grid.rows / 2

/-- Counts the number of large triangles in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid) : ℕ := 1

/-- Counts the number of extra large triangles in a triangular grid -/
def countExtraLargeTriangles (grid : TriangularGrid) : ℕ := 1

/-- Counts the total number of triangles in a triangular grid -/
def countTotalTriangles (grid : TriangularGrid) : ℕ :=
  countSmallTriangles grid + countMediumTriangles grid + 
  countLargeTriangles grid + countExtraLargeTriangles grid

theorem triangleCount_is_sixteen :
  ∀ (grid : TriangularGrid), grid.rows = 4 → countTotalTriangles grid = 16 := by
  sorry

end triangleCount_is_sixteen_l1055_105559


namespace vector_expressions_equal_AD_l1055_105552

variable {V : Type*} [AddCommGroup V]

variable (A B C D M O : V)

theorem vector_expressions_equal_AD :
  (A - D + M - B) + (B - C + C - M) = A - D ∧
  (A - B + C - D) + (B - C) = A - D ∧
  (O - C) - (O - A) + (C - D) = A - D :=
by sorry

end vector_expressions_equal_AD_l1055_105552


namespace problem_p5_l1055_105543

theorem problem_p5 (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : a * d - b * c = 1/7) :
  a * c + b * d = 4 * Real.sqrt 3 / 7 := by
sorry

end problem_p5_l1055_105543


namespace roots_and_p_value_l1055_105540

-- Define the polynomial
def f (p : ℝ) (x : ℝ) : ℝ := x^3 + 7*x^2 + 14*x - p

-- Define the condition of three distinct roots in geometric progression
def has_three_distinct_roots_in_gp (p : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f p a = 0 ∧ f p b = 0 ∧ f p c = 0 ∧
    ∃ (r : ℝ), r ≠ 0 ∧ r ≠ 1 ∧ b = a * r ∧ c = b * r

-- Theorem statement
theorem roots_and_p_value (p : ℝ) :
  has_three_distinct_roots_in_gp p →
  p = -8 ∧ f p (-1) = 0 ∧ f p (-2) = 0 ∧ f p (-4) = 0 :=
by sorry

end roots_and_p_value_l1055_105540


namespace tony_future_age_l1055_105593

def jacob_age : ℕ := 24
def tony_age : ℕ := jacob_age / 2
def years_passed : ℕ := 6

theorem tony_future_age :
  tony_age + years_passed = 18 := by
  sorry

end tony_future_age_l1055_105593


namespace percentage_of_women_in_non_union_l1055_105549

theorem percentage_of_women_in_non_union (total : ℝ) (h1 : total > 0) : 
  let men := 0.48 * total
  let unionized := 0.60 * total
  let non_unionized := total - unionized
  let women_non_union := 0.85 * non_unionized
  women_non_union / non_unionized = 0.85 := by
  sorry

end percentage_of_women_in_non_union_l1055_105549


namespace main_theorem_l1055_105534

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, Differentiable ℝ f ∧ f x + (deriv^[2] f) x > 0

/-- The main theorem -/
theorem main_theorem (f : ℝ → ℝ) (hf : satisfies_condition f) :
  ∀ a b : ℝ, a > b ↔ Real.exp a * f a > Real.exp b * f b :=
by sorry

end main_theorem_l1055_105534


namespace even_product_probability_l1055_105522

def eight_sided_die := Finset.range 8

theorem even_product_probability :
  let outcomes := eight_sided_die.product eight_sided_die
  (outcomes.filter (fun (x, y) => (x + 1) * (y + 1) % 2 = 0)).card / outcomes.card = 3 / 4 := by
  sorry

end even_product_probability_l1055_105522


namespace complex_real_condition_l1055_105533

theorem complex_real_condition (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a^2 - 1) : ℂ) + (Complex.I : ℂ) * (a + 1)).im = 0 →
  a = -1 := by
  sorry

end complex_real_condition_l1055_105533


namespace complex_equation_solution_l1055_105588

theorem complex_equation_solution (a : ℝ) (h : (1 + a * Complex.I) * Complex.I = 3 + Complex.I) : a = -3 := by
  sorry

end complex_equation_solution_l1055_105588


namespace coprime_linear_combination_l1055_105560

theorem coprime_linear_combination (a b n : ℕ+) (h1 : Nat.Coprime a b) (h2 : n > a * b) :
  ∃ (x y : ℕ+), n = a * x + b * y := by
sorry

end coprime_linear_combination_l1055_105560


namespace opposite_sign_sum_l1055_105514

theorem opposite_sign_sum (a b : ℝ) : 
  (|a + 1| + |b + 2| = 0) → (a + b = -3) := by
  sorry

end opposite_sign_sum_l1055_105514


namespace unique_three_digit_number_l1055_105509

theorem unique_three_digit_number : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (∃ (π b γ : ℕ),
    π ≠ b ∧ π ≠ γ ∧ b ≠ γ ∧
    0 ≤ π ∧ π ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ γ ∧ γ ≤ 9 ∧
    n = 100 * π + 10 * b + γ ∧
    n = (π + b + γ) * (π + b + γ + 1)) ∧
  n = 156 :=
by sorry

end unique_three_digit_number_l1055_105509


namespace railroad_grade_reduction_l1055_105526

theorem railroad_grade_reduction (rise : ℝ) (initial_grade : ℝ) (reduced_grade : ℝ) :
  rise = 800 →
  initial_grade = 0.04 →
  reduced_grade = 0.03 →
  ⌊(rise / reduced_grade - rise / initial_grade)⌋ = 6667 := by
  sorry

end railroad_grade_reduction_l1055_105526


namespace rectangle_area_l1055_105599

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) :
  length = 2 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 120 →
  area = length * width →
  area = 800 := by
  sorry

end rectangle_area_l1055_105599


namespace parabola_above_l1055_105505

theorem parabola_above (k : ℝ) : 
  (∀ x : ℝ, 2*x^2 - 2*k*x + (k^2 + 2*k + 2) > x^2 + 2*k*x - 2*k^2 - 1) ↔ 
  (-1 < k ∧ k < 3) :=
by sorry

end parabola_above_l1055_105505


namespace quadratic_inequality_impossibility_l1055_105530

/-- Given a quadratic function f(x) = ax^2 + 2ax + 1 where a ≠ 0,
    it is impossible for f(-2) > f(-1) > f(0) to be true. -/
theorem quadratic_inequality_impossibility (a : ℝ) (h : a ≠ 0) :
  ¬∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 + 2 * a * x + 1) ∧ 
  (f (-2) > f (-1) ∧ f (-1) > f 0) := by
  sorry

end quadratic_inequality_impossibility_l1055_105530


namespace set_A_representation_l1055_105548

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_representation : A = {(-1, 0), (0, -1), (1, 0)} := by
  sorry

end set_A_representation_l1055_105548


namespace wand_price_l1055_105512

theorem wand_price (P : ℚ) : (P * (1/8) = 8) → P = 64 := by
  sorry

end wand_price_l1055_105512


namespace wooden_stick_problem_xiao_hong_age_problem_l1055_105546

-- Problem 1: Wooden stick
theorem wooden_stick_problem (x : ℝ) :
  60 - 2 * x = 10 → x = 25 := by sorry

-- Problem 2: Xiao Hong's age
theorem xiao_hong_age_problem (y : ℝ) :
  2 * y + 10 = 30 → y = 10 := by sorry

end wooden_stick_problem_xiao_hong_age_problem_l1055_105546


namespace power_multiplication_l1055_105582

theorem power_multiplication : 3^6 * 4^3 = 46656 := by
  sorry

end power_multiplication_l1055_105582


namespace katya_magic_pen_problem_l1055_105572

theorem katya_magic_pen_problem (total_problems : ℕ) 
  (katya_prob : ℚ) (pen_prob : ℚ) (min_correct : ℕ) :
  total_problems = 20 →
  katya_prob = 4/5 →
  pen_prob = 1/2 →
  min_correct = 13 →
  ∃ (x : ℕ), x ≥ 10 ∧ 
    (x : ℚ) * katya_prob + (total_problems - x : ℚ) * pen_prob ≥ min_correct :=
by sorry

end katya_magic_pen_problem_l1055_105572


namespace system_solution_l1055_105595

theorem system_solution (x y z : ℝ) 
  (eq1 : x + y + z = 10)
  (eq2 : x * z = y^2)
  (eq3 : z^2 + y^2 = x^2) :
  z = 5 - Real.sqrt (Real.sqrt 3125 - 50) :=
by sorry

end system_solution_l1055_105595


namespace percent_error_multiplication_l1055_105556

theorem percent_error_multiplication (x : ℝ) (h : x > 0) : 
  (|12 * x - x / 3| / (x / 3)) * 100 = 3500 := by
sorry

end percent_error_multiplication_l1055_105556


namespace matrix_determinant_sixteen_l1055_105511

def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; x, 4*x]

theorem matrix_determinant_sixteen (x : ℝ) : 
  Matrix.det (matrix x) = 16 ↔ x = 4/3 ∨ x = -1 := by
  sorry

end matrix_determinant_sixteen_l1055_105511


namespace least_sum_m_n_l1055_105523

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (m.val.gcd (330 : ℕ) = 1) ∧ 
  (n.val.gcd (330 : ℕ) = 1) ∧
  ((m + n).val.gcd (330 : ℕ) = 1) ∧
  (∃ (k : ℕ), m.val^m.val = k * n.val^n.val) ∧
  (∀ (l : ℕ+), m.val ≠ l.val * n.val) ∧
  (∀ (p q : ℕ+), 
    (p.val.gcd (330 : ℕ) = 1) ∧ 
    (q.val.gcd (330 : ℕ) = 1) ∧
    ((p + q).val.gcd (330 : ℕ) = 1) ∧
    (∃ (r : ℕ), p.val^p.val = r * q.val^q.val) ∧
    (∀ (s : ℕ+), p.val ≠ s.val * q.val) →
    (m + n).val ≤ (p + q).val) ∧
  (m + n).val = 154 :=
by sorry

end least_sum_m_n_l1055_105523


namespace geometric_progression_identity_l1055_105555

theorem geometric_progression_identity 
  (a r : ℝ) (n p k : ℕ) (A B C : ℝ) 
  (hA : A = a * r^(n - 1)) 
  (hB : B = a * r^(p - 1)) 
  (hC : C = a * r^(k - 1)) :
  A^(p - k) * B^(k - n) * C^(n - p) = 1 := by
  sorry


end geometric_progression_identity_l1055_105555


namespace physics_marks_l1055_105590

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 80)
  (avg_PM : (P + M) / 2 = 90)
  (avg_PC : (P + C) / 2 = 70) :
  P = 80 := by
sorry

end physics_marks_l1055_105590


namespace relationship_abc_l1055_105594

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := 2^(-1/3 : ℝ)
noncomputable def c : ℝ := Real.log 30 / Real.log 3

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by sorry

end relationship_abc_l1055_105594


namespace factorization_equality_l1055_105506

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 3 * y = 3 * y * (x + 1) * (x - 1) := by
  sorry

end factorization_equality_l1055_105506


namespace min_value_theorem_l1055_105589

theorem min_value_theorem (x y : ℝ) (h1 : x * y = 1/2) (h2 : 0 < x ∧ x < 1) (h3 : 0 < y ∧ y < 1) :
  (2 / (1 - x)) + (1 / (1 - y)) ≥ 10 :=
sorry

end min_value_theorem_l1055_105589


namespace minRainfallDay4_is_21_l1055_105567

/-- Represents the rainfall data and conditions for a 4-day storm --/
structure RainfallData where
  capacity : ℝ  -- Area capacity in inches
  drain_rate : ℝ  -- Daily drainage rate in inches
  day1_rain : ℝ  -- Rainfall on day 1 in inches
  day2_rain : ℝ  -- Rainfall on day 2 in inches
  day3_rain : ℝ  -- Rainfall on day 3 in inches

/-- Calculates the minimum rainfall on day 4 for flooding to occur --/
def minRainfallDay4 (data : RainfallData) : ℝ :=
  data.capacity - (data.day1_rain + data.day2_rain + data.day3_rain - 3 * data.drain_rate)

/-- Theorem stating the minimum rainfall on day 4 for flooding --/
theorem minRainfallDay4_is_21 (data : RainfallData) :
  data.capacity = 72 ∧
  data.drain_rate = 3 ∧
  data.day1_rain = 10 ∧
  data.day2_rain = 2 * data.day1_rain ∧
  data.day3_rain = 1.5 * data.day2_rain →
  minRainfallDay4 data = 21 := by
  sorry

#eval minRainfallDay4 {
  capacity := 72,
  drain_rate := 3,
  day1_rain := 10,
  day2_rain := 20,
  day3_rain := 30
}

end minRainfallDay4_is_21_l1055_105567


namespace tangent_sum_over_cosine_l1055_105529

theorem tangent_sum_over_cosine (x : Real) :
  let a := x * π / 180  -- Convert degrees to radians
  (Real.tan a + Real.tan (2*a) + Real.tan (7*a) + Real.tan (8*a)) / Real.cos a = 32 :=
by
  sorry

end tangent_sum_over_cosine_l1055_105529


namespace log_stack_sum_l1055_105586

theorem log_stack_sum (n : ℕ) (a l : ℕ) (h1 : n = 12) (h2 : a = 15) (h3 : l = 4) :
  n * (a + l) / 2 = 114 := by
  sorry

end log_stack_sum_l1055_105586


namespace vector_subtraction_l1055_105525

/-- Given two vectors OA and OB in 2D space, prove that the vector AB is their difference -/
theorem vector_subtraction (OA OB : ℝ × ℝ) (h1 : OA = (2, 8)) (h2 : OB = (-7, 2)) :
  OB - OA = (-9, -6) := by
  sorry

end vector_subtraction_l1055_105525


namespace fraction_inequality_l1055_105551

theorem fraction_inequality (x y z a b c r : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  (x + y + a + b) / (x + y + a + b + c + r) + (y + z + b + c) / (y + z + a + b + c + r) >
  (x + z + a + c) / (x + z + a + b + c + r) := by
  sorry

end fraction_inequality_l1055_105551


namespace front_yard_eggs_count_l1055_105576

/-- The number of eggs in June's front yard nest -/
def front_yard_eggs : ℕ := sorry

/-- The total number of eggs June found -/
def total_eggs : ℕ := 17

/-- The number of nests in the first tree -/
def nests_in_first_tree : ℕ := 2

/-- The number of eggs in each nest in the first tree -/
def eggs_per_nest_first_tree : ℕ := 5

/-- The number of nests in the second tree -/
def nests_in_second_tree : ℕ := 1

/-- The number of eggs in the nest in the second tree -/
def eggs_in_second_tree : ℕ := 3

theorem front_yard_eggs_count :
  front_yard_eggs = total_eggs - (nests_in_first_tree * eggs_per_nest_first_tree + nests_in_second_tree * eggs_in_second_tree) :=
by sorry

end front_yard_eggs_count_l1055_105576


namespace abc_condition_neither_sufficient_nor_necessary_l1055_105592

theorem abc_condition_neither_sufficient_nor_necessary :
  ¬ (∀ a b c : ℝ, a * b * c = 1 → 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c ≤ a + b + c) ∧
  ¬ (∀ a b c : ℝ, 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c ≤ a + b + c → a * b * c = 1) :=
by sorry

end abc_condition_neither_sufficient_nor_necessary_l1055_105592


namespace intersection_condition_longest_chord_l1055_105531

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem 1: Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem 2: Longest chord
theorem longest_chord :
  ∃ x y : ℝ, ellipse x y ∧ line x y 0 ∧
  ∀ m x' y' : ℝ, ellipse x' y' ∧ line x' y' m →
    (x - y)^2 ≥ (x' - y')^2 :=
sorry

end intersection_condition_longest_chord_l1055_105531


namespace root_equation_implies_expression_value_l1055_105562

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 - 2*m - 1 = 0 → (m-1)^2 - (m-3)*(m+3) - (m-1)*(m-3) = 6 := by
  sorry

end root_equation_implies_expression_value_l1055_105562


namespace inequality_proof_l1055_105597

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (c * a / (b + c * a)) ≥ 1 := by
  sorry

end inequality_proof_l1055_105597


namespace average_leaves_per_hour_l1055_105508

/-- Represents the leaf fall pattern of a tree over 3 hours -/
structure TreeLeafFall where
  hour1 : ℕ
  hour2 : ℕ
  hour3 : ℕ

/-- Calculates the total number of leaves that fell from a tree -/
def totalLeaves (tree : TreeLeafFall) : ℕ :=
  tree.hour1 + tree.hour2 + tree.hour3

/-- Represents the leaf fall patterns of two trees in Rylee's backyard -/
def ryleesBackyard : (TreeLeafFall × TreeLeafFall) :=
  (⟨7, 12, 9⟩, ⟨4, 4, 6⟩)

/-- The number of hours of observation -/
def observationHours : ℕ := 3

/-- Theorem: The average number of leaves falling per hour across both trees is 14 -/
theorem average_leaves_per_hour :
  (totalLeaves ryleesBackyard.1 + totalLeaves ryleesBackyard.2) / observationHours = 14 :=
by sorry

end average_leaves_per_hour_l1055_105508


namespace unique_ecuadorian_number_l1055_105538

def is_ecuadorian (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧  -- Three-digit number
  n % 10 ≠ 0 ∧  -- Does not end in 0
  n % 36 = 0 ∧  -- Multiple of 36
  (n - (n % 10 * 100 + (n / 10 % 10) * 10 + n / 100)) > 0 ∧  -- abc - cba > 0
  (n - (n % 10 * 100 + (n / 10 % 10) * 10 + n / 100)) % 36 = 0  -- (abc - cba) is multiple of 36

theorem unique_ecuadorian_number : ∃! n : ℕ, is_ecuadorian n ∧ n = 864 := by sorry

end unique_ecuadorian_number_l1055_105538


namespace total_trees_in_park_l1055_105571

theorem total_trees_in_park (ancient_oaks : ℕ) (fir_trees : ℕ) (saplings : ℕ)
  (h1 : ancient_oaks = 15)
  (h2 : fir_trees = 23)
  (h3 : saplings = 58) :
  ancient_oaks + fir_trees + saplings = 96 :=
by sorry

end total_trees_in_park_l1055_105571


namespace garage_visitors_l1055_105532

/-- Given a number of cars, selections per car, and selections per client,
    calculate the number of clients who visited the garage. -/
def clientsVisited (numCars : ℕ) (selectionsPerCar : ℕ) (selectionsPerClient : ℕ) : ℕ :=
  (numCars * selectionsPerCar) / selectionsPerClient

/-- Theorem stating that given 15 cars, where each car is selected exactly 3 times,
    and each client selects 3 cars, the number of clients who visited the garage is 15. -/
theorem garage_visitors :
  clientsVisited 15 3 3 = 15 := by
  sorry

end garage_visitors_l1055_105532


namespace tangent_problem_l1055_105510

theorem tangent_problem (α β : ℝ) 
  (h1 : Real.tan (α - 2 * β) = 4)
  (h2 : Real.tan β = 2) :
  (Real.tan α - 2) / (1 + 2 * Real.tan α) = -6/7 := by
  sorry

end tangent_problem_l1055_105510


namespace rectangular_prism_inequality_l1055_105579

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0)
  (h_diagonal : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end rectangular_prism_inequality_l1055_105579


namespace square_side_length_l1055_105569

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 225 → side * side = area → side = 15 := by
  sorry

end square_side_length_l1055_105569


namespace square_rotation_overlap_area_l1055_105581

theorem square_rotation_overlap_area (β : Real) (h1 : 0 < β) (h2 : β < π/2) (h3 : Real.sin β = 3/5) :
  let side_length : Real := 2
  let overlap_area := 2 * (1/2 * side_length * (side_length * ((1 - Real.tan (β/2)) / (1 + Real.tan (β/2)))))
  overlap_area = 2 := by
sorry

end square_rotation_overlap_area_l1055_105581


namespace game_ends_in_three_rounds_l1055_105513

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- The state of the game at any point -/
structure GameState :=
  (tokens : Player → ℕ)

/-- Initial state of the game -/
def initial_state : GameState :=
  { tokens := λ p => match p with
    | Player.A => 12
    | Player.B => 11
    | Player.C => 10
    | Player.D => 9 }

/-- Determines if the game has ended -/
def game_ended (state : GameState) : Prop :=
  ∃ p, state.tokens p = 0

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry  -- Implementation details omitted

/-- The number of rounds played before the game ends -/
def rounds_played (state : GameState) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem stating that the game ends after exactly 3 rounds -/
theorem game_ends_in_three_rounds :
  rounds_played initial_state = 3 :=
sorry

end game_ends_in_three_rounds_l1055_105513
