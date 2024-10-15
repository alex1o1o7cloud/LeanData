import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_plot_length_l2364_236445

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 60 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 80 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l2364_236445


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l2364_236457

theorem parabola_fixed_point (u : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + u * x + 3 * u
  f (-3) = 45 := by
sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l2364_236457


namespace NUMINAMATH_CALUDE_kyle_lifts_320_l2364_236414

/-- Kyle's lifting capacity over the years -/
structure KylesLift where
  two_years_ago : ℝ
  last_year : ℝ
  this_year : ℝ

/-- Given information about Kyle's lifting capacity -/
def kyle_info (k : KylesLift) : Prop :=
  k.this_year = 1.6 * k.last_year ∧
  0.6 * k.last_year = 3 * k.two_years_ago ∧
  k.two_years_ago = 40

/-- Theorem: Kyle can lift 320 pounds this year -/
theorem kyle_lifts_320 (k : KylesLift) (h : kyle_info k) : k.this_year = 320 := by
  sorry


end NUMINAMATH_CALUDE_kyle_lifts_320_l2364_236414


namespace NUMINAMATH_CALUDE_shirt_discount_calculation_l2364_236434

/-- Given an original price and a discounted price, calculate the percentage discount -/
def calculate_discount (original_price discounted_price : ℚ) : ℚ :=
  (original_price - discounted_price) / original_price * 100

/-- Theorem: The percentage discount for a shirt with original price 933.33 and discounted price 560 is 40% -/
theorem shirt_discount_calculation :
  let original_price : ℚ := 933.33
  let discounted_price : ℚ := 560
  calculate_discount original_price discounted_price = 40 :=
by
  sorry

#eval calculate_discount 933.33 560

end NUMINAMATH_CALUDE_shirt_discount_calculation_l2364_236434


namespace NUMINAMATH_CALUDE_initial_jar_state_l2364_236416

/-- Represents the initial state of the jar of balls -/
structure JarState where
  totalBalls : ℕ
  blueBalls : ℕ
  nonBlueBalls : ℕ
  hTotalSum : totalBalls = blueBalls + nonBlueBalls

/-- Represents the state of the jar after removing some blue balls -/
structure UpdatedJarState where
  initialState : JarState
  removedBlueBalls : ℕ
  newBlueBalls : ℕ
  hNewBlue : newBlueBalls = initialState.blueBalls - removedBlueBalls
  newProbability : ℚ
  hProbability : newProbability = newBlueBalls / (initialState.totalBalls - removedBlueBalls)

/-- The main theorem stating the initial number of balls in the jar -/
theorem initial_jar_state 
  (updatedState : UpdatedJarState)
  (hInitialBlue : updatedState.initialState.blueBalls = 9)
  (hRemoved : updatedState.removedBlueBalls = 5)
  (hNewProb : updatedState.newProbability = 1/5) :
  updatedState.initialState.totalBalls = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_jar_state_l2364_236416


namespace NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l2364_236469

theorem sum_of_roots_product_polynomials :
  let p₁ : Polynomial ℝ := 3 * X^3 + 3 * X^2 - 9 * X + 27
  let p₂ : Polynomial ℝ := 4 * X^3 - 16 * X^2 + 5
  (p₁ * p₂).roots.sum = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l2364_236469


namespace NUMINAMATH_CALUDE_function_has_max_and_min_l2364_236474

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*((a + 2)*x + 1)

-- State the theorem
theorem function_has_max_and_min (a : ℝ) :
  (∃ x₁ x₂ : ℝ, ∀ x : ℝ, f a x₁ ≤ f a x ∧ f a x ≤ f a x₂) ↔ (a > 2 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_function_has_max_and_min_l2364_236474


namespace NUMINAMATH_CALUDE_prob_sum_ge_12_is_zero_l2364_236465

-- Define a uniform random variable on (0,1)
def uniform_01 : Type := {x : ℝ // 0 < x ∧ x < 1}

-- Define the sum of 5 such variables
def sum_5_uniform (X₁ X₂ X₃ X₄ X₅ : uniform_01) : ℝ :=
  X₁.val + X₂.val + X₃.val + X₄.val + X₅.val

-- State the theorem
theorem prob_sum_ge_12_is_zero :
  ∀ X₁ X₂ X₃ X₄ X₅ : uniform_01,
  sum_5_uniform X₁ X₂ X₃ X₄ X₅ < 12 :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_ge_12_is_zero_l2364_236465


namespace NUMINAMATH_CALUDE_find_a_l2364_236401

theorem find_a (A B : Set ℝ) (a : ℝ) :
  A = {-1, 1, 3} →
  B = {2, 2^a - 1} →
  A ∩ B = {1} →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2364_236401


namespace NUMINAMATH_CALUDE_cricket_average_l2364_236403

/-- Calculates the average score for the last 4 matches of a cricket series -/
theorem cricket_average (total_matches : ℕ) (first_matches : ℕ) (total_average : ℚ) (first_average : ℚ) :
  total_matches = 10 →
  first_matches = 6 →
  total_average = 389/10 →
  first_average = 42 →
  (total_matches * total_average - first_matches * first_average) / (total_matches - first_matches) = 137/4 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l2364_236403


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2364_236462

theorem partial_fraction_decomposition (M₁ M₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (50 * x - 42) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) → 
  M₁ * M₂ = -6264 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2364_236462


namespace NUMINAMATH_CALUDE_picnic_attendance_theorem_l2364_236471

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Theorem: Given the conditions of the picnic, the total number of attendees is 240 -/
theorem picnic_attendance_theorem (p : PicnicAttendance) 
  (h1 : p.men = p.women + 80)
  (h2 : p.adults = p.children + 80)
  (h3 : p.men = 120)
  : p.adults + p.children = 240 := by
  sorry

#check picnic_attendance_theorem

end NUMINAMATH_CALUDE_picnic_attendance_theorem_l2364_236471


namespace NUMINAMATH_CALUDE_cone_cut_ratio_sum_l2364_236427

/-- Represents a right circular cone --/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the result of cutting a cone parallel to its base --/
structure CutCone where
  originalCone : Cone
  cutRadius : ℝ

def surfaceAreaRatio (cc : CutCone) : ℝ := sorry

def volumeRatio (cc : CutCone) : ℝ := sorry

def isCoprime (m n : ℕ) : Prop := sorry

theorem cone_cut_ratio_sum (m n : ℕ) :
  let originalCone : Cone := { height := 6, baseRadius := 5 }
  let cc : CutCone := { originalCone := originalCone, cutRadius := 25/8 }
  surfaceAreaRatio cc = m / n →
  volumeRatio cc = m / n →
  isCoprime m n →
  m + n = 20 := by sorry

end NUMINAMATH_CALUDE_cone_cut_ratio_sum_l2364_236427


namespace NUMINAMATH_CALUDE_committee_formation_count_l2364_236402

/-- Represents a department in the division of science -/
inductive Department
| physics
| chemistry
| biology
| mathematics

/-- The number of departments in the division -/
def num_departments : Nat := 4

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors in the committee -/
def male_committee_members : Nat := 4

/-- The number of female professors in the committee -/
def female_committee_members : Nat := 4

/-- The number of departments contributing exactly two professors -/
def depts_with_two_profs : Nat := 2

/-- The number of departments contributing one male and one female professor -/
def depts_with_one_each : Nat := 2

/-- The number of ways to form the committee under the given conditions -/
def committee_formation_ways : Nat := 48114

theorem committee_formation_count :
  (num_departments = 4) →
  (male_professors_per_dept = 3) →
  (female_professors_per_dept = 3) →
  (committee_size = 8) →
  (male_committee_members = 4) →
  (female_committee_members = 4) →
  (depts_with_two_profs = 2) →
  (depts_with_one_each = 2) →
  (committee_formation_ways = 48114) := by
  sorry


end NUMINAMATH_CALUDE_committee_formation_count_l2364_236402


namespace NUMINAMATH_CALUDE_fifteen_mangoes_make_120_lassis_l2364_236491

/-- Given that 3 mangoes can make 24 lassis, this function calculates
    the number of lassis that can be made from a given number of mangoes. -/
def lassisFromMangoes (mangoes : ℕ) : ℕ :=
  (24 * mangoes) / 3

/-- Theorem stating that 15 mangoes will produce 120 lassis -/
theorem fifteen_mangoes_make_120_lassis :
  lassisFromMangoes 15 = 120 := by
  sorry

#eval lassisFromMangoes 15

end NUMINAMATH_CALUDE_fifteen_mangoes_make_120_lassis_l2364_236491


namespace NUMINAMATH_CALUDE_square_sum_ge_double_product_l2364_236419

theorem square_sum_ge_double_product :
  (∀ x y : ℝ, x^2 + y^2 ≥ 2*x*y) ↔ (x^2 + y^2 ≥ 2*x*y) := by sorry

end NUMINAMATH_CALUDE_square_sum_ge_double_product_l2364_236419


namespace NUMINAMATH_CALUDE_projection_equals_three_l2364_236459

/-- Given vectors a and b in ℝ², with a specific angle between them, 
    prove that the projection of b onto a is 3. -/
theorem projection_equals_three (a b : ℝ × ℝ) (angle : ℝ) : 
  a = (1, Real.sqrt 3) → 
  b = (3, Real.sqrt 3) → 
  angle = π / 6 → 
  (b.1 * a.1 + b.2 * a.2) / Real.sqrt (a.1^2 + a.2^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_projection_equals_three_l2364_236459


namespace NUMINAMATH_CALUDE_race_head_start_l2364_236411

theorem race_head_start (L : ℝ) (vA vB : ℝ) (h : vA = (17/15) * vB) :
  let d := (2/17) * L
  L / vA = (L - d) / vB := by sorry

end NUMINAMATH_CALUDE_race_head_start_l2364_236411


namespace NUMINAMATH_CALUDE_karen_has_128_crayons_l2364_236483

/-- The number of crayons in Judah's box -/
def judah_crayons : ℕ := 8

/-- The number of crayons in Gilbert's box -/
def gilbert_crayons : ℕ := 4 * judah_crayons

/-- The number of crayons in Beatrice's box -/
def beatrice_crayons : ℕ := 2 * gilbert_crayons

/-- The number of crayons in Karen's box -/
def karen_crayons : ℕ := 2 * beatrice_crayons

/-- Theorem stating that Karen's box contains 128 crayons -/
theorem karen_has_128_crayons : karen_crayons = 128 := by
  sorry

end NUMINAMATH_CALUDE_karen_has_128_crayons_l2364_236483


namespace NUMINAMATH_CALUDE_total_hotdogs_is_480_l2364_236432

/-- The number of hotdogs Helen's mother brought -/
def helen_hotdogs : ℕ := 101

/-- The number of hotdogs Dylan's mother brought -/
def dylan_hotdogs : ℕ := 379

/-- The total number of hotdogs -/
def total_hotdogs : ℕ := helen_hotdogs + dylan_hotdogs

/-- Theorem stating that the total number of hotdogs is 480 -/
theorem total_hotdogs_is_480 : total_hotdogs = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_is_480_l2364_236432


namespace NUMINAMATH_CALUDE_dinner_payment_difference_l2364_236444

/-- Calculates the difference in payment between John and Jane for a dinner --/
theorem dinner_payment_difference (original_price : ℝ) (discount_percent : ℝ) 
  (tip_percent : ℝ) (h1 : original_price = 40) (h2 : discount_percent = 0.1) 
  (h3 : tip_percent = 0.15) : 
  let discounted_price := original_price * (1 - discount_percent)
  let john_payment := discounted_price + original_price * tip_percent
  let jane_payment := discounted_price + discounted_price * tip_percent
  john_payment - jane_payment = 0.6 := by sorry

end NUMINAMATH_CALUDE_dinner_payment_difference_l2364_236444


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2364_236498

theorem imaginary_part_of_complex_expression : 
  let i : ℂ := Complex.I
  (((i^2016) / (2*i - 1)) * i).im = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2364_236498


namespace NUMINAMATH_CALUDE_circumradius_special_triangle_l2364_236418

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A
def angle_at_A (t : Triangle) : ℝ := sorry

-- Define the radius of the circumscribed circle
def circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem circumradius_special_triangle (t : Triangle) :
  angle_at_A t = π / 3 ∧
  distance t.B (incenter t) = 3 ∧
  distance t.C (incenter t) = 4 →
  circumradius t = Real.sqrt (37 / 3) := by
  sorry

end NUMINAMATH_CALUDE_circumradius_special_triangle_l2364_236418


namespace NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l2364_236479

def marcus_three_pointers : ℕ := 5
def marcus_two_pointers : ℕ := 10
def team_total_points : ℕ := 70

def marcus_points : ℕ := marcus_three_pointers * 3 + marcus_two_pointers * 2

theorem marcus_percentage_of_team_points :
  (marcus_points : ℚ) / team_total_points * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l2364_236479


namespace NUMINAMATH_CALUDE_upstream_speed_l2364_236455

/-- The speed of a man rowing upstream, given his speed in still water and the speed of the stream -/
theorem upstream_speed (downstream_speed still_water_speed stream_speed : ℝ) :
  downstream_speed = still_water_speed + stream_speed →
  still_water_speed > 0 →
  stream_speed > 0 →
  still_water_speed > stream_speed →
  (still_water_speed - stream_speed : ℝ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_upstream_speed_l2364_236455


namespace NUMINAMATH_CALUDE_percent_change_equality_l2364_236481

theorem percent_change_equality (x y : ℝ) (p : ℝ) 
  (h1 : x ≠ 0)
  (h2 : y = x * (1 + 0.15) * (1 - p / 100))
  (h3 : y = x) : 
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_percent_change_equality_l2364_236481


namespace NUMINAMATH_CALUDE_number_problem_l2364_236443

theorem number_problem : 
  ∃ x : ℚ, (x / 5 = 3 * (x / 6) - 40) ∧ (x = 400 / 3) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2364_236443


namespace NUMINAMATH_CALUDE_min_value_of_function_l2364_236425

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  x + 2 / (2 * x + 1) - 1 ≥ 1 / 2 ∧ 
  ∃ y > 0, y + 2 / (2 * y + 1) - 1 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2364_236425


namespace NUMINAMATH_CALUDE_band_photo_arrangement_min_band_members_l2364_236488

theorem band_photo_arrangement (n : ℕ) : n > 0 ∧ 9 ∣ n ∧ 10 ∣ n ∧ 11 ∣ n → n ≥ 990 := by
  sorry

theorem min_band_members : ∃ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 10 ∣ n ∧ 11 ∣ n ∧ n = 990 := by
  sorry

end NUMINAMATH_CALUDE_band_photo_arrangement_min_band_members_l2364_236488


namespace NUMINAMATH_CALUDE_divisibility_by_101_l2364_236482

theorem divisibility_by_101 (a b : ℕ) : 
  a < 10 → b < 10 → 
  (12 * 10^10 + a * 10^9 + b * 10^8 + 9876543) % 101 = 0 → 
  10 * a + b = 58 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l2364_236482


namespace NUMINAMATH_CALUDE_bird_families_remaining_l2364_236446

theorem bird_families_remaining (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 41 → flew_away = 27 → remaining = initial - flew_away → remaining = 14 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_remaining_l2364_236446


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l2364_236452

theorem compare_negative_fractions : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l2364_236452


namespace NUMINAMATH_CALUDE_two_primes_sum_and_product_l2364_236429

theorem two_primes_sum_and_product : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p * q = 166 ∧ p + q = 85 := by
sorry

end NUMINAMATH_CALUDE_two_primes_sum_and_product_l2364_236429


namespace NUMINAMATH_CALUDE_root_product_sum_l2364_236410

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2023 * x^3 - 4047 * x^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l2364_236410


namespace NUMINAMATH_CALUDE_correct_change_l2364_236421

/-- The price of a red candy in won -/
def red_candy_price : ℕ := 350

/-- The price of a blue candy in won -/
def blue_candy_price : ℕ := 180

/-- The number of red candies bought -/
def red_candy_count : ℕ := 3

/-- The number of blue candies bought -/
def blue_candy_count : ℕ := 2

/-- The amount Eunseo pays in won -/
def amount_paid : ℕ := 2000

/-- The change Eunseo should receive -/
def change : ℕ := amount_paid - (red_candy_price * red_candy_count + blue_candy_price * blue_candy_count)

theorem correct_change : change = 590 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l2364_236421


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_cosine_condition_l2364_236456

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- State the theorem
theorem triangle_isosceles_if_cosine_condition (t : Triangle) 
  (h : t.a * Real.cos t.B = t.b * Real.cos t.A) : 
  isIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_if_cosine_condition_l2364_236456


namespace NUMINAMATH_CALUDE_base4_calculation_l2364_236412

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 4 numbers --/
def mul_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a * base4_to_base10 b)

/-- Divides a base 4 number by another base 4 number --/
def div_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a / base4_to_base10 b)

theorem base4_calculation : 
  div_base4 (mul_base4 231 24) 3 = 1130 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l2364_236412


namespace NUMINAMATH_CALUDE_train_length_l2364_236436

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 80 * (5/18) → time = 9 → speed * time = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2364_236436


namespace NUMINAMATH_CALUDE_base_video_card_cost_l2364_236438

/-- Proves the cost of the base video card given the costs of other components --/
theorem base_video_card_cost 
  (computer_cost : ℝ)
  (peripheral_cost : ℝ)
  (upgraded_card_cost : ℝ → ℝ)
  (total_cost : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : peripheral_cost = computer_cost / 5)
  (h3 : ∀ x, upgraded_card_cost x = 2 * x)
  (h4 : total_cost = 2100)
  (h5 : ∃ x, computer_cost + peripheral_cost + upgraded_card_cost x = total_cost) :
  ∃ x, x = 150 ∧ computer_cost + peripheral_cost + upgraded_card_cost x = total_cost :=
by sorry

end NUMINAMATH_CALUDE_base_video_card_cost_l2364_236438


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l2364_236406

/-- The logarithmic function with base a > 0 and a ≠ 1 -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = 1 + log_a(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 1)

/-- Theorem: For any base a > 0 and a ≠ 1, f(x) passes through the point (2,1) -/
theorem fixed_point_of_f (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : f a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l2364_236406


namespace NUMINAMATH_CALUDE_min_tests_required_l2364_236448

/-- Represents a set of batteries -/
def Battery := Fin 8

/-- Represents a pair of batteries -/
def BatteryPair := Battery × Battery

/-- Represents the state of a battery -/
inductive BatteryState
| Good
| Bad

/-- The total number of batteries -/
def totalBatteries : Nat := 8

/-- The number of good batteries -/
def goodBatteries : Nat := 4

/-- The number of bad batteries -/
def badBatteries : Nat := 4

/-- A function that determines if a pair of batteries works -/
def works (pair : BatteryPair) (state : Battery → BatteryState) : Prop :=
  state pair.1 = BatteryState.Good ∧ state pair.2 = BatteryState.Good

/-- The main theorem stating the minimum number of tests required -/
theorem min_tests_required :
  ∀ (state : Battery → BatteryState),
  (∃ (good : Finset Battery), good.card = goodBatteries ∧ ∀ b ∈ good, state b = BatteryState.Good) →
  ∃ (tests : Finset BatteryPair),
    tests.card = 7 ∧
    (∀ (pair : BatteryPair), works pair state → pair ∈ tests) ∧
    ∀ (tests' : Finset BatteryPair),
      tests'.card < 7 →
      ∃ (pair : BatteryPair), works pair state ∧ pair ∉ tests' :=
sorry

end NUMINAMATH_CALUDE_min_tests_required_l2364_236448


namespace NUMINAMATH_CALUDE_reciprocal_plus_x_eq_three_implies_fraction_l2364_236492

theorem reciprocal_plus_x_eq_three_implies_fraction (x : ℝ) (h : 1/x + x = 3) :
  x^2 / (x^4 + x^2 + 1) = 1/8 := by sorry

end NUMINAMATH_CALUDE_reciprocal_plus_x_eq_three_implies_fraction_l2364_236492


namespace NUMINAMATH_CALUDE_candle_duration_first_scenario_l2364_236487

/-- The number of nights a candle lasts when burned for a given number of hours per night. -/
def candle_duration (hours_per_night : ℕ) : ℕ :=
  sorry

/-- The number of candles used over a given number of nights when burned for a given number of hours per night. -/
def candles_used (nights : ℕ) (hours_per_night : ℕ) : ℕ :=
  sorry

theorem candle_duration_first_scenario :
  let first_scenario_hours := 1
  let second_scenario_hours := 2
  let second_scenario_nights := 24
  let second_scenario_candles := 6
  candle_duration first_scenario_hours = 8 ∧
  candle_duration second_scenario_hours * second_scenario_candles = second_scenario_nights :=
by sorry

end NUMINAMATH_CALUDE_candle_duration_first_scenario_l2364_236487


namespace NUMINAMATH_CALUDE_randys_trip_length_l2364_236485

theorem randys_trip_length :
  ∀ (total_length : ℝ),
  (total_length / 2 : ℝ) + 30 + (total_length / 4 : ℝ) = total_length →
  total_length = 120 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2364_236485


namespace NUMINAMATH_CALUDE_calculation_proof_l2364_236496

theorem calculation_proof : 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + 0.40 + (0.5 : ℝ)^2 = 0.9666875 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2364_236496


namespace NUMINAMATH_CALUDE_postman_pete_mileage_l2364_236467

def pedometer_max : ℕ := 99999
def flips_in_year : ℕ := 50
def last_day_steps : ℕ := 25000
def steps_per_mile : ℕ := 1500

def total_steps : ℕ := flips_in_year * (pedometer_max + 1) + last_day_steps

def miles_walked : ℚ := total_steps / steps_per_mile

theorem postman_pete_mileage :
  ∃ (m : ℕ), m ≥ 3000 ∧ m ≤ 4000 ∧ 
  ∀ (n : ℕ), (n ≥ 3000 ∧ n ≤ 4000) → |miles_walked - m| ≤ |miles_walked - n| :=
sorry

end NUMINAMATH_CALUDE_postman_pete_mileage_l2364_236467


namespace NUMINAMATH_CALUDE_initial_tickets_count_l2364_236477

/-- The number of tickets sold in the first week -/
def first_week_sales : ℕ := 38

/-- The number of tickets sold in the second week -/
def second_week_sales : ℕ := 17

/-- The number of tickets left to sell -/
def remaining_tickets : ℕ := 35

/-- The initial number of tickets -/
def initial_tickets : ℕ := first_week_sales + second_week_sales + remaining_tickets

theorem initial_tickets_count : initial_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_tickets_count_l2364_236477


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2364_236424

/-- The measure of each interior angle in a regular octagon -/
theorem regular_octagon_interior_angle : ℝ := by
  -- Define the number of sides in an octagon
  let n : ℕ := 8

  -- Define the formula for the sum of interior angles of an n-sided polygon
  let sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

  -- Define the measure of each interior angle in a regular n-gon
  let interior_angle (n : ℕ) : ℝ := sum_interior_angles n / n

  -- Prove that the measure of each interior angle in a regular octagon is 135°
  have h : interior_angle n = 135 := by sorry

  -- Return the result
  exact 135


end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2364_236424


namespace NUMINAMATH_CALUDE_trigonometric_equation_proof_l2364_236440

theorem trigonometric_equation_proof : 
  4.74 * (Real.cos (64 * π / 180) * Real.cos (4 * π / 180) - 
          Real.cos (86 * π / 180) * Real.cos (26 * π / 180)) / 
         (Real.cos (71 * π / 180) * Real.cos (41 * π / 180) - 
          Real.cos (49 * π / 180) * Real.cos (19 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_proof_l2364_236440


namespace NUMINAMATH_CALUDE_absent_children_absent_children_solution_l2364_236486

theorem absent_children (total_children : ℕ) (initial_bananas_per_child : ℕ) (extra_bananas : ℕ) : ℕ :=
  let total_bananas := total_children * initial_bananas_per_child
  let final_bananas_per_child := initial_bananas_per_child + extra_bananas
  let absent_children := total_children - (total_bananas / final_bananas_per_child)
  absent_children

theorem absent_children_solution :
  absent_children 320 2 2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_absent_children_absent_children_solution_l2364_236486


namespace NUMINAMATH_CALUDE_point_coordinates_l2364_236468

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates (P : Point) 
  (h1 : SecondQuadrant P) 
  (h2 : DistanceToXAxis P = 4) 
  (h3 : DistanceToYAxis P = 3) : 
  P.x = -3 ∧ P.y = 4 := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l2364_236468


namespace NUMINAMATH_CALUDE_interest_rate_middle_period_l2364_236435

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_middle_period 
  (principal : ℝ) 
  (rate1 : ℝ) 
  (rate3 : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (time3 : ℝ) 
  (total_interest : ℝ) :
  principal = 8000 →
  rate1 = 0.08 →
  rate3 = 0.12 →
  time1 = 4 →
  time2 = 6 →
  time3 = 5 →
  total_interest = 12160 →
  ∃ (rate2 : ℝ), 
    rate2 = 0.1 ∧
    total_interest = simple_interest principal rate1 time1 + 
                     simple_interest principal rate2 time2 + 
                     simple_interest principal rate3 time3 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_middle_period_l2364_236435


namespace NUMINAMATH_CALUDE_inequality_solutions_l2364_236484

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -3/2 < x ∧ x < 3}
def solution_set2 : Set ℝ := {x | -5 < x ∧ x ≤ 3/2}
def solution_set3 : Set ℝ := ∅
def solution_set4 : Set ℝ := Set.univ

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -2*x^2 + 3*x + 9 > 0
def inequality2 (x : ℝ) : Prop := (8 - x) / (5 + x) > 1
def inequality3 (x : ℝ) : Prop := -x^2 + 2*x - 3 > 0
def inequality4 (x : ℝ) : Prop := x^2 - 14*x + 50 > 0

-- Theorem statements
theorem inequality_solutions :
  (∀ x, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x, x ∈ solution_set2 ↔ inequality2 x) ∧
  (∀ x, x ∈ solution_set3 ↔ inequality3 x) ∧
  (∀ x, x ∈ solution_set4 ↔ inequality4 x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2364_236484


namespace NUMINAMATH_CALUDE_function_properties_l2364_236428

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on an interval [a, b] if
    for all x, y in [a, b], x ≤ y implies f(x) ≤ f(y) -/
def MonoIncOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- Main theorem -/
theorem function_properties (f : ℝ → ℝ) 
    (heven : IsEven f)
    (hmono : MonoIncOn f (-1) 0)
    (hcond : ∀ x, f (1 - x) + f (1 + x) = 0) :
    (f (-3) = 0) ∧
    (MonoIncOn f 1 2) ∧
    (∀ x, f x = f (2 - x)) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l2364_236428


namespace NUMINAMATH_CALUDE_triangle_angle_120_degrees_l2364_236495

theorem triangle_angle_120_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 - b^2 = 3*b*c)
  (h3 : Real.sin C = 2 * Real.sin B)
  (h4 : A + B + C = π)
  (h5 : a / Real.sin A = b / Real.sin B)
  (h6 : b / Real.sin B = c / Real.sin C)
  : A = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_120_degrees_l2364_236495


namespace NUMINAMATH_CALUDE_intersection_M_N_l2364_236478

def M : Set ℝ := {x | x^2 ≤ 1}
def N : Set ℝ := {-2, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2364_236478


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l2364_236422

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  1/7 < a ∧ a < 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l2364_236422


namespace NUMINAMATH_CALUDE_wendy_sweaters_l2364_236480

/-- Represents the number of pieces of clothing a washing machine can wash in one load. -/
def machine_capacity : ℕ := 8

/-- Represents the number of shirts Wendy has to wash. -/
def num_shirts : ℕ := 39

/-- Represents the total number of loads Wendy has to do. -/
def total_loads : ℕ := 9

/-- Calculates the number of sweaters Wendy has to wash. -/
def num_sweaters : ℕ := (machine_capacity * total_loads) - num_shirts

theorem wendy_sweaters : num_sweaters = 33 := by
  sorry

end NUMINAMATH_CALUDE_wendy_sweaters_l2364_236480


namespace NUMINAMATH_CALUDE_typeC_migration_time_l2364_236463

/-- Represents the lakes in the migration sequence -/
inductive Lake : Type
| Jim : Lake
| Disney : Lake
| London : Lake
| Everest : Lake

/-- Defines the distance between two lakes -/
def distance (a b : Lake) : ℝ :=
  match a, b with
  | Lake.Jim, Lake.Disney => 42
  | Lake.Disney, Lake.London => 57
  | Lake.London, Lake.Everest => 65
  | Lake.Everest, Lake.Jim => 70
  | _, _ => 0  -- For all other combinations

/-- Calculates the total distance of one complete sequence -/
def totalDistance : ℝ :=
  distance Lake.Jim Lake.Disney +
  distance Lake.Disney Lake.London +
  distance Lake.London Lake.Everest +
  distance Lake.Everest Lake.Jim

/-- The average speed of Type C birds in miles per hour -/
def typeCSpeed : ℝ := 12

/-- Theorem: Type C birds take 39 hours to complete two full sequences -/
theorem typeC_migration_time :
  2 * (totalDistance / typeCSpeed) = 39 := by sorry


end NUMINAMATH_CALUDE_typeC_migration_time_l2364_236463


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2364_236475

theorem fraction_equivalence : (9 : ℚ) / (7 * 53) = 0.9 / (0.7 * 53) := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2364_236475


namespace NUMINAMATH_CALUDE_original_savings_calculation_l2364_236473

theorem original_savings_calculation (savings : ℝ) : 
  (4 / 5 : ℝ) * savings + 100 = savings → savings = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l2364_236473


namespace NUMINAMATH_CALUDE_products_from_equipment_B_l2364_236450

/-- Given a total number of products and a stratified sample, 
    calculate the number of products produced by equipment B -/
theorem products_from_equipment_B 
  (total : ℕ) 
  (sample_size : ℕ) 
  (sample_A : ℕ) 
  (h1 : total = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_A = 50) : 
  total - (total * sample_A / sample_size) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_products_from_equipment_B_l2364_236450


namespace NUMINAMATH_CALUDE_olympic_high_school_quiz_l2364_236464

theorem olympic_high_school_quiz (f s : ℚ) 
  (h1 : f > 0) 
  (h2 : s > 0) 
  (h3 : (3/7) * f = (5/7) * s) : 
  f = (5/3) * s :=
sorry

end NUMINAMATH_CALUDE_olympic_high_school_quiz_l2364_236464


namespace NUMINAMATH_CALUDE_expression_evaluation_l2364_236454

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  ((x - 2*y)*x - (x - 2*y)*(x + 2*y)) / y = -8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2364_236454


namespace NUMINAMATH_CALUDE_perfect_line_fit_l2364_236449

/-- A sample point in a 2D plane -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : SamplePoint) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Calculate the sum of squared residuals -/
def sumSquaredResiduals (points : List SamplePoint) (l : Line) : ℝ :=
  (points.map (fun p => (p.y - (l.slope * p.x + l.intercept))^2)).sum

/-- Calculate the correlation coefficient -/
def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry  -- Actual calculation of correlation coefficient

/-- Theorem: If all sample points fall on a straight line, 
    then the sum of squared residuals is 0 and 
    the absolute value of the correlation coefficient is 1 -/
theorem perfect_line_fit (points : List SamplePoint) (l : Line) :
  (∀ p ∈ points, pointOnLine p l) →
  sumSquaredResiduals points l = 0 ∧ |correlationCoefficient points| = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_line_fit_l2364_236449


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l2364_236405

/-- Given a line described by the dot product equation (-1, 2) · ((x, y) - (3, -4)) = 0,
    prove that it is equivalent to the line y = (1/2)x - 11/2 -/
theorem line_equation_equivalence (x y : ℝ) :
  (-1 : ℝ) * (x - 3) + 2 * (y - (-4)) = 0 ↔ y = (1/2 : ℝ) * x - 11/2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l2364_236405


namespace NUMINAMATH_CALUDE_odd_function_value_l2364_236472

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem odd_function_value (a b c : ℝ) :
  (∀ x, f a b c (-x) = -(f a b c x)) →
  (∀ x ∈ Set.Icc (2*b - 5) (2*b - 3), f a b c x ∈ Set.range (f a b c)) →
  f a b c (1/2) = 9/8 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l2364_236472


namespace NUMINAMATH_CALUDE_puzzle_solutions_l2364_236494

-- Define a structure for the puzzle solution
structure PuzzleSolution where
  a : Nat
  b : Nat
  v : Nat
  h : a * 10 + b = b ^ v ∧ a ≠ b ∧ a ≠ v ∧ b ≠ v ∧ a > 0 ∧ b > 0 ∧ v > 0 ∧ a < 10 ∧ b < 10 ∧ v < 10

-- Define the theorem
theorem puzzle_solutions :
  {s : PuzzleSolution | True} =
  {⟨3, 2, 5, sorry⟩, ⟨3, 6, 2, sorry⟩, ⟨6, 4, 3, sorry⟩} := by sorry

end NUMINAMATH_CALUDE_puzzle_solutions_l2364_236494


namespace NUMINAMATH_CALUDE_set_equality_l2364_236441

theorem set_equality : Finset.toSet {1, 2, 3, 4, 5} = Finset.toSet {5, 4, 3, 2, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2364_236441


namespace NUMINAMATH_CALUDE_floor_difference_inequality_l2364_236423

theorem floor_difference_inequality (x y : ℝ) : 
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by sorry

end NUMINAMATH_CALUDE_floor_difference_inequality_l2364_236423


namespace NUMINAMATH_CALUDE_right_triangle_check_l2364_236453

theorem right_triangle_check (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_check_l2364_236453


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2364_236407

theorem smallest_k_no_real_roots : 
  ∃ k : ℕ, k = 4 ∧ 
  (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 7 ≠ 0) ∧
  (∀ m : ℕ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2364_236407


namespace NUMINAMATH_CALUDE_two_pairs_probability_l2364_236476

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The probability of rolling exactly two pairs of dice showing the same value,
    with the other two dice each showing different numbers that don't match the paired numbers,
    when rolling six standard six-sided dice once -/
def probabilityTwoPairs : ℚ :=
  25 / 72

theorem two_pairs_probability :
  probabilityTwoPairs = (
    (numFaces.choose 2) *
    (numDice.choose 2) *
    ((numDice - 2).choose 2) *
    (numFaces - 2) *
    (numFaces - 3)
  ) / (numFaces ^ numDice) :=
sorry

end NUMINAMATH_CALUDE_two_pairs_probability_l2364_236476


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l2364_236497

/-- Profit function for the bookstore --/
def profit (p : ℝ) : ℝ := (p - 2) * (110 - 4 * p)

/-- The optimal price is 15 --/
def optimal_price : ℕ := 15

theorem profit_maximized_at_optimal_price :
  ∀ p : ℕ, p ≤ 22 → profit p ≤ profit optimal_price :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l2364_236497


namespace NUMINAMATH_CALUDE_min_value_inequality_l2364_236431

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2364_236431


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l2364_236400

theorem quadratic_complete_square (a b c : ℝ) (h : a = 4 ∧ b = -16 ∧ c = -200) :
  ∃ q t : ℝ, (∀ x, a * x^2 + b * x + c = 0 ↔ (x + q)^2 = t) ∧ t = 54 :=
sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l2364_236400


namespace NUMINAMATH_CALUDE_mixed_number_sum_l2364_236466

theorem mixed_number_sum : (2 + 1/10) + (3 + 11/100) = 5.21 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_sum_l2364_236466


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2364_236404

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2022
  let y : ℝ := -Real.sqrt 2
  4 * x * y + (2 * x - y) * (2 * x + y) - (2 * x + y)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2364_236404


namespace NUMINAMATH_CALUDE_min_value_theorem_l2364_236439

theorem min_value_theorem (a : ℝ) (ha : a > 0) :
  (∃ (x_min : ℝ), x_min > 0 ∧
    ∀ (x : ℝ), x > 0 → (a^2 + x^2) / x ≥ (a^2 + x_min^2) / x_min) ∧
  (∀ (x : ℝ), x > 0 → (a^2 + x^2) / x ≥ 2*a) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2364_236439


namespace NUMINAMATH_CALUDE_flower_shop_problem_l2364_236409

/-- The number of flowers brought at dawn -/
def flowers_at_dawn : ℕ := 300

/-- The fraction of flowers sold in the morning -/
def morning_sale_fraction : ℚ := 3/5

/-- The total number of flowers sold in the afternoon -/
def afternoon_sales : ℕ := 180

theorem flower_shop_problem :
  (flowers_at_dawn : ℚ) * morning_sale_fraction = afternoon_sales ∧
  (flowers_at_dawn : ℚ) * morning_sale_fraction = (flowers_at_dawn : ℚ) * (1 - morning_sale_fraction) + (afternoon_sales - (flowers_at_dawn : ℚ) * (1 - morning_sale_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_flower_shop_problem_l2364_236409


namespace NUMINAMATH_CALUDE_fourth_root_of_2560000_l2364_236470

theorem fourth_root_of_2560000 : (2560000 : ℝ) ^ (1/4 : ℝ) = 40 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_2560000_l2364_236470


namespace NUMINAMATH_CALUDE_abs_equation_sufficient_not_necessary_l2364_236433

/-- The distance from a point to the x-axis --/
def dist_to_x_axis (y : ℝ) : ℝ := |y|

/-- The distance from a point to the y-axis --/
def dist_to_y_axis (x : ℝ) : ℝ := |x|

/-- The condition that distances to both axes are equal --/
def equal_dist_to_axes (x y : ℝ) : Prop :=
  dist_to_x_axis y = dist_to_y_axis x

/-- The equation y = |x| --/
def abs_equation (x y : ℝ) : Prop := y = |x|

/-- Theorem stating that y = |x| is a sufficient but not necessary condition --/
theorem abs_equation_sufficient_not_necessary :
  (∀ x y : ℝ, abs_equation x y → equal_dist_to_axes x y) ∧
  ¬(∀ x y : ℝ, equal_dist_to_axes x y → abs_equation x y) :=
sorry

end NUMINAMATH_CALUDE_abs_equation_sufficient_not_necessary_l2364_236433


namespace NUMINAMATH_CALUDE_height_comparison_l2364_236437

theorem height_comparison (a b : ℝ) (h : a = 0.6 * b) :
  (b - a) / a * 100 = 200 / 3 :=
sorry

end NUMINAMATH_CALUDE_height_comparison_l2364_236437


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2364_236430

/-- A geometric sequence with its third term and sum of first three terms given -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  is_geometric : ∀ n, a (n + 1) = q * a n
  third_term : a 3 = 3/2
  third_sum : (a 1) + (a 2) + (a 3) = 9/2

/-- The common ratio of the geometric sequence is either 1 or -1/2 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  seq.q = 1 ∨ seq.q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2364_236430


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2364_236442

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → q = 2 → a 3 = 3 → a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2364_236442


namespace NUMINAMATH_CALUDE_cubic_complex_equation_l2364_236447

theorem cubic_complex_equation (a b c : ℕ+) :
  c = Complex.I.re * ((a + Complex.I * b) ^ 3 - 107 * Complex.I) →
  c = 198 := by
sorry

end NUMINAMATH_CALUDE_cubic_complex_equation_l2364_236447


namespace NUMINAMATH_CALUDE_petya_running_time_l2364_236451

theorem petya_running_time (V D : ℝ) (h1 : V > 0) (h2 : D > 0) : 
  (D / (1.25 * V) / 2) + (D / (0.8 * V) / 2) > D / V := by
  sorry

end NUMINAMATH_CALUDE_petya_running_time_l2364_236451


namespace NUMINAMATH_CALUDE_fraction_addition_l2364_236426

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2364_236426


namespace NUMINAMATH_CALUDE_prob_theorem_l2364_236499

/-- Represents the colors of the balls in the bag -/
inductive Color
  | Black
  | Yellow
  | Green

/-- The total number of balls in the bag -/
def total_balls : ℕ := 9

/-- The probability of drawing either a black or a yellow ball -/
def prob_black_or_yellow : ℚ := 5/9

/-- The probability of drawing either a yellow or a green ball -/
def prob_yellow_or_green : ℚ := 2/3

/-- The probability of drawing a ball of a specific color -/
def prob_color (c : Color) : ℚ :=
  match c with
  | Color.Black => 1/3
  | Color.Yellow => 2/9
  | Color.Green => 4/9

/-- The probability of drawing two balls of different colors -/
def prob_different_colors : ℚ := 13/18

/-- Theorem stating the probabilities are correct given the conditions -/
theorem prob_theorem :
  (∀ c, prob_color c ≥ 0) ∧
  (prob_color Color.Black + prob_color Color.Yellow + prob_color Color.Green = 1) ∧
  (prob_color Color.Black + prob_color Color.Yellow = prob_black_or_yellow) ∧
  (prob_color Color.Yellow + prob_color Color.Green = prob_yellow_or_green) ∧
  (prob_different_colors = 13/18) :=
  sorry

end NUMINAMATH_CALUDE_prob_theorem_l2364_236499


namespace NUMINAMATH_CALUDE_equality_of_exponents_l2364_236420

-- Define variables
variable (a b c d : ℝ)
variable (x y q z : ℝ)

-- State the theorem
theorem equality_of_exponents 
  (h1 : a^x = c^q) 
  (h2 : a^x = b) 
  (h3 : c^y = a^z) 
  (h4 : c^y = d) 
  : x * y = q * z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_exponents_l2364_236420


namespace NUMINAMATH_CALUDE_tim_income_percentage_l2364_236493

theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 1.44 * juan) : 
  tim = 0.9 * juan := by
  sorry

end NUMINAMATH_CALUDE_tim_income_percentage_l2364_236493


namespace NUMINAMATH_CALUDE_four_block_selection_count_l2364_236460

def grid_size : ℕ := 6
def blocks_to_select : ℕ := 4

theorem four_block_selection_count :
  (Nat.choose grid_size blocks_to_select) *
  (Nat.choose grid_size blocks_to_select) *
  (Nat.factorial blocks_to_select) = 5400 := by
  sorry

end NUMINAMATH_CALUDE_four_block_selection_count_l2364_236460


namespace NUMINAMATH_CALUDE_point_C_representation_l2364_236408

def point_A : ℝ := -2

def point_B : ℝ := point_A - 2

def distance_BC : ℝ := 5

theorem point_C_representation :
  ∃ (point_C : ℝ), (point_C = point_B - distance_BC ∨ point_C = point_B + distance_BC) ∧
                    (point_C = -9 ∨ point_C = 1) := by
  sorry

end NUMINAMATH_CALUDE_point_C_representation_l2364_236408


namespace NUMINAMATH_CALUDE_even_product_implies_even_factor_l2364_236413

theorem even_product_implies_even_factor (a b : ℕ) : 
  a > 0 → b > 0 → Even (a * b) → Even a ∨ Even b :=
by sorry

end NUMINAMATH_CALUDE_even_product_implies_even_factor_l2364_236413


namespace NUMINAMATH_CALUDE_six_people_arrangement_l2364_236458

/-- The number of ways to arrange n people in a line with two specific people always next to each other -/
def arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- Theorem: For 6 people with two specific people always next to each other, there are 240 possible arrangements -/
theorem six_people_arrangement : arrangements 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l2364_236458


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2364_236490

/-- Given an item with a cost price, this theorem proves that if the item is priced at 1.5 times
    its cost price and sold with a 40% profit after a 20 yuan discount, then the cost price
    of the item is 200 yuan. -/
theorem cost_price_calculation (cost_price : ℝ) : 
  (1.5 * cost_price - 20 - cost_price = 0.4 * cost_price) → cost_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2364_236490


namespace NUMINAMATH_CALUDE_fraction_inequality_l2364_236461

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c) > e / (b - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2364_236461


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l2364_236489

/-- Given a paint mixture with a ratio of red:yellow:white as 5:3:7,
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) :
  red / white = 5 / 7 →
  yellow / white = 3 / 7 →
  white = 21 →
  red = 15 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l2364_236489


namespace NUMINAMATH_CALUDE_inequality_proof_l2364_236417

theorem inequality_proof (a A b B : ℝ) 
  (h1 : |A - 3*a| ≤ 1 - a)
  (h2 : |B - 3*b| ≤ 1 - b)
  (ha : a > 0)
  (hb : b > 0) :
  |A*B/3 - 3*a*b| - 3*a*b ≤ 1 - a*b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2364_236417


namespace NUMINAMATH_CALUDE_jessica_found_41_seashells_l2364_236415

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 18

/-- The total number of seashells Mary and Jessica found together -/
def total_seashells : ℕ := 59

/-- The number of seashells Jessica found -/
def jessica_seashells : ℕ := total_seashells - mary_seashells

theorem jessica_found_41_seashells : jessica_seashells = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessica_found_41_seashells_l2364_236415
