import Mathlib

namespace NUMINAMATH_CALUDE_highest_score_l3414_341458

theorem highest_score (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (bc_gt_ad : b + c > a + d)
  (a_gt_bd : a > b + d) :
  c > a ∧ c > b ∧ c > d :=
sorry

end NUMINAMATH_CALUDE_highest_score_l3414_341458


namespace NUMINAMATH_CALUDE_fifteen_equidistant_planes_spheres_l3414_341452

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of 5 points in 3D space -/
def FivePoints := Fin 5 → Point3D

/-- Predicate to check if 5 points lie on the same plane -/
def lieOnSamePlane (points : FivePoints) : Prop := sorry

/-- Predicate to check if 5 points lie on the same sphere -/
def lieOnSameSphere (points : FivePoints) : Prop := sorry

/-- Count of equidistant planes or spheres from 5 points -/
def countEquidistantPlanesSpheres (points : FivePoints) : ℕ := sorry

/-- Theorem stating that there are exactly 15 equidistant planes or spheres -/
theorem fifteen_equidistant_planes_spheres (points : FivePoints) 
  (h1 : ¬ lieOnSamePlane points) (h2 : ¬ lieOnSameSphere points) :
  countEquidistantPlanesSpheres points = 15 := by sorry

end NUMINAMATH_CALUDE_fifteen_equidistant_planes_spheres_l3414_341452


namespace NUMINAMATH_CALUDE_probability_above_parabola_l3414_341427

/-- A single-digit positive integer -/
def SingleDigit := { n : ℕ | 1 ≤ n ∧ n ≤ 9 }

/-- The total number of possible (a, b) combinations -/
def TotalCombinations : ℕ := 81

/-- The number of valid (a, b) combinations where (a, b) lies above y = ax^2 + bx -/
def ValidCombinations : ℕ := 72

/-- The probability that a randomly chosen point (a, b) lies above y = ax^2 + bx -/
def ProbabilityAboveParabola : ℚ := ValidCombinations / TotalCombinations

theorem probability_above_parabola :
  ProbabilityAboveParabola = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l3414_341427


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3414_341400

theorem quadratic_equation_properties (m : ℝ) :
  m < 4 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + m = 0 ∧ x₂^2 - 4*x₂ + m = 0) ∧
  ((-1)^2 - 4*(-1) + m = 0 → m = -5 ∧ 5^2 - 4*5 + m = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3414_341400


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3414_341488

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  855 * (π / 180) = 59 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3414_341488


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3414_341414

theorem smallest_divisible_by_1_to_12 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3414_341414


namespace NUMINAMATH_CALUDE_expand_polynomial_l3414_341496

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3414_341496


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3414_341459

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3414_341459


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l3414_341411

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_eight : a * b * c = 8) : 
  1/a + 1/b + 1/c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l3414_341411


namespace NUMINAMATH_CALUDE_bakery_earnings_l3414_341428

/-- Represents the daily production and prices of baked goods in a bakery --/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ

/-- Calculates the total earnings for a given number of days --/
def total_earnings (data : BakeryData) (days : ℕ) : ℝ :=
  (data.cupcake_price * data.daily_cupcakes +
   data.cookie_price * data.daily_cookies +
   data.biscuit_price * data.daily_biscuits) * days

/-- Theorem stating that the total earnings for 5 days is $350 --/
theorem bakery_earnings (data : BakeryData) 
  (h1 : data.cupcake_price = 1.5)
  (h2 : data.cookie_price = 2)
  (h3 : data.biscuit_price = 1)
  (h4 : data.daily_cupcakes = 20)
  (h5 : data.daily_cookies = 10)
  (h6 : data.daily_biscuits = 20) :
  total_earnings data 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_bakery_earnings_l3414_341428


namespace NUMINAMATH_CALUDE_percentage_difference_l3414_341486

theorem percentage_difference (x y : ℝ) (h1 : y = 125 * (1 + 0.1)) (h2 : x = 123.75) :
  (y - x) / y * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3414_341486


namespace NUMINAMATH_CALUDE_pie_division_l3414_341466

theorem pie_division (total_pie : ℚ) (people : ℕ) :
  total_pie = 5 / 8 ∧ people = 4 →
  total_pie / people = 5 / 32 := by
sorry

end NUMINAMATH_CALUDE_pie_division_l3414_341466


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l3414_341449

theorem isabel_piggy_bank (X : ℝ) : 
  (X > 0) → 
  ((1 - 0.25) * (1 / 2) * (2 / 3) * X = 60) → 
  (X = 720) := by
sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l3414_341449


namespace NUMINAMATH_CALUDE_sixteen_four_eight_calculation_l3414_341478

theorem sixteen_four_eight_calculation : (16^2 / 4^3) * 8^3 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_four_eight_calculation_l3414_341478


namespace NUMINAMATH_CALUDE_equation_root_range_l3414_341404

theorem equation_root_range (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ 
   Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) = k + 1) 
  → k ∈ Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_range_l3414_341404


namespace NUMINAMATH_CALUDE_one_less_than_negative_two_l3414_341445

theorem one_less_than_negative_two : -2 - 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_negative_two_l3414_341445


namespace NUMINAMATH_CALUDE_expression_evaluation_l3414_341417

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3414_341417


namespace NUMINAMATH_CALUDE_parabola_line_intersection_length_no_isosceles_right_triangle_on_parabola_l3414_341402

/-- Given a parabola y^2 = 2px where p > 0, and a line y = k(x - p/2) intersecting 
    the parabola at points A and B, the length of AB is (2p(k^2 + 1)) / k^2 -/
theorem parabola_line_intersection_length (p k : ℝ) (hp : p > 0) :
  let f : ℝ → ℝ := λ k => (2 * p * (k^2 + 1)) / k^2
  let parabola : ℝ × ℝ → Prop := λ (x, y) => y^2 = 2 * p * x
  let line : ℝ → ℝ := λ x => k * (x - p / 2)
  let A := (x₁, line x₁)
  let B := (x₂, line x₂)
  parabola A ∧ parabola B → abs (x₂ - x₁) = f k :=
by sorry

/-- There does not exist a point C on the parabola y^2 = 2px such that 
    triangle ABC is an isosceles right triangle with C as the vertex of the right angle -/
theorem no_isosceles_right_triangle_on_parabola (p : ℝ) (hp : p > 0) :
  let parabola : ℝ × ℝ → Prop := λ (x, y) => y^2 = 2 * p * x
  ¬ ∃ (A B C : ℝ × ℝ), parabola A ∧ parabola B ∧ parabola C ∧
    (C.1 < (A.1 + B.1) / 2) ∧
    (abs (A.1 - C.1) = abs (B.1 - C.1)) ∧
    ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_length_no_isosceles_right_triangle_on_parabola_l3414_341402


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3414_341418

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 11) (h2 : Nat.lcm a b = 181) :
  a * b = 1991 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3414_341418


namespace NUMINAMATH_CALUDE_inequality_proof_l3414_341422

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3414_341422


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3414_341473

-- Define the lines l₁ and l₂
def l₁ (x y m : ℝ) : Prop := x + (1 + m) * y + (m - 2) = 0
def l₂ (x y m : ℝ) : Prop := m * x + 2 * y + 8 = 0

-- Define the parallel condition
def parallel (m : ℝ) : Prop := ∀ x y, l₁ x y m → l₂ x y m

-- Theorem statement
theorem parallel_lines_m_value (m : ℝ) : parallel m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3414_341473


namespace NUMINAMATH_CALUDE_diamond_composition_l3414_341487

/-- Define the diamond operation -/
def diamond (k : ℝ) (x y : ℝ) : ℝ := x^2 - k*y

/-- Theorem stating the result of h ◇ (h ◇ h) -/
theorem diamond_composition (h : ℝ) : diamond 3 h (diamond 3 h h) = -2*h^2 + 9*h := by
  sorry

end NUMINAMATH_CALUDE_diamond_composition_l3414_341487


namespace NUMINAMATH_CALUDE_salt_bag_weight_l3414_341446

/-- Given a bag of sugar weighing 16 kg and the fact that removing 4 kg from the combined
    weight of sugar and salt bags results in 42 kg, prove that the salt bag weighs 30 kg. -/
theorem salt_bag_weight (sugar_weight : ℕ) (combined_minus_four : ℕ) :
  sugar_weight = 16 ∧ combined_minus_four = 42 →
  ∃ (salt_weight : ℕ), salt_weight = 30 ∧ sugar_weight + salt_weight = combined_minus_four + 4 :=
by sorry

end NUMINAMATH_CALUDE_salt_bag_weight_l3414_341446


namespace NUMINAMATH_CALUDE_equal_roots_implies_c_equals_one_fourth_l3414_341439

-- Define the quadratic equation
def quadratic_equation (x c : ℝ) : Prop := x^2 + x + c = 0

-- Define the condition for two equal real roots
def has_two_equal_real_roots (c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation x c ∧ 
    ∀ y : ℝ, quadratic_equation y c → y = x

-- Theorem statement
theorem equal_roots_implies_c_equals_one_fourth :
  ∀ c : ℝ, has_two_equal_real_roots c → c = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_implies_c_equals_one_fourth_l3414_341439


namespace NUMINAMATH_CALUDE_max_terms_sum_to_target_l3414_341457

/-- The sequence of odd numbers from 1 to 101 -/
def oddSequence : List Nat := List.range 51 |>.map (fun n => 2*n + 1)

/-- The sum we're aiming for -/
def targetSum : Nat := 1949

/-- The maximum number of terms that sum to the target -/
def maxTerms : Nat := 44

theorem max_terms_sum_to_target :
  ∃ (subset : List Nat),
    subset.toFinset ⊆ oddSequence.toFinset ∧
    subset.sum = targetSum ∧
    subset.length = maxTerms ∧
    ∀ (otherSubset : List Nat),
      otherSubset.toFinset ⊆ oddSequence.toFinset →
      otherSubset.sum = targetSum →
      otherSubset.length ≤ maxTerms :=
by sorry

end NUMINAMATH_CALUDE_max_terms_sum_to_target_l3414_341457


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l3414_341468

/-- A triangle with integer side lengths satisfying specific conditions -/
structure SpecialTriangle where
  x : ℕ
  y : ℕ
  side_product : x * y = 105
  triangle_inequality : x + y > 13 ∧ x + 13 > y ∧ y + 13 > x

/-- The perimeter of the special triangle is 35 -/
theorem special_triangle_perimeter (t : SpecialTriangle) : 13 + t.x + t.y = 35 := by
  sorry

#check special_triangle_perimeter

end NUMINAMATH_CALUDE_special_triangle_perimeter_l3414_341468


namespace NUMINAMATH_CALUDE_grade_11_count_l3414_341429

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  sample_size : ℕ
  grade_10_sample : ℕ
  grade_12_sample : ℕ

/-- Calculates the number of Grade 11 students in the school -/
def grade_11_students (s : School) : ℕ :=
  ((s.sample_size - s.grade_10_sample - s.grade_12_sample) * s.total_students) / s.sample_size

/-- Theorem stating the number of Grade 11 students in the given school -/
theorem grade_11_count (s : School) 
  (h1 : s.total_students = 900)
  (h2 : s.sample_size = 45)
  (h3 : s.grade_10_sample = 20)
  (h4 : s.grade_12_sample = 10) :
  grade_11_students s = 300 := by
  sorry

#eval grade_11_students ⟨900, 45, 20, 10⟩

end NUMINAMATH_CALUDE_grade_11_count_l3414_341429


namespace NUMINAMATH_CALUDE_remaining_amount_proof_l3414_341407

-- Define the deposit percentage
def deposit_percentage : ℚ := 10 / 100

-- Define the deposit amount
def deposit_amount : ℚ := 55

-- Define the total cost
def total_cost : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_cost - deposit_amount

-- Theorem to prove
theorem remaining_amount_proof : remaining_amount = 495 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_proof_l3414_341407


namespace NUMINAMATH_CALUDE_same_label_probability_l3414_341409

def deck_size : ℕ := 50
def num_labels : ℕ := 13
def cards_per_label (i : ℕ) : ℕ :=
  if i < num_labels then 4 else if i = num_labels then 2 else 0

def total_combinations : ℕ := deck_size.choose 2

def favorable_combinations : ℕ :=
  (Finset.range num_labels).sum (λ i => (cards_per_label i).choose 2)

theorem same_label_probability :
  (favorable_combinations : ℚ) / total_combinations = 73 / 1225 := by sorry

end NUMINAMATH_CALUDE_same_label_probability_l3414_341409


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l3414_341419

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) : 
  ((5 * x - 3) / (2 * y + 10) = k) →  -- The ratio is constant
  (y = 2 → x = 3) →                   -- When y = 2, x = 3
  (y = 5 → x = 47 / 5) :=             -- When y = 5, x = 47/5
by
  sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l3414_341419


namespace NUMINAMATH_CALUDE_consecutive_sum_equals_50_l3414_341467

/-- The sum of consecutive integers from a given start to an end -/
def sum_consecutive (start : Int) (count : Nat) : Int :=
  count * (2 * start + count.pred) / 2

/-- Proves that there are exactly 100 consecutive integers starting from -49 whose sum is 50 -/
theorem consecutive_sum_equals_50 : ∃! n : Nat, sum_consecutive (-49) n = 50 ∧ n > 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_equals_50_l3414_341467


namespace NUMINAMATH_CALUDE_book_cost_problem_l3414_341423

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h_total : total_cost = 600)
  (h_loss : loss_percent = 15)
  (h_gain : gain_percent = 19) :
  ∃ (cost_loss cost_gain : ℝ),
    cost_loss + cost_gain = total_cost ∧
    cost_loss * (1 - loss_percent / 100) = cost_gain * (1 + gain_percent / 100) ∧
    cost_loss = 350 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3414_341423


namespace NUMINAMATH_CALUDE_weekly_wage_calculation_l3414_341405

def basic_daily_wage : ℕ := 200
def basic_task_quantity : ℕ := 40
def reward_per_excess : ℕ := 7
def deduction_per_incomplete : ℕ := 8
def work_days : ℕ := 5
def production_deviations : List ℤ := [5, -2, -1, 0, 4]

def total_weekly_wage : ℕ := 1039

theorem weekly_wage_calculation :
  (basic_daily_wage * work_days) +
  (production_deviations.filter (λ x => x > 0)).sum * reward_per_excess -
  (production_deviations.filter (λ x => x < 0)).sum.natAbs * deduction_per_incomplete =
  total_weekly_wage :=
sorry

end NUMINAMATH_CALUDE_weekly_wage_calculation_l3414_341405


namespace NUMINAMATH_CALUDE_cashew_mixture_problem_l3414_341455

/-- Represents the price of peanuts per pound -/
def peanut_price : ℝ := 2.40

/-- Represents the price of cashews per pound -/
def cashew_price : ℝ := 6.00

/-- Represents the total weight of the mixture in pounds -/
def total_weight : ℝ := 60

/-- Represents the selling price of the mixture per pound -/
def mixture_price : ℝ := 3.00

/-- Represents the amount of cashews in pounds -/
def cashew_amount : ℝ := 10

theorem cashew_mixture_problem :
  ∃ (peanut_amount : ℝ),
    peanut_amount + cashew_amount = total_weight ∧
    peanut_price * peanut_amount + cashew_price * cashew_amount = mixture_price * total_weight :=
by
  sorry

end NUMINAMATH_CALUDE_cashew_mixture_problem_l3414_341455


namespace NUMINAMATH_CALUDE_pool_filling_cost_l3414_341498

/-- The cost to fill Toby's swimming pool -/
theorem pool_filling_cost 
  (fill_time : ℕ) 
  (flow_rate : ℕ) 
  (water_cost : ℚ) : 
  fill_time = 50 → 
  flow_rate = 100 → 
  water_cost = 1 / 1000 → 
  (fill_time * flow_rate * water_cost : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_pool_filling_cost_l3414_341498


namespace NUMINAMATH_CALUDE_point_outside_circle_l3414_341443

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- A point with a given distance from the center of a circle -/
structure Point where
  distanceFromCenter : ℝ

/-- Determines if a point is outside a circle -/
def isOutside (c : Circle) (p : Point) : Prop :=
  p.distanceFromCenter > c.radius

/-- Theorem: If the radius of a circle is 3 and the distance from a point to the center is 4,
    then the point is outside the circle -/
theorem point_outside_circle (c : Circle) (p : Point)
    (h1 : c.radius = 3)
    (h2 : p.distanceFromCenter = 4) :
    isOutside c p := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3414_341443


namespace NUMINAMATH_CALUDE_triangle_side_sum_l3414_341465

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) (side_c : ℝ) (h5 : side_c = 8) : 
  ∃ (side_a side_b : ℝ), 
    abs ((side_a + side_b) - 18.9) < 0.05 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l3414_341465


namespace NUMINAMATH_CALUDE_student_arrangement_l3414_341451

/-- Number of ways to arrange n distinct objects from m objects -/
def arrangement (m n : ℕ) : ℕ := sorry

/-- The number of students -/
def total_students : ℕ := 14

/-- The number of female students -/
def female_students : ℕ := 6

/-- The number of male students -/
def male_students : ℕ := 8

/-- The number of female students that must be grouped together -/
def grouped_females : ℕ := 4

/-- The number of gaps after arranging male students and grouped females -/
def gaps : ℕ := male_students + 1

theorem student_arrangement :
  arrangement male_students male_students *
  arrangement gaps (female_students - grouped_females) *
  arrangement grouped_females grouped_females =
  arrangement total_students total_students := by sorry

end NUMINAMATH_CALUDE_student_arrangement_l3414_341451


namespace NUMINAMATH_CALUDE_unsold_books_l3414_341424

def initial_stock : ℕ := 800
def monday_sales : ℕ := 60
def tuesday_sales : ℕ := 10
def wednesday_sales : ℕ := 20
def thursday_sales : ℕ := 44
def friday_sales : ℕ := 66

theorem unsold_books :
  initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales) = 600 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l3414_341424


namespace NUMINAMATH_CALUDE_original_page_count_l3414_341430

/-- Represents a book with numbered pages -/
structure Book where
  pages : ℕ

/-- Calculates the total number of digits in the page numbers of remaining pages after removing even-numbered sheets -/
def remainingDigits (b : Book) : ℕ := sorry

/-- Theorem stating the possible original page counts given the remaining digit count -/
theorem original_page_count (b : Book) : 
  remainingDigits b = 845 → b.pages = 598 ∨ b.pages = 600 := by sorry

end NUMINAMATH_CALUDE_original_page_count_l3414_341430


namespace NUMINAMATH_CALUDE_sum_equals_fraction_l3414_341469

def binomial_coefficient (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def sum_expression : ℚ :=
  Finset.sum (Finset.range 8) (fun i =>
    let n := i + 3
    (binomial_coefficient n 2) / ((binomial_coefficient n 3) * (binomial_coefficient (n + 1) 3)))

theorem sum_equals_fraction :
  sum_expression = 164 / 165 :=
sorry

end NUMINAMATH_CALUDE_sum_equals_fraction_l3414_341469


namespace NUMINAMATH_CALUDE_gcd_35_and_number_between_80_90_l3414_341450

theorem gcd_35_and_number_between_80_90 :
  ∃! n : ℕ, 80 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 35 n = 7 :=
by sorry

end NUMINAMATH_CALUDE_gcd_35_and_number_between_80_90_l3414_341450


namespace NUMINAMATH_CALUDE_composite_power_sum_l3414_341462

theorem composite_power_sum (n : ℕ) (h : n % 6 = 4) : 3 ∣ (n^n + (n+1)^(n+1)) := by
  sorry

end NUMINAMATH_CALUDE_composite_power_sum_l3414_341462


namespace NUMINAMATH_CALUDE_remainder_theorem_l3414_341470

theorem remainder_theorem : (9 * 7^18 + 2^18) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3414_341470


namespace NUMINAMATH_CALUDE_cube_plane_intersection_theorem_l3414_341453

/-- A regular polygon that can be formed by the intersection of a cube and a plane -/
inductive CubeIntersectionPolygon
  | Triangle
  | Quadrilateral
  | Hexagon

/-- The set of all possible regular polygons that can be formed by the intersection of a cube and a plane -/
def possibleIntersectionPolygons : Set CubeIntersectionPolygon :=
  {CubeIntersectionPolygon.Triangle, CubeIntersectionPolygon.Quadrilateral, CubeIntersectionPolygon.Hexagon}

/-- A function that determines if a given regular polygon can be formed by the intersection of a cube and a plane -/
def isValidIntersectionPolygon (p : CubeIntersectionPolygon) : Prop :=
  p ∈ possibleIntersectionPolygons

theorem cube_plane_intersection_theorem :
  ∀ (p : CubeIntersectionPolygon), isValidIntersectionPolygon p ↔
    (p = CubeIntersectionPolygon.Triangle ∨
     p = CubeIntersectionPolygon.Quadrilateral ∨
     p = CubeIntersectionPolygon.Hexagon) :=
by sorry


end NUMINAMATH_CALUDE_cube_plane_intersection_theorem_l3414_341453


namespace NUMINAMATH_CALUDE_second_month_sale_l3414_341493

def average_sale : ℕ := 6800
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month3 : ℕ := 7230
def sale_month4 : ℕ := 6562
def sale_month5 : ℕ := 6791
def sale_month6 : ℕ := 6791

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = 13991 ∧
    (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / num_months = average_sale :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l3414_341493


namespace NUMINAMATH_CALUDE_tangent_circles_a_values_l3414_341474

/-- Two circles that intersect at exactly one point -/
structure TangentCircles where
  /-- The parameter 'a' in the equation of the second circle -/
  a : ℝ
  /-- The first circle: x^2 + y^2 = 4 -/
  circle1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 + y^2 = 4
  /-- The second circle: (x-a)^2 + y^2 = 1 -/
  circle2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + y^2 = 1
  /-- The circles intersect at exactly one point -/
  intersect_at_one_point : ∃! p : ℝ × ℝ, circle1 p.1 p.2 ∧ circle2 p.1 p.2

/-- The theorem stating that 'a' must be in the set {1, -1, 3, -3} -/
theorem tangent_circles_a_values (tc : TangentCircles) : tc.a = 1 ∨ tc.a = -1 ∨ tc.a = 3 ∨ tc.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_a_values_l3414_341474


namespace NUMINAMATH_CALUDE_difference_of_same_prime_divisors_l3414_341491

/-- For any natural number, there exist two natural numbers with the same number of distinct prime divisors whose difference is the original number. -/
theorem difference_of_same_prime_divisors (n : ℕ) : 
  ∃ a b : ℕ, n = a - b ∧ (Finset.card (Nat.factorization a).support = Finset.card (Nat.factorization b).support) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_same_prime_divisors_l3414_341491


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3414_341477

theorem inequality_solution_set (x : ℝ) :
  (Set.Icc (-2 : ℝ) 3 : Set ℝ) = {x | (x - 1)^2 * (x + 2) * (x - 3) ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3414_341477


namespace NUMINAMATH_CALUDE_alice_yard_side_length_l3414_341421

/-- Given that Alice needs to buy 12 bushes to plant around three sides of her yard,
    and each bush fills 4 feet, prove that each side of her yard is 16 feet long. -/
theorem alice_yard_side_length
  (num_bushes : ℕ)
  (bush_length : ℕ)
  (num_sides : ℕ)
  (h1 : num_bushes = 12)
  (h2 : bush_length = 4)
  (h3 : num_sides = 3) :
  (num_bushes * bush_length) / num_sides = 16 := by
  sorry

end NUMINAMATH_CALUDE_alice_yard_side_length_l3414_341421


namespace NUMINAMATH_CALUDE_find_x_in_ratio_l3414_341401

/-- Given t = 5, prove that the positive integer x satisfying 2 : m : t = m : 32 : x is 20 -/
theorem find_x_in_ratio (t : ℕ) (h_t : t = 5) :
  ∃ (m : ℤ) (x : ℕ), 2 * 32 * t = m * m * x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_x_in_ratio_l3414_341401


namespace NUMINAMATH_CALUDE_circle_properties_l3414_341440

/-- Given a circle C with equation x^2 + y^2 - 2x - 2y - 2 = 0,
    prove that its radius is 2 and its center is at (1, 1) -/
theorem circle_properties (x y : ℝ) :
  x^2 + y^2 - 2*x - 2*y - 2 = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 1) ∧ radius = 2 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3414_341440


namespace NUMINAMATH_CALUDE_problem_solution_l3414_341435

theorem problem_solution (n : ℕ+) 
  (x : ℝ) (hx : x = (Real.sqrt (n + 2) - Real.sqrt n) / (Real.sqrt (n + 2) + Real.sqrt n))
  (y : ℝ) (hy : y = (Real.sqrt (n + 2) + Real.sqrt n) / (Real.sqrt (n + 2) - Real.sqrt n))
  (h_eq : 14 * x^2 + 26 * x * y + 14 * y^2 = 2014) :
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3414_341435


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3414_341463

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The point P -/
def point_P : ℝ × ℝ := (-1, 0)

/-- The center of circle C -/
def center_C : ℝ × ℝ := (1, 2)

/-- The equation of the circle passing through the tangency points and the center of C -/
def target_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

/-- The theorem stating that the target circle passes through the tangency points and the center of C -/
theorem tangent_circle_equation : 
  ∃ (A B : ℝ × ℝ), 
    (∀ x y, circle_C x y → ((x - point_P.1)^2 + (y - point_P.2)^2 ≤ (A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)) ∧
    (∀ x y, circle_C x y → ((x - point_P.1)^2 + (y - point_P.2)^2 ≤ (B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) ∧
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    target_circle A.1 A.2 ∧
    target_circle B.1 B.2 ∧
    target_circle center_C.1 center_C.2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3414_341463


namespace NUMINAMATH_CALUDE_files_deleted_l3414_341448

/-- Given Dave's initial and final number of files, prove the number of files deleted. -/
theorem files_deleted (initial_files final_files : ℕ) 
  (h1 : initial_files = 24)
  (h2 : final_files = 21) :
  initial_files - final_files = 3 := by
  sorry

#check files_deleted

end NUMINAMATH_CALUDE_files_deleted_l3414_341448


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l3414_341495

/-- Represents the fuel efficiency of a car -/
structure FuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- The conditions of the problem -/
def problem_conditions (fe : FuelEfficiency) : Prop :=
  fe.highway * fe.tank_size = 420 ∧
  fe.city * fe.tank_size = 336 ∧
  fe.city = fe.highway - 6

/-- The theorem to be proved -/
theorem city_fuel_efficiency 
  (fe : FuelEfficiency) 
  (h : problem_conditions fe) : 
  fe.city = 24 := by
  sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l3414_341495


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_greater_than_500_l3414_341442

theorem smallest_multiple_of_seven_greater_than_500 :
  ∃ (n : ℕ), n * 7 = 504 ∧ 
  504 > 500 ∧
  ∀ (m : ℕ), m * 7 > 500 → m * 7 ≥ 504 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_greater_than_500_l3414_341442


namespace NUMINAMATH_CALUDE_billy_can_play_24_songs_l3414_341475

/-- The number of songs in Billy's music book -/
def total_songs : ℕ := 52

/-- The number of songs Billy still needs to learn -/
def songs_to_learn : ℕ := 28

/-- The number of songs Billy can play -/
def playable_songs : ℕ := total_songs - songs_to_learn

theorem billy_can_play_24_songs : playable_songs = 24 := by
  sorry

end NUMINAMATH_CALUDE_billy_can_play_24_songs_l3414_341475


namespace NUMINAMATH_CALUDE_greatest_divisor_of_sum_first_12_terms_l3414_341425

-- Define an arithmetic sequence of positive integers
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (x c : ℕ), ∀ n, a n = x + n * c

-- Define the sum of the first 12 terms
def SumFirst12Terms (a : ℕ → ℕ) : ℕ :=
  (List.range 12).map a |>.sum

-- Theorem statement
theorem greatest_divisor_of_sum_first_12_terms :
  ∀ a : ℕ → ℕ, ArithmeticSequence a →
  (∃ k : ℕ, k > 6 ∧ k ∣ SumFirst12Terms a) → False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_sum_first_12_terms_l3414_341425


namespace NUMINAMATH_CALUDE_range_of_a_l3414_341437

-- Define the statements p and q
def p (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 + (k*x₁ + 1)^2/a = 1) ∧ 
    (x₂^2 + (k*x₂ + 1)^2/a = 1)

def q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, 4^x₀ - 2^x₀ - a ≤ 0

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, ¬(p a ∧ q a)) ∧ (∀ a : ℝ, p a ∨ q a) →
  ∀ a : ℝ, -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3414_341437


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3414_341494

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a^2 > 2*a ∧ ¬(a > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3414_341494


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3414_341420

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | 2 < x < 3},
    prove that the solution set of ax^2 - bx + c > 0 is {x | -3 < x < -2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, ax^2 - b*x + c > 0 ↔ -3 < x ∧ x < -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3414_341420


namespace NUMINAMATH_CALUDE_sin_135_degrees_l3414_341461

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l3414_341461


namespace NUMINAMATH_CALUDE_triangle_theorem_l3414_341497

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the main theorem
theorem triangle_theorem (ABC : Triangle) 
  (h1 : (Real.cos ABC.B - 2 * Real.cos ABC.A) / (2 * ABC.a - ABC.b) = Real.cos ABC.C / ABC.c) :
  -- Part 1: a/b = 2
  ABC.a / ABC.b = 2 ∧
  -- Part 2: If angle A is obtuse and c = 3, then 0 < b < 3
  (ABC.A > Real.pi / 2 ∧ ABC.c = 3 → 0 < ABC.b ∧ ABC.b < 3) :=
by sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3414_341497


namespace NUMINAMATH_CALUDE_apollonius_circle_locus_l3414_341412

/-- Given two points A and B in a 2D plane, and a positive real number n,
    the Apollonius circle is the locus of points P such that PA = n * PB -/
theorem apollonius_circle_locus 
  (A B : EuclideanSpace ℝ (Fin 2))  -- Two given points in 2D space
  (n : ℝ) 
  (hn : n > 0) :  -- n is positive
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    ∀ P : EuclideanSpace ℝ (Fin 2), 
      dist P A = n * dist P B ↔ 
      dist P center = radius :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_locus_l3414_341412


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3414_341471

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  1 + a^2017 + b^2017 ≥ a^10 * b^7 + a^7 * b^2000 + a^2000 * b^10 ∧
  (1 + a^2017 + b^2017 = a^10 * b^7 + a^7 * b^2000 + a^2000 * b^10 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3414_341471


namespace NUMINAMATH_CALUDE_yellow_balloon_ratio_l3414_341408

theorem yellow_balloon_ratio (total_balloons : ℕ) (num_colors : ℕ) (anya_balloons : ℕ) : 
  total_balloons = 672 →
  num_colors = 4 →
  anya_balloons = 84 →
  (anya_balloons : ℚ) / (total_balloons / num_colors) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balloon_ratio_l3414_341408


namespace NUMINAMATH_CALUDE_regression_estimate_l3414_341476

theorem regression_estimate :
  let regression_equation (x : ℝ) := 4.75 * x + 2.57
  regression_equation 28 = 135.57 := by sorry

end NUMINAMATH_CALUDE_regression_estimate_l3414_341476


namespace NUMINAMATH_CALUDE_deepak_age_l3414_341406

/-- Proves that Deepak's current age is 42 years given the specified conditions --/
theorem deepak_age (arun deepak kamal : ℕ) : 
  arun * 7 = deepak * 5 →
  kamal * 5 = deepak * 9 →
  arun + 6 = 36 →
  kamal + 6 = 2 * (deepak + 6) →
  deepak = 42 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3414_341406


namespace NUMINAMATH_CALUDE_range_of_g_l3414_341490

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.sin x ^ 2 + Real.cos x ^ 4 ∧ Real.sin x ^ 2 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3414_341490


namespace NUMINAMATH_CALUDE_strawberries_left_l3414_341434

/-- Theorem: If Adam picked 35 strawberries and ate 2 strawberries, then he has 33 strawberries left. -/
theorem strawberries_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 35) (h2 : eaten = 2) :
  initial - eaten = 33 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_left_l3414_341434


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3414_341464

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + m*x + 1 = 0

-- Define the property of having two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  m < -2 ∨ m > 2

-- State the theorem
theorem quadratic_roots_range :
  ∀ m : ℝ, has_two_distinct_real_roots m ↔ m_range m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3414_341464


namespace NUMINAMATH_CALUDE_increase_in_average_goals_is_point_two_l3414_341416

/-- Calculates the increase in average goals score after the fifth match -/
def increase_in_average_goals (total_matches : ℕ) (total_goals : ℕ) (goals_in_fifth_match : ℕ) : ℚ :=
  let goals_before_fifth := total_goals - goals_in_fifth_match
  let matches_before_fifth := total_matches - 1
  let average_before := goals_before_fifth / matches_before_fifth
  let average_after := total_goals / total_matches
  average_after - average_before

/-- The increase in average goals score after the fifth match is 0.2 -/
theorem increase_in_average_goals_is_point_two :
  increase_in_average_goals 5 21 5 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_goals_is_point_two_l3414_341416


namespace NUMINAMATH_CALUDE_max_profit_and_optimal_price_l3414_341415

/-- Represents the profit function for a product with given initial conditions -/
def profit (x : ℝ) : ℝ :=
  (500 - 10 * x) * ((50 + x) - 40)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit_and_optimal_price :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    (∀ x : ℝ, profit x ≤ max_profit) ∧
    (profit (optimal_price - 50) = max_profit) ∧
    max_profit = 9000 ∧
    optimal_price = 70 := by
  sorry

#check max_profit_and_optimal_price

end NUMINAMATH_CALUDE_max_profit_and_optimal_price_l3414_341415


namespace NUMINAMATH_CALUDE_rope_folding_theorem_l3414_341456

def rope_segments (n : ℕ) : ℕ := 2^n + 1

theorem rope_folding_theorem :
  rope_segments 5 = 33 := by sorry

end NUMINAMATH_CALUDE_rope_folding_theorem_l3414_341456


namespace NUMINAMATH_CALUDE_abs_sum_simplification_l3414_341426

theorem abs_sum_simplification (m x : ℝ) (h1 : 0 < m) (h2 : m < 10) (h3 : m ≤ x) (h4 : x ≤ 10) :
  |x - m| + |x - 10| + |x - m - 10| = 20 - x := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_simplification_l3414_341426


namespace NUMINAMATH_CALUDE_bake_sale_total_l3414_341485

/-- Represents the number of cookies sold at a bake sale -/
structure CookieSale where
  raisin : ℕ
  oatmeal : ℕ
  chocolate_chip : ℕ

/-- Theorem stating the total number of cookies sold given the conditions -/
theorem bake_sale_total (sale : CookieSale) : 
  sale.raisin = 42 ∧ 
  sale.raisin = 6 * sale.oatmeal ∧ 
  sale.raisin = 2 * sale.chocolate_chip → 
  sale.raisin + sale.oatmeal + sale.chocolate_chip = 70 := by
  sorry

#check bake_sale_total

end NUMINAMATH_CALUDE_bake_sale_total_l3414_341485


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l3414_341410

theorem polynomial_coefficient_b (a b c : ℚ) : 
  (∀ x, (5*x^2 - 3*x + 7/3) * (a*x^2 + b*x + c) = 
        15*x^4 - 14*x^3 + 20*x^2 - 25/3*x + 14/3) →
  b = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l3414_341410


namespace NUMINAMATH_CALUDE_circle_area_above_line_l3414_341431

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 18*y + 61 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 4

-- Theorem statement
theorem circle_area_above_line : 
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (center_y > 4) ∧
    (center_y - radius > 4) ∧
    (radius = 1) ∧
    (Real.pi * radius^2 = Real.pi) :=
sorry

end NUMINAMATH_CALUDE_circle_area_above_line_l3414_341431


namespace NUMINAMATH_CALUDE_smallest_n_with_6474_l3414_341480

def contains_subsequence (s t : List Nat) : Prop :=
  ∃ i, t = s.drop i ++ s.take i

def digits_to_list (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

def concatenate_digits (a b c : Nat) : List Nat :=
  (digits_to_list a) ++ (digits_to_list b) ++ (digits_to_list c)

theorem smallest_n_with_6474 :
  ∀ n : Nat, n < 46 →
    ¬(contains_subsequence (concatenate_digits n (n+1) (n+2)) [6,4,7,4]) ∧
  contains_subsequence (concatenate_digits 46 47 48) [6,4,7,4] :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_6474_l3414_341480


namespace NUMINAMATH_CALUDE_walking_time_calculation_l3414_341484

/-- Given a person walking at a constant rate who covers 45 meters in 15 minutes,
    prove that it will take 30 minutes to cover an additional 90 meters. -/
theorem walking_time_calculation (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
    (h1 : initial_distance = 45)
    (h2 : initial_time = 15)
    (h3 : additional_distance = 90) :
    additional_distance / (initial_distance / initial_time) = 30 := by
  sorry


end NUMINAMATH_CALUDE_walking_time_calculation_l3414_341484


namespace NUMINAMATH_CALUDE_competition_results_l3414_341492

structure GradeData where
  boys_rate : ℝ
  girls_rate : ℝ

def seventh_grade : GradeData :=
  { boys_rate := 0.4, girls_rate := 0.6 }

def eighth_grade : GradeData :=
  { boys_rate := 0.5, girls_rate := 0.7 }

theorem competition_results :
  (seventh_grade.boys_rate < eighth_grade.boys_rate) ∧
  ((seventh_grade.boys_rate + eighth_grade.boys_rate) / 2 < (seventh_grade.girls_rate + eighth_grade.girls_rate) / 2) :=
by sorry

end NUMINAMATH_CALUDE_competition_results_l3414_341492


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l3414_341433

theorem last_three_digits_of_7_power_10000 (h : 7^500 ≡ 1 [ZMOD 1250]) :
  7^10000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l3414_341433


namespace NUMINAMATH_CALUDE_stuffed_animals_theorem_l3414_341489

/-- Represents the number of stuffed animals for each girl -/
structure StuffedAnimals where
  mckenna : ℕ
  kenley : ℕ
  tenly : ℕ

/-- Calculates the total number of stuffed animals -/
def total (sa : StuffedAnimals) : ℕ :=
  sa.mckenna + sa.kenley + sa.tenly

/-- Calculates the average number of stuffed animals per girl -/
def average (sa : StuffedAnimals) : ℚ :=
  (total sa : ℚ) / 3

/-- Calculates the percentage of total stuffed animals McKenna has -/
def mckennaPercentage (sa : StuffedAnimals) : ℚ :=
  (sa.mckenna : ℚ) / (total sa : ℚ) * 100

theorem stuffed_animals_theorem (sa : StuffedAnimals) 
  (h1 : sa.mckenna = 34)
  (h2 : sa.kenley = 2 * sa.mckenna)
  (h3 : sa.tenly = sa.kenley + 5) :
  total sa = 175 ∧ 
  58.32 < average sa ∧ average sa < 58.34 ∧
  19.42 < mckennaPercentage sa ∧ mckennaPercentage sa < 19.44 := by
  sorry

#eval total { mckenna := 34, kenley := 68, tenly := 73 }
#eval average { mckenna := 34, kenley := 68, tenly := 73 }
#eval mckennaPercentage { mckenna := 34, kenley := 68, tenly := 73 }

end NUMINAMATH_CALUDE_stuffed_animals_theorem_l3414_341489


namespace NUMINAMATH_CALUDE_john_cookies_problem_l3414_341499

theorem john_cookies_problem (cookies_left : ℕ) (cookies_eaten : ℕ) (dozen : ℕ) :
  cookies_left = 21 →
  cookies_eaten = 3 →
  dozen = 12 →
  (cookies_left + cookies_eaten) / dozen = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_cookies_problem_l3414_341499


namespace NUMINAMATH_CALUDE_middle_number_proof_l3414_341454

theorem middle_number_proof (a b c : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_order : a < b ∧ b < c)
  (h_sum_ab : a + b = 18)
  (h_sum_ac : a + c = 22)
  (h_sum_bc : b + c = 26)
  (h_diff : c - a = 10) : 
  b = 11 := by sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3414_341454


namespace NUMINAMATH_CALUDE_sum_parity_l3414_341432

theorem sum_parity (a b : ℤ) (h : a + b = 1998) : 
  ∃ k : ℤ, 7 * a + 3 * b = 2 * k ∧ 7 * a + 3 * b ≠ 6799 := by
sorry

end NUMINAMATH_CALUDE_sum_parity_l3414_341432


namespace NUMINAMATH_CALUDE_candy_mix_proof_l3414_341483

/-- Proves that mixing 1 pound of candy A with 4 pounds of candy B
    produces 5 pounds of mixed candy that costs $2.00 per pound -/
theorem candy_mix_proof (candy_a_cost candy_b_cost mix_cost : ℝ)
                        (candy_a_weight candy_b_weight : ℝ) :
  candy_a_cost = 3.20 →
  candy_b_cost = 1.70 →
  mix_cost = 2.00 →
  candy_a_weight = 1 →
  candy_b_weight = 4 →
  (candy_a_cost * candy_a_weight + candy_b_cost * candy_b_weight) / 
    (candy_a_weight + candy_b_weight) = mix_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_mix_proof_l3414_341483


namespace NUMINAMATH_CALUDE_curve_representation_l3414_341472

-- Define the equations
def equation1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0
def equation2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- Define what it means for an equation to represent a line and a circle
def represents_line_and_circle (f : ℝ → ℝ → Prop) : Prop :=
  (∃ a : ℝ, ∀ y, f a y) ∧ 
  (∃ c r : ℝ, ∀ x y, f x y ↔ (x - c)^2 + y^2 = r^2)

-- Define what it means for an equation to represent two points
def represents_two_points (f : ℝ → ℝ → Prop) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∨ y1 ≠ y2 ∧ 
    (∀ x y, f x y ↔ (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- State the theorem
theorem curve_representation :
  represents_line_and_circle equation1 ∧ 
  represents_two_points equation2 := by sorry

end NUMINAMATH_CALUDE_curve_representation_l3414_341472


namespace NUMINAMATH_CALUDE_range_of_m_l3414_341441

theorem range_of_m (m : ℝ) : 
  (m + 4)^(-1/2 : ℝ) < (3 - 2*m)^(-1/2 : ℝ) → 
  -1/3 < m ∧ m < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3414_341441


namespace NUMINAMATH_CALUDE_percentage_problem_l3414_341479

theorem percentage_problem (P : ℝ) : 
  (1/10 * 7000 - P/100 * 7000 = 700) → P = 0 :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3414_341479


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3414_341482

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Defines a line passing through two points -/
structure Line where
  m : ℝ
  c : ℝ

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties
  (M : Ellipse)
  (h_eccentricity : M.a^2 - M.b^2 = M.a^2 / 2)
  (AB : Line)
  (h_AB_points : AB.m * (-M.a) + AB.c = 0 ∧ AB.c = M.b)
  (h_AB_distance : (M.a * M.b / Real.sqrt (M.a^2 + M.b^2))^2 = 2/3)
  (l : Line)
  (h_l_point : l.c = -1)
  (h_intersection_ratio : ∃ (y₁ y₂ : ℝ), y₁ = -3 * y₂ ∧
    y₁ + y₂ = -2 * l.m / (l.m^2 + 2) ∧
    y₁ * y₂ = -1 / (l.m^2 + 2)) :
  M.a^2 = 2 ∧ M.b^2 = 1 ∧ l.m = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3414_341482


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3414_341460

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3414_341460


namespace NUMINAMATH_CALUDE_absent_children_l3414_341447

theorem absent_children (total_children : ℕ) (total_bananas : ℕ) : 
  total_children = 610 →
  total_bananas = 610 * 2 →
  total_bananas = (610 - (total_children - (610 - 305))) * 4 →
  610 - 305 = total_children - (610 - 305) :=
by
  sorry

end NUMINAMATH_CALUDE_absent_children_l3414_341447


namespace NUMINAMATH_CALUDE_percentage_of_120_to_80_l3414_341481

theorem percentage_of_120_to_80 : ∃ (p : ℝ), (120 : ℝ) / 80 * 100 = p ∧ p = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_80_l3414_341481


namespace NUMINAMATH_CALUDE_orange_weight_change_l3414_341436

theorem orange_weight_change (initial_weight : ℝ) (initial_water_percent : ℝ) (water_decrease : ℝ) : 
  initial_weight = 5 →
  initial_water_percent = 95 →
  water_decrease = 5 →
  let non_water_weight := initial_weight * (100 - initial_water_percent) / 100
  let new_water_percent := initial_water_percent - water_decrease
  let new_total_weight := non_water_weight / ((100 - new_water_percent) / 100)
  new_total_weight = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_weight_change_l3414_341436


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l3414_341444

theorem restaurant_bill_theorem (num_teenagers : ℕ) (avg_meal_cost : ℝ) (gratuity_rate : ℝ) :
  num_teenagers = 7 →
  avg_meal_cost = 100 →
  gratuity_rate = 0.20 →
  let total_before_gratuity := num_teenagers * avg_meal_cost
  let gratuity := total_before_gratuity * gratuity_rate
  let total_bill := total_before_gratuity + gratuity
  total_bill = 840 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l3414_341444


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3414_341413

-- Define the original number
def original_number : ℝ := 850000

-- Define the scientific notation components
def coefficient : ℝ := 8.5
def exponent : ℤ := 5

-- Theorem statement
theorem scientific_notation_equality :
  original_number = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3414_341413


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3414_341403

theorem cube_volume_problem (a : ℕ) : 
  (a - 2) * a * (a + 2) = a^3 - 14 → a^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3414_341403


namespace NUMINAMATH_CALUDE_range_of_b_given_false_proposition_l3414_341438

theorem range_of_b_given_false_proposition :
  (¬ ∃ a : ℝ, a < 0 ∧ a + 1/a > b) →
  ∀ b : ℝ, b ≥ -2 ↔ b ∈ Set.Ici (-2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_given_false_proposition_l3414_341438
