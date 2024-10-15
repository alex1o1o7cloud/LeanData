import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_group_a_l3247_324719

/-- Calculates the number of cities to be selected from a group in stratified sampling -/
def stratifiedSampleSize (totalCities : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℚ :=
  (groupSize : ℚ) * (sampleSize : ℚ) / (totalCities : ℚ)

/-- Theorem: In a stratified sampling of 6 cities from 24 total cities, 
    where 4 cities belong to group A, 1 city should be selected from group A -/
theorem stratified_sampling_group_a : 
  stratifiedSampleSize 24 4 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_a_l3247_324719


namespace NUMINAMATH_CALUDE_largest_common_term_l3247_324767

def is_in_ap1 (a : ℕ) : Prop := ∃ k : ℕ, a = 4 + 5 * k

def is_in_ap2 (a : ℕ) : Prop := ∃ k : ℕ, a = 7 + 11 * k

def is_common_term (a : ℕ) : Prop := is_in_ap1 a ∧ is_in_ap2 a

theorem largest_common_term :
  ∃ a : ℕ, a = 984 ∧ is_common_term a ∧ a < 1000 ∧
  ∀ b : ℕ, is_common_term b ∧ b < 1000 → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l3247_324767


namespace NUMINAMATH_CALUDE_fraction_inequality_l3247_324741

theorem fraction_inequality (x y : ℝ) (h : x > y) : x / 5 > y / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3247_324741


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3247_324782

theorem arithmetic_operations : 
  ((-9) + ((-4) * 5) = -29) ∧ 
  ((6 * (-2)) / (2/3) = -18) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3247_324782


namespace NUMINAMATH_CALUDE_total_earnings_is_18_56_l3247_324729

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 150

/-- Represents the percentage of white marbles -/
def white_percent : ℚ := 20 / 100

/-- Represents the percentage of black marbles -/
def black_percent : ℚ := 25 / 100

/-- Represents the percentage of blue marbles -/
def blue_percent : ℚ := 30 / 100

/-- Represents the percentage of green marbles -/
def green_percent : ℚ := 15 / 100

/-- Represents the percentage of red marbles -/
def red_percent : ℚ := 10 / 100

/-- Represents the price of a white marble in dollars -/
def white_price : ℚ := 5 / 100

/-- Represents the price of a black marble in dollars -/
def black_price : ℚ := 10 / 100

/-- Represents the price of a blue marble in dollars -/
def blue_price : ℚ := 15 / 100

/-- Represents the price of a green marble in dollars -/
def green_price : ℚ := 12 / 100

/-- Represents the price of a red marble in dollars -/
def red_price : ℚ := 25 / 100

/-- Theorem stating that the total earnings from selling all marbles is $18.56 -/
theorem total_earnings_is_18_56 : 
  (↑total_marbles * white_percent * white_price) +
  (↑total_marbles * black_percent * black_price) +
  (↑total_marbles * blue_percent * blue_price) +
  (↑total_marbles * green_percent * green_price) +
  (↑total_marbles * red_percent * red_price) = 1856 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_18_56_l3247_324729


namespace NUMINAMATH_CALUDE_robotics_workshop_average_age_l3247_324774

theorem robotics_workshop_average_age (total_members : Nat) (overall_avg : Nat) 
  (num_girls num_boys num_adults : Nat) (avg_girls avg_boys : Nat) :
  total_members = 50 →
  overall_avg = 21 →
  num_girls = 25 →
  num_boys = 20 →
  num_adults = 5 →
  avg_girls = 18 →
  avg_boys = 20 →
  (total_members * overall_avg - num_girls * avg_girls - num_boys * avg_boys) / num_adults = 40 :=
by sorry

end NUMINAMATH_CALUDE_robotics_workshop_average_age_l3247_324774


namespace NUMINAMATH_CALUDE_total_spent_on_toys_l3247_324732

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_toys_l3247_324732


namespace NUMINAMATH_CALUDE_h_derivative_l3247_324737

/-- Given f = 5, g = 4g', and h(x) = (f + 2) / x, prove that h'(x) = 5/16 -/
theorem h_derivative (f g g' : ℝ) (h : ℝ → ℝ) :
  f = 5 →
  g = 4 * g' →
  (∀ x, h x = (f + 2) / x) →
  ∀ x, deriv h x = 5 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_h_derivative_l3247_324737


namespace NUMINAMATH_CALUDE_min_segments_polyline_l3247_324731

/-- Represents a square grid divided into n^2 smaller squares -/
structure SquareGrid (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents a polyline that passes through the centers of all smaller squares -/
structure Polyline (n : ℕ) where
  grid : SquareGrid n
  segments : ℕ
  passes_all_centers : segments ≥ 1

/-- Theorem stating the minimum number of segments in the polyline -/
theorem min_segments_polyline (n : ℕ) (h : n > 0) :
  ∃ (p : Polyline n), ∀ (q : Polyline n), p.segments ≤ q.segments ∧ p.segments = 2 * n - 2 :=
sorry

end NUMINAMATH_CALUDE_min_segments_polyline_l3247_324731


namespace NUMINAMATH_CALUDE_circle_max_min_value_l3247_324770

theorem circle_max_min_value (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S_max S_min : ℝ),
    (∀ S, S = 3*x - y → S ≤ S_max ∧ S ≥ S_min) ∧
    S_max = 5 + 2 * Real.sqrt 10 ∧
    S_min = 5 - 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_min_value_l3247_324770


namespace NUMINAMATH_CALUDE_valid_numbers_l3247_324766

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_valid_sequence (a bc de fg : ℕ) : Prop :=
  2 ∣ a ∧
  is_prime bc ∧
  5 ∣ de ∧
  3 ∣ fg ∧
  fg - de = de - bc ∧
  de - bc = bc - a

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a bc de fg : ℕ),
    is_valid_sequence a bc de fg ∧
    n = de * 100 + bc

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 2013 ∨ n = 4023 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3247_324766


namespace NUMINAMATH_CALUDE_average_weight_of_removed_carrots_l3247_324748

/-- The average weight of 4 removed carrots given the following conditions:
    - There are initially 20 carrots, 10 apples, and 5 oranges
    - The total initial weight is 8.70 kg
    - After removal, there are 16 carrots and 8 apples
    - The average weight after removal is 206 grams
    - The average weight of an apple is 210 grams -/
theorem average_weight_of_removed_carrots :
  ∀ (total_weight : ℝ) 
    (initial_carrots initial_apples initial_oranges : ℕ)
    (remaining_carrots remaining_apples : ℕ)
    (avg_weight_after_removal avg_weight_apple : ℝ),
  total_weight = 8.70 ∧
  initial_carrots = 20 ∧
  initial_apples = 10 ∧
  initial_oranges = 5 ∧
  remaining_carrots = 16 ∧
  remaining_apples = 8 ∧
  avg_weight_after_removal = 206 ∧
  avg_weight_apple = 210 →
  (total_weight * 1000 - 
   (remaining_carrots + remaining_apples) * avg_weight_after_removal - 
   (initial_apples - remaining_apples) * avg_weight_apple) / 
   (initial_carrots - remaining_carrots) = 834 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_removed_carrots_l3247_324748


namespace NUMINAMATH_CALUDE_gcd_operation_result_l3247_324725

theorem gcd_operation_result : (Nat.gcd 7350 165 - 15) * 3 = 0 := by sorry

end NUMINAMATH_CALUDE_gcd_operation_result_l3247_324725


namespace NUMINAMATH_CALUDE_number_pairs_theorem_l3247_324727

theorem number_pairs_theorem (a b : ℝ) :
  a^2 + b^2 = 15 * (a + b) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b)) →
  (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
by sorry

end NUMINAMATH_CALUDE_number_pairs_theorem_l3247_324727


namespace NUMINAMATH_CALUDE_four_intersections_implies_a_range_l3247_324708

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs x + a - 1

-- State the theorem
theorem four_intersections_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) →
  1 < a ∧ a < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_four_intersections_implies_a_range_l3247_324708


namespace NUMINAMATH_CALUDE_third_test_score_l3247_324796

def test1 : ℝ := 85
def test2 : ℝ := 79
def test4 : ℝ := 84
def test5 : ℝ := 85
def targetAverage : ℝ := 85
def numTests : ℕ := 5

theorem third_test_score (test3 : ℝ) : 
  (test1 + test2 + test3 + test4 + test5) / numTests = targetAverage → 
  test3 = 92 := by
sorry

end NUMINAMATH_CALUDE_third_test_score_l3247_324796


namespace NUMINAMATH_CALUDE_remaining_funds_is_38817_l3247_324743

/-- Represents the family's financial situation and tax obligations -/
structure FamilyFinances where
  father_income : ℕ
  mother_income : ℕ
  grandmother_pension : ℕ
  mikhail_scholarship : ℕ
  tax_deduction_per_child : ℕ
  num_children : ℕ
  income_tax_rate : ℚ
  monthly_savings : ℕ
  monthly_household_expenses : ℕ
  apartment_area : ℕ
  apartment_cadastral_value : ℕ
  car1_horsepower : ℕ
  car1_months_owned : ℕ
  car2_horsepower : ℕ
  car2_months_registered : ℕ
  land_area : ℕ
  land_cadastral_value : ℕ
  tour_cost_per_person : ℕ
  num_people_for_tour : ℕ

/-- Calculates the remaining funds for additional expenses -/
def calculate_remaining_funds (f : FamilyFinances) : ℕ :=
  sorry

/-- Theorem stating that the remaining funds for additional expenses is 38817 rubles -/
theorem remaining_funds_is_38817 (f : FamilyFinances) 
  (h1 : f.father_income = 50000)
  (h2 : f.mother_income = 28000)
  (h3 : f.grandmother_pension = 15000)
  (h4 : f.mikhail_scholarship = 3000)
  (h5 : f.tax_deduction_per_child = 1400)
  (h6 : f.num_children = 2)
  (h7 : f.income_tax_rate = 13 / 100)
  (h8 : f.monthly_savings = 10000)
  (h9 : f.monthly_household_expenses = 65000)
  (h10 : f.apartment_area = 78)
  (h11 : f.apartment_cadastral_value = 6240000)
  (h12 : f.car1_horsepower = 106)
  (h13 : f.car1_months_owned = 3)
  (h14 : f.car2_horsepower = 122)
  (h15 : f.car2_months_registered = 8)
  (h16 : f.land_area = 10)
  (h17 : f.land_cadastral_value = 420300)
  (h18 : f.tour_cost_per_person = 17900)
  (h19 : f.num_people_for_tour = 5) :
  calculate_remaining_funds f = 38817 :=
by sorry

end NUMINAMATH_CALUDE_remaining_funds_is_38817_l3247_324743


namespace NUMINAMATH_CALUDE_complex_number_real_l3247_324791

theorem complex_number_real (m : ℝ) :
  (m ≠ -5) →
  (∃ (z : ℂ), z = (m + 5)⁻¹ + (m^2 + 2*m - 15)*I ∧ z.im = 0) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_real_l3247_324791


namespace NUMINAMATH_CALUDE_total_cans_count_l3247_324742

-- Define the given conditions
def total_oil : ℕ := 290
def small_cans : ℕ := 10
def small_can_volume : ℕ := 8
def large_can_volume : ℕ := 15

-- State the theorem
theorem total_cans_count : 
  ∃ (large_cans : ℕ), 
    small_cans * small_can_volume + large_cans * large_can_volume = total_oil ∧
    small_cans + large_cans = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_count_l3247_324742


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3247_324739

theorem magnitude_of_z (z : ℂ) (h : z^2 = 24 - 32*I) : Complex.abs z = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3247_324739


namespace NUMINAMATH_CALUDE_compare_abc_l3247_324745

def tower_exp (base : ℕ) : ℕ → ℕ
| 0 => 1
| (n + 1) => base ^ (tower_exp base n)

def a : ℕ := tower_exp 3 25
def b : ℕ := tower_exp 4 20
def c : ℕ := 5^5

theorem compare_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_compare_abc_l3247_324745


namespace NUMINAMATH_CALUDE_percentage_increase_sum_l3247_324794

theorem percentage_increase_sum (A B C x y : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A = 120 → B = 110 → C = 100 →
  A = C * (1 + x / 100) →
  B = C * (1 + y / 100) →
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_sum_l3247_324794


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3247_324792

theorem complex_magnitude_problem (z : ℂ) : z = 2 / (1 - I) + I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3247_324792


namespace NUMINAMATH_CALUDE_integer_polynomial_roots_l3247_324730

/-- A polynomial with integer coefficients of the form x^4 + a₃x³ + a₂x² + a₁x - 27 = 0 -/
def IntegerPolynomial (a₃ a₂ a₁ : ℤ) (x : ℤ) : ℤ :=
  x^4 + a₃*x^3 + a₂*x^2 + a₁*x - 27

/-- The set of possible integer roots of the polynomial -/
def PossibleRoots : Set ℤ :=
  {-27, -9, -3, -1, 1, 3, 9, 27}

theorem integer_polynomial_roots (a₃ a₂ a₁ : ℤ) :
  ∀ x : ℤ, (IntegerPolynomial a₃ a₂ a₁ x = 0) ↔ x ∈ PossibleRoots :=
sorry

end NUMINAMATH_CALUDE_integer_polynomial_roots_l3247_324730


namespace NUMINAMATH_CALUDE_muffin_packs_per_case_l3247_324789

/-- Proves the number of packs per case for Nora's muffin sale -/
theorem muffin_packs_per_case 
  (total_amount : ℕ) 
  (price_per_muffin : ℕ) 
  (num_cases : ℕ) 
  (muffins_per_pack : ℕ) 
  (h1 : total_amount = 120)
  (h2 : price_per_muffin = 2)
  (h3 : num_cases = 5)
  (h4 : muffins_per_pack = 4) :
  (total_amount / price_per_muffin) / num_cases / muffins_per_pack = 3 := by
  sorry

#check muffin_packs_per_case

end NUMINAMATH_CALUDE_muffin_packs_per_case_l3247_324789


namespace NUMINAMATH_CALUDE_helen_amy_height_difference_l3247_324747

/-- Given the heights of Angela, Amy, and the height difference between Angela and Helen,
    prove that Helen is 3 cm taller than Amy. -/
theorem helen_amy_height_difference
  (angela_height : ℕ)
  (amy_height : ℕ)
  (angela_helen_diff : ℕ)
  (h1 : angela_height = 157)
  (h2 : amy_height = 150)
  (h3 : angela_height = angela_helen_diff + helen_height)
  (helen_height : ℕ) :
  helen_height - amy_height = 3 :=
sorry

end NUMINAMATH_CALUDE_helen_amy_height_difference_l3247_324747


namespace NUMINAMATH_CALUDE_sherry_banana_bread_l3247_324776

/-- Calculates the number of bananas needed for a given number of loaves -/
def bananas_needed (total_loaves : ℕ) (loaves_per_batch : ℕ) (bananas_per_batch : ℕ) : ℕ :=
  (total_loaves / loaves_per_batch) * bananas_per_batch

theorem sherry_banana_bread (total_loaves : ℕ) (loaves_per_batch : ℕ) (bananas_per_batch : ℕ) 
  (h1 : total_loaves = 99)
  (h2 : loaves_per_batch = 3)
  (h3 : bananas_per_batch = 1) :
  bananas_needed total_loaves loaves_per_batch bananas_per_batch = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_sherry_banana_bread_l3247_324776


namespace NUMINAMATH_CALUDE_super_ball_distance_l3247_324764

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let descentDistances := List.range (numBounces + 1) |>.map (fun i => initialHeight * bounceRatio^i)
  let ascentDistances := List.range numBounces |>.map (fun i => initialHeight * bounceRatio^(i + 1))
  (descentDistances.sum + ascentDistances.sum)

/-- Theorem: The total distance traveled by a ball dropped from 20 meters, 
    bouncing 5/8 of its previous height each time, and hitting the ground 4 times, 
    is 73.442078125 meters. -/
theorem super_ball_distance :
  totalDistance 20 (5/8) 4 = 73.442078125 := by
  sorry


end NUMINAMATH_CALUDE_super_ball_distance_l3247_324764


namespace NUMINAMATH_CALUDE_smallest_unbounded_population_l3247_324728

theorem smallest_unbounded_population : ∃ N : ℕ, N = 61 ∧ 
  (∀ m : ℕ, m < N → 2 * (m - 30) ≤ m) ∧ 
  (2 * (N - 30) > N) := by
  sorry

end NUMINAMATH_CALUDE_smallest_unbounded_population_l3247_324728


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_union_complement_B_A_l3247_324754

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- State the theorems
theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {x : ℝ | x ≥ 6 ∨ x < 3} := by sorry

theorem union_complement_B_A : 
  Bᶜ ∪ A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_union_complement_B_A_l3247_324754


namespace NUMINAMATH_CALUDE_face_masks_per_box_l3247_324724

/-- Proves the number of face masks in each box given the problem conditions --/
theorem face_masks_per_box :
  ∀ (num_boxes : ℕ) (sell_price : ℚ) (total_cost : ℚ) (total_profit : ℚ),
    num_boxes = 3 →
    sell_price = 1/2 →
    total_cost = 15 →
    total_profit = 15 →
    ∃ (masks_per_box : ℕ),
      masks_per_box = 20 ∧
      (num_boxes * masks_per_box : ℚ) * sell_price - total_cost = total_profit :=
by
  sorry


end NUMINAMATH_CALUDE_face_masks_per_box_l3247_324724


namespace NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3247_324780

theorem complex_equation_imaginary_part :
  ∀ (z : ℂ), (3 + 4*I) * z = 5 → z.im = -4/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3247_324780


namespace NUMINAMATH_CALUDE_mckenna_work_hours_l3247_324763

-- Define the start and end times of Mckenna's work day
def start_time : ℕ := 8
def office_end_time : ℕ := 11
def conference_end_time : ℕ := 13
def work_end_time : ℕ := conference_end_time + 2

-- Define the duration of each part of Mckenna's work day
def office_duration : ℕ := office_end_time - start_time
def conference_duration : ℕ := conference_end_time - office_end_time
def after_conference_duration : ℕ := 2

-- Theorem to prove
theorem mckenna_work_hours :
  office_duration + conference_duration + after_conference_duration = 7 := by
  sorry


end NUMINAMATH_CALUDE_mckenna_work_hours_l3247_324763


namespace NUMINAMATH_CALUDE_subset_proof_l3247_324771

def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem subset_proof : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_proof_l3247_324771


namespace NUMINAMATH_CALUDE_exercise_books_count_l3247_324790

/-- Given a shop with pencils, pens, and exercise books in the ratio 10 : 2 : 3,
    prove that if there are 120 pencils, then there are 36 exercise books. -/
theorem exercise_books_count (pencils pens books : ℕ) : 
  pencils = 120 →
  pencils / 10 = pens / 2 →
  pencils / 10 = books / 3 →
  books = 36 := by
sorry

end NUMINAMATH_CALUDE_exercise_books_count_l3247_324790


namespace NUMINAMATH_CALUDE_wendys_bake_sale_l3247_324702

/-- Wendy's bake sale problem -/
theorem wendys_bake_sale
  (cupcakes : ℕ)
  (cookies : ℕ)
  (leftover : ℕ)
  (h1 : cupcakes = 4)
  (h2 : cookies = 29)
  (h3 : leftover = 24) :
  cupcakes + cookies - leftover = 9 :=
by sorry

end NUMINAMATH_CALUDE_wendys_bake_sale_l3247_324702


namespace NUMINAMATH_CALUDE_equation_solution_l3247_324703

theorem equation_solution (t : ℝ) : 
  (Real.sqrt (3 * Real.sqrt (3 * t - 6)) = (8 - t) ^ (1/4)) ↔ 
  (t = (-43 + Real.sqrt 2321) / 2 ∨ t = (-43 - Real.sqrt 2321) / 2) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3247_324703


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l3247_324795

/-- Given polynomial functions f, g, h, and j, prove their sum equals 3x^2 + x - 2 -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => 2 * x^2 - 4 * x + 3
  let g := fun (x : ℝ) => -3 * x^2 + 7 * x - 6
  let h := fun (x : ℝ) => 3 * x^2 - 3 * x + 2
  let j := fun (x : ℝ) => x^2 + x - 1
  f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l3247_324795


namespace NUMINAMATH_CALUDE_absolute_value_subtraction_l3247_324735

theorem absolute_value_subtraction : 4 - |(-3)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_subtraction_l3247_324735


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3247_324749

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |f a b c x| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |2 * a * x + b| ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3247_324749


namespace NUMINAMATH_CALUDE_small_bottle_volume_proof_l3247_324798

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℝ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℝ := 2700

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℝ := 600

/-- The amount saved in pesetas by buying a big bottle instead of smaller bottles for the same volume -/
def savings : ℝ := 300

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℝ := 6

theorem small_bottle_volume_proof :
  small_bottle_volume * (big_bottle_cost / big_bottle_volume) =
  small_bottle_cost + (savings / big_bottle_volume) * small_bottle_volume :=
by sorry

end NUMINAMATH_CALUDE_small_bottle_volume_proof_l3247_324798


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l3247_324756

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + (2/3) * a 2 = 3 →
  a 4^2 = (1/9) * a 3 * a 7 →
  a 4 = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l3247_324756


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3247_324733

/-- Represents a hyperbola with equation x²/9 - y²/m = 1 -/
structure Hyperbola where
  m : ℝ

/-- Represents a line with equation x + y = 5 -/
def focus_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 5}

/-- Represents the asymptotes of a hyperbola -/
structure Asymptotes where
  slope : ℝ

theorem hyperbola_asymptotes (h : Hyperbola) (focus_on_line : ∃ p : ℝ × ℝ, p ∈ focus_line ∧ p.2 = 0) :
  Asymptotes.mk (4/3) = Asymptotes.mk (-4/3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3247_324733


namespace NUMINAMATH_CALUDE_spinner_probability_D_l3247_324736

-- Define the spinner with four regions
structure Spinner :=
  (A B C D : ℝ)

-- Define the properties of the spinner
def valid_spinner (s : Spinner) : Prop :=
  s.A = 1/4 ∧ s.B = 1/3 ∧ s.A + s.B + s.C + s.D = 1

-- Theorem statement
theorem spinner_probability_D (s : Spinner) 
  (h : valid_spinner s) : s.D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_D_l3247_324736


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3247_324715

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (other_salary : ℕ)
  (h1 : total_workers = 14)
  (h2 : technicians = 7)
  (h3 : technician_salary = 12000)
  (h4 : other_salary = 6000) :
  (technicians * technician_salary + (total_workers - technicians) * other_salary) / total_workers = 9000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3247_324715


namespace NUMINAMATH_CALUDE_inequality_solution_l3247_324779

theorem inequality_solution (x : ℝ) :
  x ≠ 1 ∧ x ≠ 2 →
  ((x^3 - 3*x^2 + 2*x) / (x^2 - 3*x + 2) ≤ 1 ↔ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3247_324779


namespace NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l3247_324753

/-- Given two vectors a and b that are parallel and α is an acute angle, prove that α = 45° -/
theorem parallel_vectors_acute_angle (α : Real) 
  (h_acute : 0 < α ∧ α < Real.pi / 2)
  (a : Fin 2 → Real)
  (b : Fin 2 → Real)
  (h_a : a = ![3/2, 1 + Real.sin α])
  (h_b : b = ![1 - Real.cos α, 1/3])
  (h_parallel : ∃ (k : Real), a = k • b) :
  α = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l3247_324753


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3247_324726

theorem simplify_and_rationalize : 
  (Real.sqrt 6 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 15) * (Real.sqrt 8 / Real.sqrt 14) = 
  (2 * Real.sqrt 7) / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3247_324726


namespace NUMINAMATH_CALUDE_cos_theta_value_l3247_324762

theorem cos_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi) : 
  Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_value_l3247_324762


namespace NUMINAMATH_CALUDE_current_speed_l3247_324772

/-- Given a boat's upstream and downstream speeds, calculate the speed of the current. -/
theorem current_speed (upstream_time : ℝ) (downstream_time : ℝ) :
  upstream_time = 20 →
  downstream_time = 9 →
  let upstream_speed := 60 / upstream_time
  let downstream_speed := 60 / downstream_time
  abs ((downstream_speed - upstream_speed) / 2 - 1.835) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3247_324772


namespace NUMINAMATH_CALUDE_combined_mean_l3247_324709

theorem combined_mean (set1_count : Nat) (set1_mean : ℝ) (set2_count : Nat) (set2_mean : ℝ) 
  (h1 : set1_count = 5)
  (h2 : set1_mean = 13)
  (h3 : set2_count = 6)
  (h4 : set2_mean = 24) :
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_mean_l3247_324709


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l3247_324784

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l3247_324784


namespace NUMINAMATH_CALUDE_problem_statement_l3247_324751

theorem problem_statement (p q r u v w : ℝ) 
  (eq1 : 17 * u + q * v + r * w = 0)
  (eq2 : p * u + 29 * v + r * w = 0)
  (eq3 : p * u + q * v + 56 * w = 0)
  (h1 : p ≠ 17)
  (h2 : u ≠ 0) :
  p / (p - 17) + q / (q - 29) + r / (r - 56) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3247_324751


namespace NUMINAMATH_CALUDE_ratio_problem_l3247_324718

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 1 / 4) :
  d / e = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3247_324718


namespace NUMINAMATH_CALUDE_complex_subtraction_l3247_324716

/-- Given complex numbers c and d, prove that c - 3d = 2 + 6i -/
theorem complex_subtraction (c d : ℂ) (hc : c = 5 + 3*I) (hd : d = 1 - I) :
  c - 3*d = 2 + 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3247_324716


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l3247_324793

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 250

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 50

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l3247_324793


namespace NUMINAMATH_CALUDE_return_speed_calculation_l3247_324777

/-- Given a round trip with the following conditions:
    - Total distance is 20 km (10 km each way)
    - Return speed is twice the outbound speed
    - Total travel time is 6 hours
    Prove that the return speed is 5 km/h -/
theorem return_speed_calculation (distance : ℝ) (total_time : ℝ) : 
  distance = 10 →
  total_time = 6 →
  ∃ (outbound_speed : ℝ),
    outbound_speed > 0 ∧
    distance / outbound_speed + distance / (2 * outbound_speed) = total_time →
    2 * outbound_speed = 5 := by
  sorry

#check return_speed_calculation

end NUMINAMATH_CALUDE_return_speed_calculation_l3247_324777


namespace NUMINAMATH_CALUDE_lower_limit_of_a_l3247_324721

theorem lower_limit_of_a (a b : ℤ) (h1 : a < 15) (h2 : b > 6) (h3 : b < 21)
  (h4 : (a : ℝ) / 7 - (a : ℝ) / 20 = 1.55) : a ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_of_a_l3247_324721


namespace NUMINAMATH_CALUDE_tractor_count_tractor_count_proof_l3247_324778

theorem tractor_count : ℝ → Prop :=
  fun T : ℝ =>
    let field_work : ℝ := T * 12
    let second_scenario_work : ℝ := 15 * 6.4
    (field_work = second_scenario_work) → T = 8

-- Proof
theorem tractor_count_proof : tractor_count 8 := by
  sorry

end NUMINAMATH_CALUDE_tractor_count_tractor_count_proof_l3247_324778


namespace NUMINAMATH_CALUDE_total_pictures_l3247_324744

theorem total_pictures (randy peter quincy : ℕ) : 
  randy = 5 → 
  peter = randy + 3 → 
  quincy = peter + 20 → 
  randy + peter + quincy = 41 := by sorry

end NUMINAMATH_CALUDE_total_pictures_l3247_324744


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l3247_324710

theorem cupcakes_remaining (total : ℕ) (given_away_fraction : ℚ) (eaten : ℕ) : 
  total = 60 → given_away_fraction = 4/5 → eaten = 3 →
  total * (1 - given_away_fraction) - eaten = 9 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l3247_324710


namespace NUMINAMATH_CALUDE_cubic_root_relation_l3247_324723

theorem cubic_root_relation (x₀ : ℝ) (z : ℝ) : 
  x₀^3 - x₀ - 1 = 0 →
  z = x₀^2 + 3 * x₀ + 1 →
  z^3 - 5*z^2 - 10*z - 11 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_relation_l3247_324723


namespace NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l3247_324769

theorem rectangle_to_square_area_ratio :
  let large_square_side : ℝ := 50
  let grid_size : ℕ := 5
  let rectangle_rows : ℕ := 2
  let rectangle_cols : ℕ := 3
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_side : ℝ := large_square_side / grid_size
  let rectangle_area : ℝ := (rectangle_rows * small_square_side) * (rectangle_cols * small_square_side)
  rectangle_area / large_square_area = 6 / 25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l3247_324769


namespace NUMINAMATH_CALUDE_consecutive_product_bound_l3247_324788

theorem consecutive_product_bound (π : Fin 90 → Fin 90) (h : Function.Bijective π) :
  ∃ i : Fin 89, (π i).val * (π (i + 1)).val ≥ 2014 ∨
    (π (Fin.last 89)).val * (π 0).val ≥ 2014 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_bound_l3247_324788


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l3247_324714

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem smallest_positive_period_of_f :
  is_smallest_positive_period f Real.pi := by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l3247_324714


namespace NUMINAMATH_CALUDE_vector_operation_result_l3247_324746

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

variable (O A B C E : E)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = A - E := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3247_324746


namespace NUMINAMATH_CALUDE_max_pads_purchase_existence_of_max_purchase_l3247_324701

def cost_pin : ℕ := 2
def cost_pen : ℕ := 3
def cost_pad : ℕ := 9
def total_budget : ℕ := 60

def is_valid_purchase (pins pens pads : ℕ) : Prop :=
  pins ≥ 1 ∧ pens ≥ 1 ∧ pads ≥ 1 ∧
  cost_pin * pins + cost_pen * pens + cost_pad * pads = total_budget

theorem max_pads_purchase :
  ∀ pins pens pads : ℕ, is_valid_purchase pins pens pads → pads ≤ 5 :=
by sorry

theorem existence_of_max_purchase :
  ∃ pins pens : ℕ, is_valid_purchase pins pens 5 :=
by sorry

end NUMINAMATH_CALUDE_max_pads_purchase_existence_of_max_purchase_l3247_324701


namespace NUMINAMATH_CALUDE_part_one_part_two_l3247_324700

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part 1
theorem part_one : 
  (A (-1) ∪ B = {x | x < 2 ∨ x > 5}) ∧ 
  ((Set.univ \ A (-1)) ∩ B = {x | x < -2 ∨ x > 5}) := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ (a ≥ 3 ∨ (-1/2 ≤ a ∧ a ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3247_324700


namespace NUMINAMATH_CALUDE_variance_scaling_l3247_324707

variable {n : ℕ}
variable (a : Fin n → ℝ)

/-- The variance of a dataset -/
def variance (x : Fin n → ℝ) : ℝ := sorry

/-- The scaled dataset where each element is multiplied by 2 -/
def scaled_data (x : Fin n → ℝ) : Fin n → ℝ := λ i => 2 * x i

theorem variance_scaling (h : variance a = 4) : 
  variance (scaled_data a) = 16 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l3247_324707


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_1_solve_quadratic_equation_2_l3247_324705

-- Problem 1
theorem solve_quadratic_equation_1 :
  ∀ x : ℝ, x^2 - 4*x = 5 ↔ x = 5 ∨ x = -1 :=
sorry

-- Problem 2
theorem solve_quadratic_equation_2 :
  ∀ x : ℝ, 2*x^2 - 3*x + 1 = 0 ↔ x = 1 ∨ x = 1/2 :=
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_1_solve_quadratic_equation_2_l3247_324705


namespace NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_l3247_324785

theorem right_triangle_with_inscribed_circle (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    b = a + d ∧
    c = b + d ∧
    a^2 + b^2 = c^2 ∧
    (a + b - c) / 2 = d :=
by
  sorry

#check right_triangle_with_inscribed_circle

end NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_l3247_324785


namespace NUMINAMATH_CALUDE_chessboard_coloring_theorem_l3247_324797

/-- Represents a coloring of a chessboard -/
def Coloring (n k : ℕ) := Fin (2*n) → Fin k → Fin n

/-- Checks if a coloring has a monochromatic rectangle -/
def has_monochromatic_rectangle (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (r₁ r₂ : Fin (2*n)) (c₁ c₂ : Fin k),
    r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
    c r₁ c₁ = c r₁ c₂ ∧ c r₁ c₁ = c r₂ c₁ ∧ c r₁ c₁ = c r₂ c₂

/-- The main theorem -/
theorem chessboard_coloring_theorem (n : ℕ) (h : n > 0) :
  ∀ k : ℕ, (k ≥ n*(2*n-1) + 1) →
    ∀ c : Coloring n k, has_monochromatic_rectangle n k c :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_theorem_l3247_324797


namespace NUMINAMATH_CALUDE_quadratic_roots_l3247_324755

theorem quadratic_roots (a b c : ℝ) (h : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁^3 - x₂^3 = 2011) :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ a * y₁^2 + 2 * b * y₁ + 4 * c = 0 ∧ a * y₂^2 + 2 * b * y₂ + 4 * c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3247_324755


namespace NUMINAMATH_CALUDE_converse_negation_equivalence_triangle_angles_arithmetic_sequence_inequality_system_not_equivalent_squared_inequality_implication_l3247_324704

-- 1. Converse and negation of a proposition
theorem converse_negation_equivalence (P Q : Prop) : 
  (P → Q) ↔ ¬Q → ¬P := by sorry

-- 2. Triangle angles forming arithmetic sequence
theorem triangle_angles_arithmetic_sequence (A B C : ℝ) :
  (A + B + C = 180) → (B = 60 ↔ 2 * B = A + C) := by sorry

-- 3. Inequality system counterexample
theorem inequality_system_not_equivalent :
  ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ ¬(x > 1 ∧ y > 2) := by sorry

-- 4. Squared inequality implication
theorem squared_inequality_implication (a b : ℝ) :
  (∀ m : ℝ, a * m^2 < b * m^2 → a < b) ∧
  ¬(∀ a b : ℝ, a < b → ∀ m : ℝ, a * m^2 < b * m^2) := by sorry

end NUMINAMATH_CALUDE_converse_negation_equivalence_triangle_angles_arithmetic_sequence_inequality_system_not_equivalent_squared_inequality_implication_l3247_324704


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3247_324775

/-- Represents the total number of schools of each type -/
structure SchoolCounts where
  universities : ℕ
  middleSchools : ℕ
  primarySchools : ℕ

/-- Calculates the total number of schools -/
def totalSchools (counts : SchoolCounts) : ℕ :=
  counts.universities + counts.middleSchools + counts.primarySchools

/-- Represents the sample size for middle schools -/
def middleSchoolSample : ℕ := 10

/-- Theorem: In a stratified sampling of schools, if 10 middle schools are sampled
    from a population with 20 universities, 200 middle schools, and 480 primary schools,
    then the total sample size is 35. -/
theorem stratified_sample_size 
  (counts : SchoolCounts) 
  (h1 : counts.universities = 20) 
  (h2 : counts.middleSchools = 200) 
  (h3 : counts.primarySchools = 480) :
  (middleSchoolSample : ℚ) / counts.middleSchools = 
  (35 : ℚ) / (totalSchools counts) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3247_324775


namespace NUMINAMATH_CALUDE_inequality_preservation_l3247_324786

theorem inequality_preservation (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3247_324786


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3247_324768

theorem min_value_theorem (a : ℝ) (h : a > 1) : (4 / (a - 1)) + a ≥ 6 := by
  sorry

theorem min_value_achieved (ε : ℝ) (h : ε > 0) : 
  ∃ a : ℝ, a > 1 ∧ (4 / (a - 1)) + a < 6 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3247_324768


namespace NUMINAMATH_CALUDE_book_distribution_count_correct_l3247_324752

/-- The number of ways to distribute 5 distinct books among 3 people,
    where one person receives 1 book and two people receive 2 books each. -/
def book_distribution_count : ℕ := 90

/-- Theorem stating that the number of book distribution methods is correct. -/
theorem book_distribution_count_correct :
  let n_books : ℕ := 5
  let n_people : ℕ := 3
  let books_per_person : List ℕ := [2, 2, 1]
  true → book_distribution_count = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_book_distribution_count_correct_l3247_324752


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l3247_324783

theorem greatest_common_multiple_under_120 : 
  ∀ n : ℕ, n < 120 → n % 10 = 0 → n % 15 = 0 → n ≤ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l3247_324783


namespace NUMINAMATH_CALUDE_valentines_given_away_l3247_324757

/-- Given Mrs. Franklin's initial and remaining Valentines, calculate how many she gave away. -/
theorem valentines_given_away
  (initial : ℝ)
  (remaining : ℝ)
  (h_initial : initial = 58.5)
  (h_remaining : remaining = 16.25) :
  initial - remaining = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_valentines_given_away_l3247_324757


namespace NUMINAMATH_CALUDE_weight_of_b_l3247_324720

/-- Given three weights a, b, and c, prove that b = 31 under certain conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →  -- average of a, b, and c is 45
  (a + b) / 2 = 40 →      -- average of a and b is 40
  (b + c) / 2 = 43 →      -- average of b and c is 43
  b = 31 := by
    sorry


end NUMINAMATH_CALUDE_weight_of_b_l3247_324720


namespace NUMINAMATH_CALUDE_bicycle_price_theorem_l3247_324711

/-- The price C pays for a bicycle, given the initial cost and two successive profit margins -/
def final_price (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : ℝ :=
  let price1 := initial_cost * (1 + profit1)
  price1 * (1 + profit2)

/-- Theorem stating that the final price of the bicycle is 225 -/
theorem bicycle_price_theorem :
  final_price 150 0.20 0.25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_theorem_l3247_324711


namespace NUMINAMATH_CALUDE_seven_by_seven_checkerboard_shading_l3247_324760

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a checkerboard shading pattern on a grid -/
def checkerboard_shading (g : Grid) : ℕ :=
  (g.size * g.size) / 2

/-- Calculates the percentage of shaded squares in a grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (checkerboard_shading g : ℚ) / (g.size * g.size : ℚ) * 100

/-- Theorem: The percentage of shaded squares in a 7x7 checkerboard is 2400/49 -/
theorem seven_by_seven_checkerboard_shading :
  shaded_percentage { size := 7 } = 2400 / 49 := by
  sorry

end NUMINAMATH_CALUDE_seven_by_seven_checkerboard_shading_l3247_324760


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3247_324706

/-- The focal length of an ellipse with equation x^2/25 + y^2/16 = 1 is 6 -/
theorem ellipse_focal_length : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 - b^2)
  2 * c = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l3247_324706


namespace NUMINAMATH_CALUDE_isaac_ribbon_length_l3247_324722

theorem isaac_ribbon_length :
  ∀ (total_parts : ℕ) (used_parts : ℕ) (unused_length : ℝ),
    total_parts = 6 →
    used_parts = 4 →
    unused_length = 10 →
    (unused_length / (total_parts - used_parts : ℝ)) * total_parts = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_isaac_ribbon_length_l3247_324722


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_eq_x_l3247_324787

theorem x_positive_sufficient_not_necessary_for_abs_x_eq_x :
  (∀ x : ℝ, x > 0 → |x| = x) ∧
  (∃ x : ℝ, |x| = x ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_eq_x_l3247_324787


namespace NUMINAMATH_CALUDE_complex_cube_real_l3247_324717

theorem complex_cube_real (a b : ℝ) (hb : b ≠ 0) 
  (h : ∃ (r : ℝ), (Complex.mk a b)^3 = r) : b^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_real_l3247_324717


namespace NUMINAMATH_CALUDE_divide_meter_into_hundred_parts_l3247_324750

theorem divide_meter_into_hundred_parts : 
  ∀ (total_length : ℝ) (num_parts : ℕ),
    total_length = 1 →
    num_parts = 100 →
    (total_length / num_parts : ℝ) = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_divide_meter_into_hundred_parts_l3247_324750


namespace NUMINAMATH_CALUDE_set_disjoint_iff_m_range_l3247_324758

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m < x ∧ x < m+1}

theorem set_disjoint_iff_m_range (m : ℝ) : 
  (∀ x ∈ A, x ∉ B m) ↔ m ∈ Set.Iic (-2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_set_disjoint_iff_m_range_l3247_324758


namespace NUMINAMATH_CALUDE_electric_guitars_sold_l3247_324781

theorem electric_guitars_sold (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (acoustic_price : ℕ) :
  total_guitars = 9 →
  total_revenue = 3611 →
  electric_price = 479 →
  acoustic_price = 339 →
  ∃ (electric_sold : ℕ) (acoustic_sold : ℕ),
    electric_sold + acoustic_sold = total_guitars ∧
    electric_sold * electric_price + acoustic_sold * acoustic_price = total_revenue ∧
    electric_sold = 4 :=
by sorry

end NUMINAMATH_CALUDE_electric_guitars_sold_l3247_324781


namespace NUMINAMATH_CALUDE_count_divisible_by_11_between_100_and_500_l3247_324759

def count_divisible (lower upper divisor : ℕ) : ℕ :=
  (upper / divisor - (lower - 1) / divisor)

theorem count_divisible_by_11_between_100_and_500 :
  count_divisible 100 500 11 = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_between_100_and_500_l3247_324759


namespace NUMINAMATH_CALUDE_max_tickets_with_budget_l3247_324712

theorem max_tickets_with_budget (ticket_price : ℚ) (budget : ℚ) (max_tickets : ℕ) : 
  ticket_price = 15 → budget = 120 → max_tickets = 8 → 
  (∀ n : ℕ, n * ticket_price ≤ budget ↔ n ≤ max_tickets) := by
sorry

end NUMINAMATH_CALUDE_max_tickets_with_budget_l3247_324712


namespace NUMINAMATH_CALUDE_smallest_y_value_l3247_324773

theorem smallest_y_value : ∃ y : ℝ, 
  (∀ z : ℝ, 3 * z^2 + 27 * z - 90 = z * (z + 15) → y ≤ z) ∧ 
  (3 * y^2 + 27 * y - 90 = y * (y + 15)) ∧ 
  y = -9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_value_l3247_324773


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l3247_324761

theorem coefficient_x5_in_expansion : 
  let n : ℕ := 9
  let k : ℕ := 4
  let a : ℝ := 3 * Real.sqrt 2
  (Nat.choose n k) * a^k = 40824 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l3247_324761


namespace NUMINAMATH_CALUDE_application_schemes_five_graduates_three_universities_l3247_324734

/-- The number of possible application schemes for high school graduates choosing universities. -/
def application_schemes (num_graduates : ℕ) (num_universities : ℕ) : ℕ :=
  num_universities ^ num_graduates

/-- Theorem: The number of possible application schemes for 5 high school graduates
    choosing from 3 universities, where each graduate can only fill in one preference,
    is equal to 3^5. -/
theorem application_schemes_five_graduates_three_universities :
  application_schemes 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_application_schemes_five_graduates_three_universities_l3247_324734


namespace NUMINAMATH_CALUDE_triangle_side_relation_l3247_324740

theorem triangle_side_relation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a^2 - 16*b^2 - c^2 + 6*a*b + 10*b*c = 0) :
  a + c = 2*b := by sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l3247_324740


namespace NUMINAMATH_CALUDE_weighted_cauchy_schwarz_l3247_324738

theorem weighted_cauchy_schwarz (p q x y : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q < 1) : 
  (p * x + q * y)^2 ≤ p * x^2 + q * y^2 := by
  sorry

end NUMINAMATH_CALUDE_weighted_cauchy_schwarz_l3247_324738


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l3247_324799

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 11 = 0 → n ≤ m :=
by
  use 33
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l3247_324799


namespace NUMINAMATH_CALUDE_largest_prime_factor_l3247_324765

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^4 + 2 * 17^2 + 1 - 16^4) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (17^4 + 2 * 17^2 + 1 - 16^4) → q ≤ p) →
  (∃ p : ℕ, p = 17 ∧ Nat.Prime p ∧ p ∣ (17^4 + 2 * 17^2 + 1 - 16^4) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (17^4 + 2 * 17^2 + 1 - 16^4) → q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l3247_324765


namespace NUMINAMATH_CALUDE_red_card_events_mutually_exclusive_not_opposite_l3247_324713

-- Define the set of cards
inductive Card : Type
  | Black : Card
  | Red : Card
  | White : Card

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsRed (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem statement
theorem red_card_events_mutually_exclusive_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsRed d)) ∧
  -- The events are not opposite (i.e., it's possible for neither to occur)
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsRed d) :=
sorry

end NUMINAMATH_CALUDE_red_card_events_mutually_exclusive_not_opposite_l3247_324713
