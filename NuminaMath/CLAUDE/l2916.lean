import Mathlib

namespace one_third_of_seven_times_nine_l2916_291638

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l2916_291638


namespace two_thousand_plus_sqrt_two_thousand_one_in_A_l2916_291610

-- Define the set A
variable (A : Set ℝ)

-- Define the conditions
axiom one_in_A : 1 ∈ A
axiom square_in_A : ∀ x : ℝ, x ∈ A → x^2 ∈ A
axiom inverse_square_in_A : ∀ x : ℝ, (x^2 - 4*x + 4) ∈ A → x ∈ A

-- State the theorem
theorem two_thousand_plus_sqrt_two_thousand_one_in_A :
  (2000 + Real.sqrt 2001) ∈ A := by
  sorry

end two_thousand_plus_sqrt_two_thousand_one_in_A_l2916_291610


namespace potato_cost_proof_l2916_291615

/-- The initial cost of one bag of potatoes in rubles -/
def initial_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_cost_proof :
  (bags_bought * initial_cost * andrey_increase) +
  earnings_difference =
  (boris_first_sale * initial_cost * boris_first_increase) +
  (boris_second_sale * initial_cost * boris_first_increase * boris_second_increase) :=
by sorry

end potato_cost_proof_l2916_291615


namespace ends_with_two_zeros_l2916_291667

theorem ends_with_two_zeros (x y : ℕ) :
  (x^2 + x*y + y^2) % 10 = 0 → (x^2 + x*y + y^2) % 100 = 0 :=
by sorry

end ends_with_two_zeros_l2916_291667


namespace stating_prob_reach_heaven_l2916_291678

/-- A point in the 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The starting point of the walk -/
def start : LatticePoint := ⟨1, 1⟩

/-- Predicate for heaven points -/
def is_heaven (p : LatticePoint) : Prop :=
  ∃ m n : ℤ, p.x = 6 * m ∧ p.y = 6 * n

/-- Predicate for hell points -/
def is_hell (p : LatticePoint) : Prop :=
  ∃ m n : ℤ, p.x = 6 * m + 3 ∧ p.y = 6 * n + 3

/-- The probability of reaching heaven -/
def prob_heaven : ℚ := 13 / 22

/-- 
Theorem stating that the probability of reaching heaven 
before hell in a random lattice walk starting from (1,1) is 13/22 
-/
theorem prob_reach_heaven : 
  prob_heaven = 13 / 22 :=
sorry

end stating_prob_reach_heaven_l2916_291678


namespace platform_length_l2916_291611

/-- Given a train of length 900 m that takes 39 sec to cross a platform and 18 sec to cross a signal pole, the length of the platform is 1050 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 900)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 18) :
  train_length + (train_length / time_cross_pole * time_cross_platform) - train_length = 1050 :=
by sorry

end platform_length_l2916_291611


namespace survey_optimism_l2916_291637

theorem survey_optimism (a b c : ℕ) (m n : ℤ) : 
  a + b + c = 100 →
  m = a + b / 2 →
  n = a - c →
  m = 40 →
  n = -20 :=
by sorry

end survey_optimism_l2916_291637


namespace lucy_lovely_age_problem_l2916_291604

theorem lucy_lovely_age_problem (lucy_age : ℕ) (lovely_age : ℕ) (years_until_twice : ℕ) : 
  lucy_age = 50 →
  lucy_age - 5 = 3 * (lovely_age - 5) →
  lucy_age + years_until_twice = 2 * (lovely_age + years_until_twice) →
  years_until_twice = 10 := by
sorry

end lucy_lovely_age_problem_l2916_291604


namespace hat_problem_probabilities_q_div_p_undefined_l2916_291653

/-- The number of slips in the hat -/
def total_slips : ℕ := 42

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 14

/-- The number of slips for each number -/
def slips_per_number : ℕ := 3

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing four slips with the same number -/
def p : ℚ := 0

/-- The number of ways to choose two distinct numbers and two slips for each -/
def favorable_outcomes : ℕ := Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2

/-- The probability of drawing two pairs of slips with different numbers -/
def q : ℚ := favorable_outcomes / Nat.choose total_slips drawn_slips

theorem hat_problem_probabilities :
  p = 0 ∧ q = 819 / Nat.choose total_slips drawn_slips :=
sorry

theorem q_div_p_undefined : ¬∃ (x : ℚ), q / p = x :=
sorry

end hat_problem_probabilities_q_div_p_undefined_l2916_291653


namespace problem_solution_l2916_291690

noncomputable def f (x : ℝ) := |Real.log x|

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  (a * b = 1) ∧ 
  ((a + b) / 2 > 1) ∧ 
  (∃ b₀ : ℝ, 3 < b₀ ∧ b₀ < 4 ∧ 1 / b₀^2 + b₀^2 + 2 - 4 * b₀ = 0) := by
  sorry

end problem_solution_l2916_291690


namespace book_cost_calculation_l2916_291697

/-- Calculates the cost of each book given the total customers, return rate, and total sales after returns. -/
theorem book_cost_calculation (total_customers : ℕ) (return_rate : ℚ) (total_sales : ℚ) : 
  total_customers = 1000 → 
  return_rate = 37 / 100 → 
  total_sales = 9450 → 
  (total_sales / (total_customers * (1 - return_rate))) = 15 := by
  sorry

end book_cost_calculation_l2916_291697


namespace unique_solution_condition_l2916_291627

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = k + 3 * x) ↔ k = 5 := by
  sorry

end unique_solution_condition_l2916_291627


namespace max_pies_without_ingredients_l2916_291644

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (blueberry_fraction raspberry_fraction blackberry_fraction walnut_fraction : ℚ)
  (h_total : total_pies = 30)
  (h_blueberry : blueberry_fraction = 1/3)
  (h_raspberry : raspberry_fraction = 3/5)
  (h_blackberry : blackberry_fraction = 5/6)
  (h_walnut : walnut_fraction = 1/10) :
  ∃ (max_without_ingredients : ℕ), 
    max_without_ingredients ≤ total_pies ∧
    max_without_ingredients = total_pies - (total_pies * blackberry_fraction).floor ∧
    max_without_ingredients = 5 :=
by sorry

end max_pies_without_ingredients_l2916_291644


namespace triangle_at_most_one_obtuse_angle_proof_by_contradiction_uses_correct_assumption_l2916_291666

/-- A triangle has at most one obtuse angle -/
theorem triangle_at_most_one_obtuse_angle : 
  ∀ (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop),
  (∀ t : T, is_triangle t → 
    ∃! a : T, is_obtuse_angle t a) :=
by
  sorry

/-- The correct assumption for proof by contradiction of the above theorem -/
def contradiction_assumption (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop) : Prop :=
  ∃ t : T, is_triangle t ∧ ∃ a b : T, a ≠ b ∧ is_obtuse_angle t a ∧ is_obtuse_angle t b

/-- The proof by contradiction uses the correct assumption -/
theorem proof_by_contradiction_uses_correct_assumption :
  ∀ (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop),
  ¬(contradiction_assumption T is_triangle is_obtuse_angle) →
  (∀ t : T, is_triangle t → 
    ∃! a : T, is_obtuse_angle t a) :=
by
  sorry

end triangle_at_most_one_obtuse_angle_proof_by_contradiction_uses_correct_assumption_l2916_291666


namespace triangle_inequality_l2916_291693

/-- Theorem: For any triangle ABC, the sum of square roots of specific ratios involving side lengths, altitude, and inradius is less than or equal to 3/4. -/
theorem triangle_inequality (a b c h_a r : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_h_a : 0 < h_a) (h_pos_r : 0 < r) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let f (x y z w v) := Real.sqrt (x * (w - 2 * v) / ((3 * x + y + z) * (w + 2 * v)))
  (f a b c h_a r) + (f b c a h_a r) + (f c a b h_a r) ≤ 3 / 4 := by
  sorry

end triangle_inequality_l2916_291693


namespace min_fencing_length_proof_l2916_291605

/-- The minimum length of bamboo fencing needed to enclose a rectangular flower bed -/
def min_fencing_length : ℝ := 20

/-- The area of the rectangular flower bed -/
def flower_bed_area : ℝ := 50

theorem min_fencing_length_proof :
  ∀ (length width : ℝ),
  length > 0 →
  width > 0 →
  length * width = flower_bed_area →
  length + 2 * width ≥ min_fencing_length :=
by
  sorry

#check min_fencing_length_proof

end min_fencing_length_proof_l2916_291605


namespace max_area_inscribed_isosceles_triangle_l2916_291694

/-- An isosceles triangle inscribed in a circle --/
structure InscribedIsoscelesTriangle where
  /-- The radius of the circle --/
  radius : ℝ
  /-- The height of the triangle to its base --/
  height : ℝ

/-- The area of an inscribed isosceles triangle --/
def area (t : InscribedIsoscelesTriangle) : ℝ := sorry

/-- Theorem: The area of an isosceles triangle inscribed in a circle with radius 6
    is maximized when the height to the base is 9 --/
theorem max_area_inscribed_isosceles_triangle :
  ∀ t : InscribedIsoscelesTriangle,
  t.radius = 6 →
  area t ≤ area { radius := 6, height := 9 } :=
sorry

end max_area_inscribed_isosceles_triangle_l2916_291694


namespace sqrt_of_sqrt_81_l2916_291630

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by
  sorry

end sqrt_of_sqrt_81_l2916_291630


namespace sum_product_inequality_cubic_inequality_l2916_291639

-- Part 1
theorem sum_product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := by
sorry

-- Part 2
theorem cubic_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
sorry

end sum_product_inequality_cubic_inequality_l2916_291639


namespace union_of_intervals_l2916_291669

open Set

theorem union_of_intervals (A B : Set ℝ) :
  A = Ioc (-1) 1 → B = Ioo 0 2 → A ∪ B = Ioo (-1) 2 := by
  sorry

end union_of_intervals_l2916_291669


namespace sum_palindromic_primes_l2916_291699

def isPrime (n : Nat) : Bool := sorry

def reverseDigits (n : Nat) : Nat := sorry

def isPalindromicPrime (n : Nat) : Bool :=
  isPrime n ∧ isPrime (reverseDigits n)

def palindromicPrimes : List Nat :=
  (List.range 90).filter (fun n => n ≥ 10 ∧ isPalindromicPrime n)

theorem sum_palindromic_primes :
  palindromicPrimes.sum = 429 := by sorry

end sum_palindromic_primes_l2916_291699


namespace chairs_to_hall_l2916_291659

theorem chairs_to_hall (num_students : ℕ) (chairs_per_trip : ℕ) (num_trips : ℕ) :
  num_students = 5 → chairs_per_trip = 5 → num_trips = 10 →
  num_students * chairs_per_trip * num_trips = 250 := by
  sorry

end chairs_to_hall_l2916_291659


namespace divisor_and_equation_solution_l2916_291681

theorem divisor_and_equation_solution :
  ∃ (k : ℕ) (base : ℕ+),
    (929260 : ℕ) % (base : ℕ)^k = 0 ∧
    3^k - k^3 = 1 ∧
    base = 17 ∧
    k = 4 := by
  sorry

end divisor_and_equation_solution_l2916_291681


namespace isosceles_triangle_perimeter_l2916_291671

/-- An isosceles triangle with two given side lengths -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side1_pos : side1 > 0
  side2_pos : side2 > 0

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : Set ℝ :=
  if t.side1 = t.side2 then
    {2 * t.side1 + t.side2}
  else
    {2 * t.side1 + t.side2, t.side1 + 2 * t.side2}

theorem isosceles_triangle_perimeter :
  ∀ (t : IsoscelesTriangle),
    (t.side1 = 4 ∧ t.side2 = 6) ∨ (t.side1 = 6 ∧ t.side2 = 4) →
      perimeter t = {14, 16} ∧
    (t.side1 = 2 ∧ t.side2 = 6) ∨ (t.side1 = 6 ∧ t.side2 = 2) →
      perimeter t = {14} :=
by sorry

end isosceles_triangle_perimeter_l2916_291671


namespace chocolate_difference_l2916_291687

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℚ := 3 / 7 * 70

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℚ := 120 / 100 * 40

/-- The number of chocolates Penny ate -/
def penny_chocolates : ℚ := 3 / 8 * 80

/-- The number of chocolates Dime ate -/
def dime_chocolates : ℚ := 1 / 2 * 90

/-- The difference between the number of chocolates eaten by Robert and Nickel combined
    and the number of chocolates eaten by Penny and Dime combined -/
theorem chocolate_difference :
  (robert_chocolates + nickel_chocolates) - (penny_chocolates + dime_chocolates) = -3 := by
  sorry

end chocolate_difference_l2916_291687


namespace complex_fraction_sum_l2916_291650

theorem complex_fraction_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a * b ≠ a^3) :
  let sum := (a^2 - b^2) / (a * b) + (a * b + b^2) / (a * b - a^3)
  sum ≠ 1 ∧ sum ≠ (b^2 + b) / (b - a^2) ∧ sum ≠ 0 ∧ sum ≠ (a^2 + b) / (a^2 - b) :=
by sorry

end complex_fraction_sum_l2916_291650


namespace mitch_max_boat_length_l2916_291674

/-- The maximum length of boat Mitch can buy given his savings and expenses --/
def max_boat_length (savings : ℚ) (cost_per_foot : ℚ) (license_fee : ℚ) : ℚ :=
  let docking_fee := 3 * license_fee
  let total_fees := license_fee + docking_fee
  let remaining_money := savings - total_fees
  remaining_money / cost_per_foot

/-- Theorem stating the maximum length of boat Mitch can buy --/
theorem mitch_max_boat_length :
  max_boat_length 20000 1500 500 = 12 := by
sorry

end mitch_max_boat_length_l2916_291674


namespace bridge_length_l2916_291618

/-- The length of a bridge given specific train and crossing conditions -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 125 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 250 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end bridge_length_l2916_291618


namespace plane_air_time_l2916_291619

/-- Proves that the time the plane spent in the air is 10/3 hours given the problem conditions. -/
theorem plane_air_time (total_distance : ℝ) (icebreaker_speed : ℝ) (plane_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 840)
  (h2 : icebreaker_speed = 20)
  (h3 : plane_speed = 120)
  (h4 : total_time = 22) :
  (total_distance - icebreaker_speed * total_time) / plane_speed = 10 / 3 := by
  sorry

#check plane_air_time

end plane_air_time_l2916_291619


namespace odot_equation_solution_l2916_291624

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem odot_equation_solution (g : ℝ) (h1 : g ≥ 0) (h2 : odot 4 g = 20) : g = 12 := by
  sorry

end odot_equation_solution_l2916_291624


namespace berry_picking_difference_l2916_291622

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  dima_basket_ratio : ℚ
  sergei_basket_ratio : ℚ
  dima_speed_multiplier : ℕ

/-- Calculates the difference in berries placed in the basket between Dima and Sergei -/
def berry_difference (scenario : BerryPicking) : ℕ :=
  sorry

/-- The main theorem stating the difference in berries placed in the basket -/
theorem berry_picking_difference (scenario : BerryPicking) 
  (h1 : scenario.total_berries = 450)
  (h2 : scenario.dima_basket_ratio = 1/2)
  (h3 : scenario.sergei_basket_ratio = 2/3)
  (h4 : scenario.dima_speed_multiplier = 2) :
  berry_difference scenario = 50 :=
sorry

end berry_picking_difference_l2916_291622


namespace sqrt_x4_minus_x2_l2916_291682

theorem sqrt_x4_minus_x2 (x : ℝ) : Real.sqrt (x^4 - x^2) = |x| * Real.sqrt (x^2 - 1) := by
  sorry

end sqrt_x4_minus_x2_l2916_291682


namespace estimate_three_plus_sqrt_ten_l2916_291612

theorem estimate_three_plus_sqrt_ten : 6 < 3 + Real.sqrt 10 ∧ 3 + Real.sqrt 10 < 7 := by
  sorry

end estimate_three_plus_sqrt_ten_l2916_291612


namespace coefficient_x5_in_expansion_l2916_291661

theorem coefficient_x5_in_expansion : 
  (Finset.range 61).sum (fun k => Nat.choose 60 k * (1 : ℕ)^(60 - k) * (1 : ℕ)^k) = 2^60 ∧ 
  Nat.choose 60 5 = 446040 :=
by sorry

end coefficient_x5_in_expansion_l2916_291661


namespace triple_a_student_distribution_l2916_291643

theorem triple_a_student_distribution (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 36 :=
sorry

end triple_a_student_distribution_l2916_291643


namespace sum_of_rectangle_areas_l2916_291635

/-- The number of available squares -/
def n : ℕ := 9

/-- The side length of each square in cm -/
def side_length : ℝ := 1

/-- The set of possible widths for rectangles -/
def possible_widths : Finset ℕ := Finset.range n

/-- The set of possible heights for rectangles -/
def possible_heights : Finset ℕ := Finset.range n

/-- The area of a rectangle with given width and height -/
def rectangle_area (w h : ℕ) : ℝ := (w : ℝ) * (h : ℝ) * side_length ^ 2

/-- The set of all valid rectangles (width, height) that can be formed -/
def valid_rectangles : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 * p.2 ≤ n) (possible_widths.product possible_heights)

/-- The sum of areas of all distinct rectangles -/
def sum_of_areas : ℝ := Finset.sum valid_rectangles (fun p => rectangle_area p.1 p.2)

theorem sum_of_rectangle_areas :
  sum_of_areas = 72 := by sorry

end sum_of_rectangle_areas_l2916_291635


namespace original_number_proof_l2916_291621

theorem original_number_proof (w : ℝ) : 
  (w + 0.125 * w) - (w - 0.25 * w) = 30 → w = 80 := by
  sorry

end original_number_proof_l2916_291621


namespace triangle_area_l2916_291668

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l2916_291668


namespace plane_through_points_l2916_291616

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ × ℝ := (a, 0, 0)
def B (b : ℝ) : ℝ × ℝ × ℝ := (0, b, 0)
def C (c : ℝ) : ℝ × ℝ × ℝ := (0, 0, c)

-- Define the plane equation
def plane_equation (a b c x y z : ℝ) : Prop :=
  x / a + y / b + z / c = 1

-- Theorem statement
theorem plane_through_points (a b c : ℝ) (h : a * b * c ≠ 0) :
  ∃ (f : ℝ × ℝ × ℝ → Prop),
    (∀ x y z, f (x, y, z) ↔ plane_equation a b c x y z) ∧
    f (A a) ∧ f (B b) ∧ f (C c) :=
sorry

end plane_through_points_l2916_291616


namespace evaluate_expression_l2916_291646

theorem evaluate_expression (a b : ℤ) (ha : a = 3) (hb : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end evaluate_expression_l2916_291646


namespace intersection_points_form_geometric_sequence_l2916_291670

-- Define the curve C
def curve (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x

-- Define the line l
def line (t : ℝ) : ℝ × ℝ := (-2 + t, -4 + t)

-- Define the point P
def P : ℝ × ℝ := (-2, -4)

-- Define the property of geometric sequence for three positive real numbers
def is_geometric_sequence (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ b^2 = a*c

-- Main theorem
theorem intersection_points_form_geometric_sequence (a : ℝ) :
  a > 0 →
  ∃ t₁ t₂ : ℝ,
    let M := line t₁
    let N := line t₂
    curve a M.1 M.2 ∧
    curve a N.1 N.2 ∧
    is_geometric_sequence (Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2))
                          (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2))
                          (Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)) →
  a = 1 := by
  sorry


end intersection_points_form_geometric_sequence_l2916_291670


namespace digit_swap_difference_l2916_291636

theorem digit_swap_difference (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  ∃ k : ℤ, (100 * a + 10 * b + c) - (10 * a + 100 * b + c) = 90 * k :=
sorry

end digit_swap_difference_l2916_291636


namespace test_questions_count_l2916_291645

theorem test_questions_count : ∀ (total : ℕ), 
  (total % 5 = 0) →  -- The test has 5 equal sections
  (32 : ℚ) / total > (70 : ℚ) / 100 →  -- Percentage of correct answers > 70%
  (32 : ℚ) / total < (77 : ℚ) / 100 →  -- Percentage of correct answers < 77%
  total = 45 :=
by
  sorry

end test_questions_count_l2916_291645


namespace product_of_numbers_with_given_sum_and_difference_l2916_291603

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 → x - y = 10 → x * y = 875 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l2916_291603


namespace leftover_value_is_fifteen_l2916_291601

def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 60
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

def james_quarters : ℕ := 97
def james_dimes : ℕ := 178
def lindsay_quarters : ℕ := 143
def lindsay_dimes : ℕ := 292

def total_quarters : ℕ := james_quarters + lindsay_quarters
def total_dimes : ℕ := james_dimes + lindsay_dimes

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll

def leftover_value : ℚ := leftover_quarters * quarter_value + leftover_dimes * dime_value

theorem leftover_value_is_fifteen :
  leftover_value = 15 := by sorry

end leftover_value_is_fifteen_l2916_291601


namespace inequality_proof_l2916_291632

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 2*c)) + (b / (c + 2*a)) + (c / (a + 2*b)) > 1/2 := by
  sorry

end inequality_proof_l2916_291632


namespace k_range_theorem_l2916_291642

/-- The range of k given the conditions in the problem -/
def k_range : Set ℝ := Set.Iic 0 ∪ Set.Ioo (1/2) (5/2)

/-- p: the function y=kx+1 is increasing on ℝ -/
def p (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1

/-- q: the equation x^2+(2k-3)x+1=0 has real solutions -/
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2*k - 3)*x + 1 = 0

/-- Main theorem stating the range of k -/
theorem k_range_theorem (h1 : ∀ k : ℝ, ¬(p k ∧ q k)) (h2 : ∀ k : ℝ, p k ∨ q k) : 
  ∀ k : ℝ, k ∈ k_range ↔ (p k ∨ q k) :=
sorry

end k_range_theorem_l2916_291642


namespace S_a_is_three_rays_with_common_point_l2916_291664

/-- The set S_a for a positive integer a -/
def S_a (a : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (a = p.1 + 2 ∧ p.2 - 4 ≤ a) ∨
    (a = p.2 - 4 ∧ p.1 + 2 ≤ a) ∨
    (p.1 + 2 = p.2 - 4 ∧ a ≤ p.1 + 2)}

/-- The common point of the three rays -/
def common_point (a : ℕ) : ℝ × ℝ := (a - 2, a + 4)

/-- The three rays that form S_a -/
def ray1 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a - 2 ∧ p.2 ≤ a + 4}
def ray2 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a + 4 ∧ p.1 ≤ a - 2}
def ray3 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 6 ∧ p.1 ≥ a - 2}

/-- Theorem stating that S_a is the union of three rays with a common point -/
theorem S_a_is_three_rays_with_common_point (a : ℕ) :
  S_a a = ray1 a ∪ ray2 a ∪ ray3 a ∧
  common_point a ∈ ray1 a ∧
  common_point a ∈ ray2 a ∧
  common_point a ∈ ray3 a :=
sorry

end S_a_is_three_rays_with_common_point_l2916_291664


namespace building_height_l2916_291652

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves that the height of the building is 22 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 55)
  : (flagpole_height / flagpole_shadow) * building_shadow = 22 :=
by sorry

end building_height_l2916_291652


namespace total_revenue_is_1168_l2916_291607

/-- Calculates the total revenue from apple and orange sales given the following conditions:
  * 50 boxes of apples and 30 boxes of oranges on Saturday
  * 25 boxes of apples and 15 boxes of oranges on Sunday
  * 10 apples in each apple box
  * 8 oranges in each orange box
  * Each apple sold for $1.20
  * Each orange sold for $0.80
  * Total of 720 apples and 380 oranges sold on Saturday and Sunday -/
def total_revenue : ℝ :=
  let apple_boxes_saturday : ℕ := 50
  let orange_boxes_saturday : ℕ := 30
  let apple_boxes_sunday : ℕ := 25
  let orange_boxes_sunday : ℕ := 15
  let apples_per_box : ℕ := 10
  let oranges_per_box : ℕ := 8
  let apple_price : ℝ := 1.20
  let orange_price : ℝ := 0.80
  let total_apples_sold : ℕ := 720
  let total_oranges_sold : ℕ := 380
  let apple_revenue : ℝ := (total_apples_sold : ℝ) * apple_price
  let orange_revenue : ℝ := (total_oranges_sold : ℝ) * orange_price
  apple_revenue + orange_revenue

/-- Theorem stating that the total revenue is $1168 -/
theorem total_revenue_is_1168 : total_revenue = 1168 := by
  sorry

end total_revenue_is_1168_l2916_291607


namespace square_division_l2916_291631

/-- A square can be divided into two equal parts in at least four different ways. -/
theorem square_division (s : ℝ) (h : s > 0) :
  ∃ (rect1 rect2 : ℝ × ℝ) (tri1 tri2 : ℝ × ℝ × ℝ),
    -- Vertical division
    rect1 = (s, s/2) ∧
    -- Horizontal division
    rect2 = (s/2, s) ∧
    -- Diagonal division (top-left to bottom-right)
    tri1 = (s, s, Real.sqrt 2 * s) ∧
    -- Diagonal division (top-right to bottom-left)
    tri2 = (s, s, Real.sqrt 2 * s) ∧
    -- All divisions result in equal areas
    s * (s/2) = (s/2) * s ∧
    s * (s/2) = (1/2) * s * s ∧
    -- All divisions are valid (non-negative dimensions)
    s > 0 ∧ s/2 > 0 ∧ Real.sqrt 2 * s > 0 :=
by
  sorry

end square_division_l2916_291631


namespace sticker_enlargement_l2916_291647

/-- Given a rectangle with original width and height, and a new width,
    calculate the new height when enlarged proportionately -/
def new_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem stating that a 3x2 inch rectangle enlarged to 12 inches wide
    will be 8 inches tall -/
theorem sticker_enlargement :
  new_height 3 2 12 = 8 := by sorry

end sticker_enlargement_l2916_291647


namespace melted_ice_cream_height_l2916_291614

/-- The height of a melted ice cream scoop -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h →
  h = 1 / 4 := by sorry

end melted_ice_cream_height_l2916_291614


namespace inscribed_square_area_l2916_291634

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 12

/-- The square inscribed in the region bound by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  sideLength : ℝ  -- side length of the square
  top_on_parabola : parabola (center + sideLength/2) = sideLength
  bottom_on_xaxis : center - sideLength/2 ≥ 0

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.sideLength^2 = 24 - 8*Real.sqrt 5 := by
  sorry

end inscribed_square_area_l2916_291634


namespace hot_air_balloon_balloons_l2916_291626

theorem hot_air_balloon_balloons (initial_balloons : ℕ) : 
  (initial_balloons : ℚ) * (2 / 5) = 80 → initial_balloons = 200 :=
by
  sorry

#check hot_air_balloon_balloons

end hot_air_balloon_balloons_l2916_291626


namespace number_difference_l2916_291655

theorem number_difference (x y : ℝ) (sum_eq : x + y = 42) (prod_eq : x * y = 437) :
  |x - y| = 4 := by
sorry

end number_difference_l2916_291655


namespace product_evaluation_l2916_291672

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end product_evaluation_l2916_291672


namespace savings_percentage_approx_l2916_291648

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 5650
def savings : ℕ := 2350

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

def percentage_saved : ℚ := (savings : ℚ) / (total_salary : ℚ) * 100

theorem savings_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ abs (percentage_saved - 8.87) < ε :=
sorry

end savings_percentage_approx_l2916_291648


namespace solution_set_when_a_is_4_range_of_a_for_all_x_geq_4_l2916_291656

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_all_x_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) ↔ (a ≤ -3 ∨ a ≥ 5) := by sorry

end solution_set_when_a_is_4_range_of_a_for_all_x_geq_4_l2916_291656


namespace max_abs_quadratic_function_l2916_291625

theorem max_abs_quadratic_function (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (|f 0| ≤ 2) → (|f 2| ≤ 2) → (|f (-2)| ≤ 2) →
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, |f x| ≤ 5/2 :=
by sorry

end max_abs_quadratic_function_l2916_291625


namespace coefficient_x_squared_in_expansion_l2916_291660

theorem coefficient_x_squared_in_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k : ℕ) * 2^k * (if k = 2 then 1 else 0)) = 40 := by
  sorry

end coefficient_x_squared_in_expansion_l2916_291660


namespace min_values_constraint_l2916_291629

theorem min_values_constraint (x y z : ℝ) (h : x - 2*y + z = 4) :
  (∀ a b c : ℝ, a - 2*b + c = 4 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (∀ a b c : ℝ, a - 2*b + c = 4 → x^2 + (y - 1)^2 + z^2 ≤ a^2 + (b - 1)^2 + c^2) ∧
  (∃ a b c : ℝ, a - 2*b + c = 4 ∧ a^2 + b^2 + c^2 = 8/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 4 ∧ a^2 + (b - 1)^2 + c^2 = 6) := by
  sorry

end min_values_constraint_l2916_291629


namespace surface_area_after_removing_corners_l2916_291609

/-- Represents the dimensions of a cube in centimeters -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (d : CubeDimensions) : ℝ :=
  6 * d.length * d.width

/-- Represents the problem setup -/
structure CubeWithCornersRemoved where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions

/-- The main theorem to be proved -/
theorem surface_area_after_removing_corners
  (c : CubeWithCornersRemoved)
  (h1 : c.originalCube.length = 4)
  (h2 : c.originalCube.width = 4)
  (h3 : c.originalCube.height = 4)
  (h4 : c.cornerCube.length = 2)
  (h5 : c.cornerCube.width = 2)
  (h6 : c.cornerCube.height = 2) :
  surfaceArea c.originalCube = 96 := by
  sorry

end surface_area_after_removing_corners_l2916_291609


namespace reflection_line_sum_l2916_291657

/-- Given a line y = mx + c, if the reflection of point (-2, 0) across this line is (6, 4), then m + c = 4 -/
theorem reflection_line_sum (m c : ℝ) : 
  (∀ (x y : ℝ), y = m * x + c → 
    (x + 2) * (x - 6) + (y - 4) * (y - 0) = 0 ∧ 
    (x - 2) = m * (y - 2)) → 
  m + c = 4 := by sorry

end reflection_line_sum_l2916_291657


namespace function_properties_l2916_291663

-- Define the function f
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem function_properties
  (a b c : ℝ)
  (h_min : ∀ x : ℝ, f a b c x ≥ 0 ∧ ∃ y : ℝ, f a b c y = 0)
  (h_sym : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1))
  (h_bound : ∀ x ∈ Set.Ioo 0 5, x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1) :
  (f a b c 1 = 1) ∧
  (∀ x : ℝ, f a b c x = (1/4) * (x + 1)^2) ∧
  (∃ m : ℝ, m > 1 ∧ 
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) ∧
    (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    m = 9) :=
by sorry

end function_properties_l2916_291663


namespace inscribed_semicircle_radius_l2916_291662

/-- Given a right triangle ABC with AC = 12 and BC = 5, the radius of the inscribed semicircle is 10/3 -/
theorem inscribed_semicircle_radius (A B C : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A C = 12) →
  (d B C = 5) →
  (d A B)^2 = (d A C)^2 + (d B C)^2 →
  (∃ r : ℝ, r = 10/3 ∧ 
    ∃ O : ℝ × ℝ, 
      d O A + d O B = d A B ∧
      d O C = r ∧
      ∀ P : ℝ × ℝ, d O P = r → 
        (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 ∧
        (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1)) := by
  sorry

end inscribed_semicircle_radius_l2916_291662


namespace inverse_of_21_mod_47_l2916_291675

theorem inverse_of_21_mod_47 (h : (8⁻¹ : ZMod 47) = 6) : (21⁻¹ : ZMod 47) = 38 := by
  sorry

end inverse_of_21_mod_47_l2916_291675


namespace stating_speed_ratio_equals_one_plus_head_start_l2916_291695

/-- The ratio of runner A's speed to runner B's speed in a race where A gives B a head start -/
def speed_ratio : ℝ := 1.11764705882352941

/-- The fraction of the race length that runner A gives as a head start to runner B -/
def head_start : ℝ := 0.11764705882352941

/-- 
Theorem stating that the speed ratio of runner A to runner B is equal to 1 plus the head start fraction,
given that the race ends in a dead heat when A gives B the specified head start.
-/
theorem speed_ratio_equals_one_plus_head_start : 
  speed_ratio = 1 + head_start := by sorry

end stating_speed_ratio_equals_one_plus_head_start_l2916_291695


namespace parabola_equation_from_distances_l2916_291617

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 2 * C.p * y

/-- Theorem: If a point on the parabola is 8 units from the focus and 6 units from the x-axis,
    then the parabola's equation is x^2 = 8y -/
theorem parabola_equation_from_distances (C : Parabola) (P : PointOnParabola C)
    (h_focus : Real.sqrt ((P.x)^2 + (P.y - C.p/2)^2) = 8)
    (h_xaxis : P.y = 6) :
    C.p = 4 ∧ ∀ (x y : ℝ), x^2 = 2 * C.p * y ↔ x^2 = 8 * y := by
  sorry

end parabola_equation_from_distances_l2916_291617


namespace least_multiple_with_digit_product_multiple_l2916_291685

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns true if n is a multiple of m -/
def isMultipleOf (n m : ℕ) : Prop := sorry

/-- Returns true if n is the least number satisfying the given property -/
def isLeast (n : ℕ) (property : ℕ → Prop) : Prop := sorry

theorem least_multiple_with_digit_product_multiple :
  isLeast 315 (λ n : ℕ => isMultipleOf n 15 ∧ 
                          n > 0 ∧ 
                          isMultipleOf (digitProduct n) 15 ∧ 
                          digitProduct n > 0) := by
  sorry

end least_multiple_with_digit_product_multiple_l2916_291685


namespace divisibility_problem_l2916_291620

theorem divisibility_problem (a : ℝ) : 
  (∃ k : ℤ, 2 * 10^10 + a = 11 * k) → 
  0 ≤ a → 
  a < 11 → 
  a = 9 := by sorry

end divisibility_problem_l2916_291620


namespace no_base_for_perfect_square_l2916_291613

theorem no_base_for_perfect_square : ¬ ∃ (b : ℕ), ∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end no_base_for_perfect_square_l2916_291613


namespace custom_op_two_five_l2916_291628

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

-- State the theorem
theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end custom_op_two_five_l2916_291628


namespace sector_max_area_l2916_291683

theorem sector_max_area (R c : ℝ) (h : c > 0) :
  let perimeter := 2 * R + R * (c / R - 2)
  let area := (1 / 2) * R * (c / R - 2) * R
  ∀ R > 0, perimeter = c → area ≤ c^2 / 16 :=
sorry

end sector_max_area_l2916_291683


namespace total_sweets_l2916_291676

theorem total_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) 
  (h1 : num_crates = 4) 
  (h2 : sweets_per_crate = 16) : 
  num_crates * sweets_per_crate = 64 := by
  sorry

end total_sweets_l2916_291676


namespace expression_evaluation_expression_simplification_l2916_291649

-- Part 1
theorem expression_evaluation :
  Real.sqrt 2 + (1 : ℝ)^2014 + 2 * Real.cos (45 * π / 180) + Real.sqrt 16 = 2 * Real.sqrt 2 + 5 := by
  sorry

-- Part 2
theorem expression_simplification (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  (x^2 + y^2 - 2*x*y) / (x - y) / ((x / y) - (y / x)) = Real.sqrt 2 / 4 := by
  sorry

end expression_evaluation_expression_simplification_l2916_291649


namespace bobs_weight_l2916_291677

theorem bobs_weight (jim_weight bob_weight : ℝ) 
  (h1 : jim_weight + bob_weight = 200)
  (h2 : bob_weight - jim_weight = bob_weight / 3) : 
  bob_weight = 120 := by
sorry

end bobs_weight_l2916_291677


namespace condition_equivalent_to_inequality_l2916_291691

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem condition_equivalent_to_inequality
  (f : ℝ → ℝ) (h : IncreasingFunction f) :
  (∀ a b : ℝ, a + b > 0 ↔ f a + f b > f (-a) + f (-b)) :=
by sorry

end condition_equivalent_to_inequality_l2916_291691


namespace equation_substitution_l2916_291689

theorem equation_substitution :
  ∀ x y : ℝ,
  (y = x + 1) →
  (3 * x - y = 18) →
  (3 * x - x - 1 = 18) :=
by
  sorry

end equation_substitution_l2916_291689


namespace iron_cars_count_l2916_291679

/-- Represents the initial state and rules for a train delivery problem -/
structure TrainProblem where
  coal_cars : ℕ
  wood_cars : ℕ
  station_distance : ℕ
  travel_time : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  total_delivery_time : ℕ

/-- Calculates the number of iron cars given a TrainProblem -/
def calculate_iron_cars (problem : TrainProblem) : ℕ :=
  let num_stations := problem.total_delivery_time / problem.travel_time
  num_stations * problem.max_iron_deposit

/-- Theorem stating that for the given problem, the number of iron cars is 12 -/
theorem iron_cars_count (problem : TrainProblem) 
  (h1 : problem.coal_cars = 6)
  (h2 : problem.wood_cars = 2)
  (h3 : problem.station_distance = 6)
  (h4 : problem.travel_time = 25)
  (h5 : problem.max_coal_deposit = 2)
  (h6 : problem.max_iron_deposit = 3)
  (h7 : problem.max_wood_deposit = 1)
  (h8 : problem.total_delivery_time = 100) :
  calculate_iron_cars problem = 12 := by
  sorry

end iron_cars_count_l2916_291679


namespace quadratic_inequality_solution_set_real_line_l2916_291600

/-- The solution set of a quadratic inequality is the entire real line -/
theorem quadratic_inequality_solution_set_real_line 
  (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_inequality_solution_set_real_line_l2916_291600


namespace optimal_plan_maximizes_profit_l2916_291684

/-- Represents the production plan for transformers --/
structure ProductionPlan where
  typeA : ℕ
  typeB : ℕ

/-- Calculates the profit for a given production plan --/
def profit (plan : ProductionPlan) : ℕ :=
  12 * plan.typeA + 10 * plan.typeB

/-- Checks if a production plan is feasible given the resource constraints --/
def isFeasible (plan : ProductionPlan) : Prop :=
  5 * plan.typeA + 3 * plan.typeB ≤ 481 ∧
  3 * plan.typeA + 2 * plan.typeB ≤ 301

/-- The optimal production plan --/
def optimalPlan : ProductionPlan :=
  { typeA := 1, typeB := 149 }

/-- Theorem stating that the optimal plan achieves the maximum profit --/
theorem optimal_plan_maximizes_profit :
  isFeasible optimalPlan ∧
  ∀ plan, isFeasible plan → profit plan ≤ profit optimalPlan :=
by sorry

#eval profit optimalPlan  -- Should output 1502

end optimal_plan_maximizes_profit_l2916_291684


namespace bruce_bank_savings_l2916_291673

/-- The amount of money Bruce puts in the bank given his birthday gifts -/
def money_in_bank (aunt_gift : ℕ) (grandfather_gift : ℕ) : ℕ :=
  (aunt_gift + grandfather_gift) / 5

/-- Theorem stating that Bruce puts $45 in the bank -/
theorem bruce_bank_savings : money_in_bank 75 150 = 45 := by
  sorry

end bruce_bank_savings_l2916_291673


namespace triangle_cosine_inequality_l2916_291686

theorem triangle_cosine_inequality (A B C : Real) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0)
  (h_triangle : A + B + C = Real.pi) : 
  (Real.cos A)^2 / (Real.cos B)^2 + 
  (Real.cos B)^2 / (Real.cos C)^2 + 
  (Real.cos C)^2 / (Real.cos A)^2 ≥ 
  4 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2) := by
  sorry

end triangle_cosine_inequality_l2916_291686


namespace max_stamps_per_page_l2916_291623

theorem max_stamps_per_page (album1 album2 album3 : ℕ) 
  (h1 : album1 = 945)
  (h2 : album2 = 1260)
  (h3 : album3 = 1575) :
  Nat.gcd album1 (Nat.gcd album2 album3) = 315 :=
by sorry

end max_stamps_per_page_l2916_291623


namespace derivative_of_f_l2916_291651

/-- Given a function f(x) = (x^2 + 2x - 1)e^(2-x), this theorem states its derivative. -/
theorem derivative_of_f (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x^2 + 2*x - 1) * Real.exp (2 - x)
  deriv f x = (3 - x^2) * Real.exp (2 - x) := by
sorry

end derivative_of_f_l2916_291651


namespace max_xyz_value_l2916_291680

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + 2 * z = (x + z) * (y + z)) :
  x * y * z ≤ 8 / 27 := by
sorry

end max_xyz_value_l2916_291680


namespace orange_banana_ratio_l2916_291606

/-- Proves that the ratio of oranges to bananas is 2:1 given the problem conditions --/
theorem orange_banana_ratio :
  ∀ (orange_price pear_price banana_price : ℚ),
  pear_price - orange_price = banana_price →
  orange_price + pear_price = 120 →
  pear_price = 90 →
  200 * banana_price + (24000 - 200 * banana_price) / orange_price = 400 →
  (24000 - 200 * banana_price) / orange_price / 200 = 2 :=
by
  sorry

#check orange_banana_ratio

end orange_banana_ratio_l2916_291606


namespace pure_imaginary_ratio_l2916_291640

theorem pure_imaginary_ratio (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (c + d * Complex.I) = y * Complex.I) : 
  c / d = -4 / 3 := by
  sorry

end pure_imaginary_ratio_l2916_291640


namespace tessa_initial_apples_l2916_291658

theorem tessa_initial_apples :
  ∀ (initial_apples : ℕ),
    (initial_apples + 5 = 9) →
    initial_apples = 4 := by
  sorry

end tessa_initial_apples_l2916_291658


namespace meaningful_expression_l2916_291665

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 3)) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) := by
  sorry

end meaningful_expression_l2916_291665


namespace inequality_for_positive_integers_l2916_291698

theorem inequality_for_positive_integers (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1)^2 := by
  sorry

end inequality_for_positive_integers_l2916_291698


namespace property_price_calculation_l2916_291688

/-- The price of the property in dollars given the price per square foot, house size, and barn size. -/
def property_price (price_per_sqft : ℚ) (house_size : ℚ) (barn_size : ℚ) : ℚ :=
  price_per_sqft * (house_size + barn_size)

/-- Theorem stating that the property price is $333,200 given the specified conditions. -/
theorem property_price_calculation :
  property_price 98 2400 1000 = 333200 := by
  sorry

end property_price_calculation_l2916_291688


namespace sheila_mon_wed_fri_hours_l2916_291641

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating the number of hours Sheila works on Monday, Wednesday, and Friday --/
theorem sheila_mon_wed_fri_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_tue_thu = 6 * 2)
  (h2 : schedule.weekly_earnings = 360)
  (h3 : schedule.hourly_rate = 10)
  (h4 : schedule.weekly_earnings = schedule.hourly_rate * (schedule.hours_mon_wed_fri + schedule.hours_tue_thu)) :
  schedule.hours_mon_wed_fri = 24 := by
  sorry

#check sheila_mon_wed_fri_hours

end sheila_mon_wed_fri_hours_l2916_291641


namespace increasing_quadratic_max_value_function_inequality_positive_reals_l2916_291602

-- Statement 1
theorem increasing_quadratic (x : ℝ) (h : x > 0) :
  Monotone (fun x => 2 * x^2 + x + 1) := by sorry

-- Statement 2
theorem max_value_function (x : ℝ) (h : x > 0) :
  (2 - 3*x - 4/x) ≤ (2 - 4*Real.sqrt 3) := by sorry

-- Statement 3
theorem inequality_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z := by sorry

end increasing_quadratic_max_value_function_inequality_positive_reals_l2916_291602


namespace quadratic_solution_set_l2916_291696

/-- A quadratic function f(x) = ax^2 - bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 - b * x + 1

theorem quadratic_solution_set (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 1/4 < x ∧ x < 1/3) →
  a = 12 ∧ b = 7 := by
  sorry

end quadratic_solution_set_l2916_291696


namespace expression_factorization_l2916_291608

theorem expression_factorization (a : ℝ) :
  (6 * a^3 + 92 * a^2 - 7) - (-7 * a^3 + a^2 - 7) = 13 * a^2 * (a + 7) := by
  sorry

end expression_factorization_l2916_291608


namespace sum_of_numbers_in_ratio_l2916_291692

theorem sum_of_numbers_in_ratio (x : ℝ) :
  x > 0 →
  x^2 + (2*x)^2 + (5*x)^2 = 4320 →
  x + 2*x + 5*x = 96 := by
sorry

end sum_of_numbers_in_ratio_l2916_291692


namespace number_line_expressions_l2916_291633

theorem number_line_expressions (P Q R S T : ℝ) 
  (hP : P > 3 ∧ P < 4)
  (hQ : Q > 1 ∧ Q < 1.2)
  (hR : R > -0.2 ∧ R < 0)
  (hS : S > 0.8 ∧ S < 1)
  (hT : T > 1.4 ∧ T < 1.6) :
  R / (P * Q) < 0 ∧ (S + T) / R < 0 ∧ P - Q ≥ 0 ∧ P * Q ≥ 0 ∧ (S / Q) * P ≥ 0 := by
  sorry

end number_line_expressions_l2916_291633


namespace martha_and_john_money_l2916_291654

theorem martha_and_john_money : (5 / 8 : ℚ) + (2 / 5 : ℚ) = 1.025 := by sorry

end martha_and_john_money_l2916_291654
