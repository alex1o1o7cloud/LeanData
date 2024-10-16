import Mathlib

namespace NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l1851_185104

/-- Given a machine that fills boxes at a constant rate, 
    this theorem proves how many boxes it can fill in 5 minutes. -/
theorem boxes_filled_in_five_minutes 
  (boxes_per_hour : ℚ) 
  (h1 : boxes_per_hour = 24 / 60) : 
  boxes_per_hour * 5 = 2 := by
  sorry

#check boxes_filled_in_five_minutes

end NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l1851_185104


namespace NUMINAMATH_CALUDE_range_of_t_for_right_angle_l1851_185126

/-- The theorem stating the range of t for point M(3,t) given the conditions -/
theorem range_of_t_for_right_angle (t : ℝ) : 
  let M : ℝ × ℝ := (3, t)
  let O : ℝ × ℝ := (0, 0)
  let circle_O := {(x, y) : ℝ × ℝ | x^2 + y^2 = 6}
  ∃ (A B : ℝ × ℝ), A ∈ circle_O ∧ B ∈ circle_O ∧ 
    ((M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2) = 0) →
  -Real.sqrt 3 ≤ t ∧ t ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_for_right_angle_l1851_185126


namespace NUMINAMATH_CALUDE_expected_waiting_time_for_last_suitcase_l1851_185106

theorem expected_waiting_time_for_last_suitcase 
  (total_suitcases : ℕ) 
  (business_suitcases : ℕ) 
  (placement_interval : ℕ) 
  (h1 : total_suitcases = 200) 
  (h2 : business_suitcases = 10) 
  (h3 : placement_interval = 2) :
  (((total_suitcases + 1) * placement_interval * business_suitcases) / (business_suitcases + 1) : ℚ) = 4020 / 11 := by
  sorry

#check expected_waiting_time_for_last_suitcase

end NUMINAMATH_CALUDE_expected_waiting_time_for_last_suitcase_l1851_185106


namespace NUMINAMATH_CALUDE_cycling_distance_conversion_l1851_185142

/-- Converts a list of digits in base 9 to a number in base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

/-- The cycling distance in base 9 -/
def cyclingDistanceBase9 : List Nat := [3, 6, 1, 8]

theorem cycling_distance_conversion :
  base9ToBase10 cyclingDistanceBase9 = 2690 := by
  sorry

end NUMINAMATH_CALUDE_cycling_distance_conversion_l1851_185142


namespace NUMINAMATH_CALUDE_friday_fries_ratio_l1851_185132

/-- Represents the number of fries sold -/
structure FriesSold where
  total : ℕ
  small : ℕ

/-- Calculates the ratio of large fries to small fries -/
def largeToSmallRatio (fs : FriesSold) : ℚ :=
  (fs.total - fs.small : ℚ) / fs.small

theorem friday_fries_ratio :
  let fs : FriesSold := { total := 24, small := 4 }
  largeToSmallRatio fs = 5 := by
  sorry

end NUMINAMATH_CALUDE_friday_fries_ratio_l1851_185132


namespace NUMINAMATH_CALUDE_oplus_four_two_l1851_185144

-- Define the operation ⊕ for real numbers
def oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem oplus_four_two : oplus 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_oplus_four_two_l1851_185144


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_f_l1851_185130

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem sum_of_max_and_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               (M + m = 2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_f_l1851_185130


namespace NUMINAMATH_CALUDE_prob_genuine_given_equal_weights_l1851_185177

/-- Represents a bag of coins -/
structure CoinBag where
  total : ℕ
  genuine : ℕ
  counterfeit : ℕ

/-- Represents the result of selecting coins -/
inductive Selection
  | AllGenuine
  | Mixed
  | AllCounterfeit

/-- Calculates the probability of selecting all genuine coins -/
def prob_all_genuine (bag : CoinBag) : ℚ :=
  (bag.genuine.choose 2 : ℚ) * ((bag.genuine - 2).choose 2 : ℚ) /
  ((bag.total.choose 2 : ℚ) * ((bag.total - 2).choose 2 : ℚ))

/-- Calculates the probability of equal weights -/
def prob_equal_weights (bag : CoinBag) : ℚ :=
  sorry  -- Actual calculation would go here

/-- The main theorem to prove -/
theorem prob_genuine_given_equal_weights (bag : CoinBag) 
  (h1 : bag.total = 12)
  (h2 : bag.genuine = 9)
  (h3 : bag.counterfeit = 3) :
  prob_all_genuine bag / prob_equal_weights bag = 42 / 165 := by
  sorry

end NUMINAMATH_CALUDE_prob_genuine_given_equal_weights_l1851_185177


namespace NUMINAMATH_CALUDE_sock_time_correct_l1851_185154

/-- Represents the time (in hours) to knit each sock -/
def sock_time : ℝ := 1.5

/-- Represents the number of grandchildren -/
def num_grandchildren : ℕ := 3

/-- Time (in hours) to knit a hat -/
def hat_time : ℝ := 2

/-- Time (in hours) to knit a scarf -/
def scarf_time : ℝ := 3

/-- Time (in hours) to knit a sweater -/
def sweater_time : ℝ := 6

/-- Time (in hours) to knit each mitten -/
def mitten_time : ℝ := 1

/-- Total time (in hours) to knit all outfits -/
def total_time : ℝ := 48

/-- Theorem stating that the calculated sock_time satisfies the given conditions -/
theorem sock_time_correct : 
  num_grandchildren * (hat_time + scarf_time + sweater_time + 2 * mitten_time + 2 * sock_time) = total_time := by
  sorry

end NUMINAMATH_CALUDE_sock_time_correct_l1851_185154


namespace NUMINAMATH_CALUDE_rectangle_width_problem_l1851_185111

theorem rectangle_width_problem (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 75 →
  width = 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_problem_l1851_185111


namespace NUMINAMATH_CALUDE_total_jumps_l1851_185193

theorem total_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 :=
by sorry

end NUMINAMATH_CALUDE_total_jumps_l1851_185193


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l1851_185148

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l1851_185148


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l1851_185186

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 7

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_to_add : ℕ := total_flour - added_flour

theorem mary_flour_calculation :
  flour_to_add = 5 := by sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l1851_185186


namespace NUMINAMATH_CALUDE_mia_gift_spending_l1851_185172

def christmas_gift_problem (num_siblings : ℕ) (sibling_gift_cost : ℚ) (total_spent : ℚ) : ℚ :=
  let total_spent_on_siblings := num_siblings * sibling_gift_cost
  let remaining_for_parents := total_spent - total_spent_on_siblings
  remaining_for_parents / 2

theorem mia_gift_spending :
  christmas_gift_problem 3 30 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mia_gift_spending_l1851_185172


namespace NUMINAMATH_CALUDE_perp_lines_parallel_perp_planes_parallel_l1851_185121

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (linePerpToPlane : Line → Plane → Prop)
variable (planePerpToLine : Plane → Line → Prop)

-- Axioms
axiom distinct_lines (a b : Line) : a ≠ b
axiom distinct_planes (α β : Plane) : α ≠ β

-- Theorem 1
theorem perp_lines_parallel (a b : Line) (α : Plane) :
  linePerpToPlane a α → linePerpToPlane b α → parallelLines a b :=
sorry

-- Theorem 2
theorem perp_planes_parallel (a : Line) (α β : Plane) :
  planePerpToLine α a → planePerpToLine β a → parallelPlanes α β :=
sorry

end NUMINAMATH_CALUDE_perp_lines_parallel_perp_planes_parallel_l1851_185121


namespace NUMINAMATH_CALUDE_probability_of_four_ones_in_twelve_dice_l1851_185127

def number_of_dice : ℕ := 12
def sides_per_die : ℕ := 6
def desired_ones : ℕ := 4

def probability_of_one : ℚ := 1 / sides_per_die
def probability_of_not_one : ℚ := 1 - probability_of_one

def binomial_coefficient (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

def probability_exact_ones : ℚ :=
  (binomial_coefficient number_of_dice desired_ones : ℚ) *
  (probability_of_one ^ desired_ones) *
  (probability_of_not_one ^ (number_of_dice - desired_ones))

theorem probability_of_four_ones_in_twelve_dice :
  probability_exact_ones = 495 * 390625 / 2176782336 :=
sorry

-- The following line is to show the approximate decimal value
#eval (495 * 390625 : ℚ) / 2176782336

end NUMINAMATH_CALUDE_probability_of_four_ones_in_twelve_dice_l1851_185127


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1851_185175

/-- Given two quadratic equations, where the roots of the first are three less than the roots of the second, 
    this theorem proves that the constant term of the first equation is -14.5 -/
theorem quadratic_roots_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 2*y^2 - 11*y - 14 = 0 ∧ x = y - 3) →
  c = -14.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1851_185175


namespace NUMINAMATH_CALUDE_birth_interval_is_3_7_l1851_185117

/-- Represents the ages of 5 children -/
structure ChildrenAges where
  ages : Fin 5 → ℕ
  sum_65 : ages 0 + ages 1 + ages 2 + ages 3 + ages 4 = 65
  youngest_7 : ages 0 = 7

/-- The interval between births, assuming equal spacing -/
def birthInterval (c : ChildrenAges) : ℚ :=
  ((c.ages 4 - c.ages 0) : ℚ) / 4

/-- Theorem stating the birth interval is 3.7 years -/
theorem birth_interval_is_3_7 (c : ChildrenAges) : birthInterval c = 37/10 := by
  sorry

end NUMINAMATH_CALUDE_birth_interval_is_3_7_l1851_185117


namespace NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1851_185147

theorem square_sum_ge_twice_product {x y : ℝ} (h : x ≥ y) : x^2 + y^2 ≥ 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1851_185147


namespace NUMINAMATH_CALUDE_equation_to_parabola_l1851_185124

/-- The equation y^4 - 16x^2 = 2y^2 - 64 can be transformed into a parabolic form -/
theorem equation_to_parabola :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    y^4 - 16*x^2 = 2*y^2 - 64 →
    ∃ (t : ℝ), y^2 = a*x + b*t + c :=
by sorry

end NUMINAMATH_CALUDE_equation_to_parabola_l1851_185124


namespace NUMINAMATH_CALUDE_class_test_percentages_l1851_185141

theorem class_test_percentages (total : ℝ) (first : ℝ) (second : ℝ) (both : ℝ) 
  (h_total : total = 100)
  (h_first : first = 75)
  (h_second : second = 30)
  (h_both : both = 25) :
  total - (first + second - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_percentages_l1851_185141


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l1851_185139

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 878 / 1000)
  : (total_bananas - (total_oranges + total_bananas - 
     (good_fruits_percentage * (total_oranges + total_bananas)).floor - 
     (rotten_oranges_percentage * total_oranges).floor)) / total_bananas = 8 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l1851_185139


namespace NUMINAMATH_CALUDE_tammy_orange_earnings_l1851_185176

/-- Calculates Tammy's earnings from selling oranges over 3 weeks --/
def tammys_earnings (num_trees : ℕ) (oranges_per_tree : ℕ) (oranges_per_pack : ℕ) 
  (price_per_pack : ℚ) (days : ℕ) : ℚ :=
  let oranges_per_day := num_trees * oranges_per_tree
  let packs_per_day := oranges_per_day / oranges_per_pack
  let packs_in_period := packs_per_day * days
  packs_in_period * price_per_pack

/-- Proves that Tammy's earnings after 3 weeks will be $840 --/
theorem tammy_orange_earnings : 
  tammys_earnings 10 12 6 2 21 = 840 := by sorry

end NUMINAMATH_CALUDE_tammy_orange_earnings_l1851_185176


namespace NUMINAMATH_CALUDE_work_hours_theorem_l1851_185190

/-- Calculates the total hours worked given the number of days and hours per day -/
def total_hours (days : ℝ) (hours_per_day : ℝ) : ℝ :=
  days * hours_per_day

/-- Proves that working 2 hours per day for 4 days results in 8 total hours -/
theorem work_hours_theorem :
  let days : ℝ := 4
  let hours_per_day : ℝ := 2
  total_hours days hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_theorem_l1851_185190


namespace NUMINAMATH_CALUDE_log_problem_l1851_185197

theorem log_problem (x : ℝ) : 
  x = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) →
  Real.log x / Real.log 7 = -2 * Real.log 2 / Real.log 7 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l1851_185197


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_two_l1851_185108

theorem negation_of_forall_greater_than_two :
  (¬ (∀ x : ℝ, x > 2)) ↔ (∃ x : ℝ, x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_two_l1851_185108


namespace NUMINAMATH_CALUDE_incorrect_copy_difference_l1851_185134

theorem incorrect_copy_difference (square : ℝ) : 
  let x := 4 * (square - 3)
  let y := 4 * square - 3
  x - y = -9 := by sorry

end NUMINAMATH_CALUDE_incorrect_copy_difference_l1851_185134


namespace NUMINAMATH_CALUDE_second_company_base_rate_l1851_185151

/-- United Telephone's base rate -/
def united_base_rate : ℝ := 9

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Second company's per-minute rate -/
def second_per_minute : ℝ := 0.20

/-- Number of minutes for equal billing -/
def equal_minutes : ℝ := 60

/-- Second company's base rate -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + equal_minutes * united_per_minute =
  second_base_rate + equal_minutes * second_per_minute :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l1851_185151


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_product_l1851_185164

theorem consecutive_integers_sum_product (x : ℤ) : 
  (x + (x + 1) + (x + 2) = 27) → (x * (x + 1) * (x + 2) = 720) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_product_l1851_185164


namespace NUMINAMATH_CALUDE_books_from_first_shop_is_32_l1851_185100

/-- Represents the number of books bought from the first shop -/
def books_from_first_shop : ℕ := sorry

/-- The total amount spent on books from the first shop in Rs -/
def amount_first_shop : ℕ := 1500

/-- The number of books bought from the second shop -/
def books_from_second_shop : ℕ := 60

/-- The total amount spent on books from the second shop in Rs -/
def amount_second_shop : ℕ := 340

/-- The average price per book for all books in Rs -/
def average_price : ℕ := 20

/-- Theorem stating that the number of books bought from the first shop is 32 -/
theorem books_from_first_shop_is_32 : books_from_first_shop = 32 := by
  sorry

end NUMINAMATH_CALUDE_books_from_first_shop_is_32_l1851_185100


namespace NUMINAMATH_CALUDE_product_of_tripled_numbers_with_reciprocals_l1851_185165

theorem product_of_tripled_numbers_with_reciprocals (x : ℝ) : 
  (x + 1/x = 3*x) → (∃ y : ℝ, (y + 1/y = 3*y) ∧ (x * y = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_tripled_numbers_with_reciprocals_l1851_185165


namespace NUMINAMATH_CALUDE_club_member_selection_l1851_185133

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of members in the club -/
def totalMembers : ℕ := 15

/-- The number of members to be chosen -/
def chosenMembers : ℕ := 4

/-- The number of remaining members after excluding the two specific members -/
def remainingMembers : ℕ := totalMembers - 2

theorem club_member_selection :
  choose totalMembers chosenMembers - choose remainingMembers (chosenMembers - 2) = 1287 := by
  sorry

end NUMINAMATH_CALUDE_club_member_selection_l1851_185133


namespace NUMINAMATH_CALUDE_green_apples_count_l1851_185162

theorem green_apples_count (total : ℕ) (red : ℕ) (yellow : ℕ) (h1 : total = 19) (h2 : red = 3) (h3 : yellow = 14) :
  total - (red + yellow) = 2 := by
  sorry

end NUMINAMATH_CALUDE_green_apples_count_l1851_185162


namespace NUMINAMATH_CALUDE_complement_of_A_in_I_l1851_185191

def I : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {2,4,6,7}

theorem complement_of_A_in_I :
  I \ A = {1,3,5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_I_l1851_185191


namespace NUMINAMATH_CALUDE_sum_difference_is_270_l1851_185173

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  (x + 2) / 5 * 5

def sarah_sum (n : ℕ) : ℕ :=
  List.range n |> List.map round_to_nearest_five |> List.sum

theorem sum_difference_is_270 :
  sum_to_n 60 - sarah_sum 60 = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_270_l1851_185173


namespace NUMINAMATH_CALUDE_mixing_hcl_solutions_l1851_185188

/-- Represents a hydrochloric acid solution --/
structure HClSolution where
  mass : ℝ
  concentration : ℝ

/-- Calculates the mass of pure HCl in a solution --/
def HClMass (solution : HClSolution) : ℝ :=
  solution.mass * solution.concentration

theorem mixing_hcl_solutions
  (solution1 : HClSolution)
  (solution2 : HClSolution)
  (mixed : HClSolution)
  (h1 : solution1.concentration = 0.3)
  (h2 : solution2.concentration = 0.1)
  (h3 : mixed.concentration = 0.15)
  (h4 : mixed.mass = 600)
  (h5 : solution1.mass + solution2.mass = mixed.mass)
  (h6 : HClMass solution1 + HClMass solution2 = HClMass mixed) :
  solution1.mass = 150 ∧ solution2.mass = 450 := by
  sorry

end NUMINAMATH_CALUDE_mixing_hcl_solutions_l1851_185188


namespace NUMINAMATH_CALUDE_painter_problem_solution_l1851_185166

/-- Given a painting job with a total number of rooms, time per room, and some rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem stating that for the specific problem, the time to paint the remaining rooms is 63 hours. -/
theorem painter_problem_solution :
  time_to_paint_remaining 11 7 2 = 63 := by
  sorry

#eval time_to_paint_remaining 11 7 2

end NUMINAMATH_CALUDE_painter_problem_solution_l1851_185166


namespace NUMINAMATH_CALUDE_rajas_income_l1851_185114

theorem rajas_income (household_percent : ℝ) (clothes_percent : ℝ) (medicines_percent : ℝ)
  (transportation_percent : ℝ) (entertainment_percent : ℝ) (savings : ℝ) (income : ℝ) :
  household_percent = 0.45 →
  clothes_percent = 0.12 →
  medicines_percent = 0.08 →
  transportation_percent = 0.15 →
  entertainment_percent = 0.10 →
  savings = 5000 →
  household_percent * income + clothes_percent * income + medicines_percent * income +
    transportation_percent * income + entertainment_percent * income + savings = income →
  income = 50000 := by
  sorry

end NUMINAMATH_CALUDE_rajas_income_l1851_185114


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1851_185102

-- Define the operation ⋄
noncomputable def diamond (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem diamond_equation_solution :
  ∃ y : ℝ, diamond 5 y = 15 ∧ y = 90 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1851_185102


namespace NUMINAMATH_CALUDE_equation_solutions_l1851_185110

theorem equation_solutions :
  (∃! x : ℝ, (2 / x = 3 / (x + 2)) ∧ x = 4) ∧
  (¬ ∃ x : ℝ, 1 / (x - 2) = (1 - x) / (2 - x) - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1851_185110


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1851_185174

/-- Given a point P in the second quadrant with absolute x-coordinate 5 and absolute y-coordinate 7,
    the point symmetric to P with respect to the origin has coordinates (5, -7). -/
theorem symmetric_point_coordinates :
  ∀ (x y : ℝ),
    x < 0 →  -- Point is in the second quadrant (x is negative)
    y > 0 →  -- Point is in the second quadrant (y is positive)
    |x| = 5 →
    |y| = 7 →
    (- x, - y) = (5, -7) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1851_185174


namespace NUMINAMATH_CALUDE_complement_of_A_l1851_185169

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x + 2) / x < 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≥ 0 ∨ x ≤ -2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1851_185169


namespace NUMINAMATH_CALUDE_rectangle_area_l1851_185135

/-- The area of a rectangle with perimeter 200 cm, which can be divided into five identical squares -/
theorem rectangle_area (side : ℝ) (h1 : side > 0) (h2 : 12 * side = 200) : 
  5 * side^2 = 12500 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1851_185135


namespace NUMINAMATH_CALUDE_snickers_count_l1851_185160

theorem snickers_count (total : ℕ) (mars : ℕ) (butterfingers : ℕ) (snickers : ℕ) : 
  total = 12 → mars = 2 → butterfingers = 7 → total = mars + butterfingers + snickers → snickers = 3 := by
  sorry

end NUMINAMATH_CALUDE_snickers_count_l1851_185160


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1851_185138

theorem arithmetic_mean_of_fractions : 
  let a : ℚ := 3/4
  let b : ℚ := 5/8
  (a + b) / 2 = 11/16 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1851_185138


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1851_185157

-- Define the slopes of two lines
def slope1 (a : ℝ) := -a
def slope2 : ℝ := 3

-- Define the perpendicular condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, perpendicular (slope1 a) slope2 → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1851_185157


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1851_185143

/-- The sum of the coordinates of the midpoint of a segment with endpoints (6, 12) and (0, -6) is 6. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (6, 12)
  let p2 : ℝ × ℝ := (0, -6)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1851_185143


namespace NUMINAMATH_CALUDE_cuckoo_clock_strikes_l1851_185159

def clock_strikes (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

def total_strikes (start_hour end_hour : Nat) : Nat :=
  (List.range (end_hour - start_hour + 1)).map (fun i => clock_strikes (start_hour + i))
    |>.sum

theorem cuckoo_clock_strikes :
  total_strikes 10 16 = 43 := by
  sorry

end NUMINAMATH_CALUDE_cuckoo_clock_strikes_l1851_185159


namespace NUMINAMATH_CALUDE_smallest_perimeter_triangle_l1851_185179

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle formed by two rays from a vertex -/
structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line passing through a point -/
structure Line where
  point : Point
  direction : ℝ

/-- Checks if a point is inside an angle -/
def isPointInsideAngle (p : Point) (a : Angle) : Prop := sorry

/-- Finds the larger inscribed circle passing through a point in an angle -/
def largerInscribedCircle (p : Point) (a : Angle) : Circle := sorry

/-- Checks if a line is tangent to a circle at a point -/
def isTangentLine (l : Line) (c : Circle) (p : Point) : Prop := sorry

/-- Calculates the perimeter of a triangle formed by a line intersecting an angle -/
def trianglePerimeter (l : Line) (a : Angle) : ℝ := sorry

/-- The main theorem -/
theorem smallest_perimeter_triangle 
  (M : Point) (KAL : Angle) 
  (h_inside : isPointInsideAngle M KAL) :
  let S := largerInscribedCircle M KAL
  let tangent_line := Line.mk M (sorry : ℝ)  -- Direction that makes it tangent
  ∀ (l : Line), 
    l.point = M → 
    isTangentLine tangent_line S M → 
    trianglePerimeter l KAL ≥ trianglePerimeter tangent_line KAL :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_triangle_l1851_185179


namespace NUMINAMATH_CALUDE_triangle_formation_l1851_185178

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 3 6 ∧
  ¬can_form_triangle 1 2 3 ∧
  ¬can_form_triangle 7 8 16 ∧
  ¬can_form_triangle 9 10 20 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1851_185178


namespace NUMINAMATH_CALUDE_jeremy_payment_l1851_185153

theorem jeremy_payment (rate : ℚ) (rooms : ℚ) (h1 : rate = 13 / 3) (h2 : rooms = 5 / 2) :
  rate * rooms = 65 / 6 := by sorry

end NUMINAMATH_CALUDE_jeremy_payment_l1851_185153


namespace NUMINAMATH_CALUDE_triangle_theorem_cosine_rule_sine_rule_l1851_185128

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the main theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 3 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C) :
  Real.cos t.A = 1/3 ∧ 
  (t.a = 1 ∧ Real.cos t.B + Real.cos t.C = 2 * Real.sqrt 3 / 3 → t.c = Real.sqrt 3 / 2) := by
  sorry

-- Define helper theorems for cosine and sine rules
theorem cosine_rule (t : Triangle) :
  2 * t.a * t.c * Real.cos t.B = t.a^2 + t.c^2 - t.b^2 := by
  sorry

theorem sine_rule (t : Triangle) :
  t.a / Real.sin t.A = t.b / Real.sin t.B := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_cosine_rule_sine_rule_l1851_185128


namespace NUMINAMATH_CALUDE_bilingual_point_part1_bilingual_points_part2_bilingual_point_part3_l1851_185171

/-- Definition of a bilingual point -/
def is_bilingual_point (x y : ℝ) : Prop := y = 2 * x

/-- Part 1: Bilingual point of y = 3x + 1 -/
theorem bilingual_point_part1 : 
  ∃ x y : ℝ, is_bilingual_point x y ∧ y = 3 * x + 1 ∧ x = -1 ∧ y = -2 := by sorry

/-- Part 2: Bilingual points of y = k/x -/
theorem bilingual_points_part2 (k : ℝ) (h : k ≠ 0) :
  (∃ x y : ℝ, is_bilingual_point x y ∧ y = k / x) ↔ k > 0 := by sorry

/-- Part 3: Conditions for the function y = 1/4 * x^2 + (n-k-1)x + m+k+2 -/
theorem bilingual_point_part3 (n m k : ℝ) :
  (∃! x y : ℝ, is_bilingual_point x y ∧ 
    y = 1/4 * x^2 + (n - k - 1) * x + m + k + 2) ∧
  1 ≤ n ∧ n ≤ 3 ∧
  (∀ m' : ℝ, m' ≥ m → 
    ∃! x y : ℝ, is_bilingual_point x y ∧ 
      y = 1/4 * x^2 + (n - k - 1) * x + m' + k + 2) →
  k = 1 + Real.sqrt 3 ∨ k = -1 := by sorry

end NUMINAMATH_CALUDE_bilingual_point_part1_bilingual_points_part2_bilingual_point_part3_l1851_185171


namespace NUMINAMATH_CALUDE_car_value_after_depreciation_l1851_185119

/-- Calculates the current value of a car given its initial price and depreciation rate. -/
def currentCarValue (initialPrice : ℝ) (depreciationRate : ℝ) : ℝ :=
  initialPrice * (1 - depreciationRate)

/-- Theorem stating that a car initially priced at $4000 with 30% depreciation is now worth $2800. -/
theorem car_value_after_depreciation :
  currentCarValue 4000 0.3 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_car_value_after_depreciation_l1851_185119


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1851_185158

theorem floor_ceil_sum : ⌊(-0.237 : ℝ)⌋ + ⌈(4.987 : ℝ)⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1851_185158


namespace NUMINAMATH_CALUDE_equation_condition_l1851_185183

theorem equation_condition (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  (20 * a + b) * (20 * a + c) = 400 * a * (a + 1) + 10 * b * c →
  b + c = 20 :=
sorry

end NUMINAMATH_CALUDE_equation_condition_l1851_185183


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l1851_185109

theorem cosine_sine_inequality (a b : ℝ) 
  (h : ∀ x : ℝ, Real.cos (a * Real.sin x) > Real.sin (b * Real.cos x)) : 
  a^2 + b^2 < (Real.pi^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l1851_185109


namespace NUMINAMATH_CALUDE_remainder_problem_l1851_185146

theorem remainder_problem (D : ℕ) (R : ℕ) (h1 : D > 0) 
  (h2 : 242 % D = 4) 
  (h3 : 698 % D = R) 
  (h4 : 940 % D = 7) : R = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1851_185146


namespace NUMINAMATH_CALUDE_coefficient_of_b_fourth_l1851_185103

theorem coefficient_of_b_fourth (b : ℝ) : 
  (∃ b : ℝ, b^4 - 41*b^2 + 100 = 0) ∧ 
  (∃ b₁ b₂ : ℝ, b₁ ≥ b₂ ∧ b₂ ≥ 0 ∧ b₁ + b₂ = 4.5 ∧ 
    b₁^4 - 41*b₁^2 + 100 = 0 ∧ b₂^4 - 41*b₂^2 + 100 = 0) →
  (∃ a : ℝ, ∀ b : ℝ, a*b^4 - 41*b^2 + 100 = 0 → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_b_fourth_l1851_185103


namespace NUMINAMATH_CALUDE_negation_of_existence_real_roots_l1851_185149

theorem negation_of_existence_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_real_roots_l1851_185149


namespace NUMINAMATH_CALUDE_correct_calculation_l1851_185118

theorem correct_calculation (square : ℕ) (h : (325 - square) * 5 = 1500) : 
  325 - square * 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1851_185118


namespace NUMINAMATH_CALUDE_no_intersection_and_in_circle_l1851_185123

theorem no_intersection_and_in_circle : ¬∃ (a b : ℝ),
  (∃ (n m : ℤ), n = m ∧ n * a + b = 3 * m^2 + 15) ∧
  (a^2 + b^2 ≤ 144) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_and_in_circle_l1851_185123


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l1851_185189

/-- Proves that a price reduction resulting in an 80% increase in sales and a 26% increase in total revenue implies a 30% price reduction -/
theorem price_reduction_theorem (P S : ℝ) (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 100) 
  (h3 : P > 0) 
  (h4 : S > 0) 
  (h5 : P * (1 - x / 100) * (S * 1.8) = P * S * 1.26) : 
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l1851_185189


namespace NUMINAMATH_CALUDE_investment_return_percentage_l1851_185156

/-- Proves that the yearly return percentage of a $500 investment is 7% given specific conditions --/
theorem investment_return_percentage : 
  ∀ (total_investment small_investment large_investment : ℝ)
    (combined_return_rate small_return_rate large_return_rate : ℝ),
  total_investment = 2000 →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.10 →
  large_return_rate = 0.11 →
  combined_return_rate * total_investment = 
    small_return_rate * small_investment + large_return_rate * large_investment →
  small_return_rate = 0.07 := by
sorry


end NUMINAMATH_CALUDE_investment_return_percentage_l1851_185156


namespace NUMINAMATH_CALUDE_time_to_hear_second_blast_l1851_185152

/-- The time taken for a man to hear a second blast, given specific conditions -/
theorem time_to_hear_second_blast 
  (speed_of_sound : ℝ) 
  (time_between_blasts : ℝ) 
  (distance_at_second_blast : ℝ) 
  (h1 : speed_of_sound = 330)
  (h2 : time_between_blasts = 30 * 60)
  (h3 : distance_at_second_blast = 4950) :
  speed_of_sound * (time_between_blasts + distance_at_second_blast / speed_of_sound) = 1815 * speed_of_sound :=
by sorry

end NUMINAMATH_CALUDE_time_to_hear_second_blast_l1851_185152


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l1851_185198

/-- Represents a 2x4 grid of points -/
def Grid := Fin 2 × Fin 4

/-- Represents a triangle formed by three points on the grid -/
def Triangle := Fin 3 → Grid

/-- Checks if three points are collinear -/
def collinear (p q r : Grid) : Prop := sorry

/-- Counts the number of distinct triangles in a 2x4 grid -/
def count_distinct_triangles : ℕ := sorry

/-- Theorem stating that the number of distinct triangles in a 2x4 grid is 44 -/
theorem distinct_triangles_count : count_distinct_triangles = 44 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_count_l1851_185198


namespace NUMINAMATH_CALUDE_canoe_rental_cost_l1851_185180

/-- The cost of renting a canoe per day -/
def canoe_cost : ℚ := 9

/-- The cost of renting a kayak per day -/
def kayak_cost : ℚ := 12

/-- The ratio of canoes to kayaks rented -/
def canoe_kayak_ratio : ℚ := 4/3

/-- The number of additional canoes compared to kayaks -/
def additional_canoes : ℕ := 6

/-- The total revenue for the day -/
def total_revenue : ℚ := 432

theorem canoe_rental_cost :
  let kayaks : ℕ := 18
  let canoes : ℕ := kayaks + additional_canoes
  canoe_cost * canoes + kayak_cost * kayaks = total_revenue ∧
  (canoes : ℚ) / kayaks = canoe_kayak_ratio :=
by sorry

end NUMINAMATH_CALUDE_canoe_rental_cost_l1851_185180


namespace NUMINAMATH_CALUDE_irrational_pi_among_options_l1851_185199

theorem irrational_pi_among_options : 
  (∃ (a b : ℤ), (3.142 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), (Real.sqrt 4 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), (22 / 7 : ℝ) = a / b) ∧ 
  (¬ ∃ (a b : ℤ), (Real.pi : ℝ) = a / b) :=
by sorry

end NUMINAMATH_CALUDE_irrational_pi_among_options_l1851_185199


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1851_185101

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 36 * r = b ∧ b * r = 2 / 9) → b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1851_185101


namespace NUMINAMATH_CALUDE_half_percent_as_repeating_decimal_l1851_185155

theorem half_percent_as_repeating_decimal : 
  (1 / 2 : ℚ) / 100 = 0.00500 := by sorry

end NUMINAMATH_CALUDE_half_percent_as_repeating_decimal_l1851_185155


namespace NUMINAMATH_CALUDE_candles_per_box_l1851_185116

/-- Given Kerry's birthday celebration scenario, prove the number of candles in a box. -/
theorem candles_per_box (kerry_age : ℕ) (num_cakes : ℕ) (total_cost : ℚ) (box_cost : ℚ) 
  (h1 : kerry_age = 8)
  (h2 : num_cakes = 3)
  (h3 : total_cost = 5)
  (h4 : box_cost = 5/2) :
  (kerry_age * num_cakes) / (total_cost / box_cost) = 12 := by
  sorry

end NUMINAMATH_CALUDE_candles_per_box_l1851_185116


namespace NUMINAMATH_CALUDE_simon_change_calculation_l1851_185167

def pansy_price : ℝ := 2.50
def pansy_quantity : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_quantity : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_quantity : ℕ := 5
def discount_rate : ℝ := 0.10
def paid_amount : ℝ := 50.00

theorem simon_change_calculation :
  let total_before_discount := pansy_price * pansy_quantity + hydrangea_price * hydrangea_quantity + petunia_price * petunia_quantity
  let discount := total_before_discount * discount_rate
  let total_after_discount := total_before_discount - discount
  let change := paid_amount - total_after_discount
  change = 23.00 := by sorry

end NUMINAMATH_CALUDE_simon_change_calculation_l1851_185167


namespace NUMINAMATH_CALUDE_matt_keychains_purchase_l1851_185181

/-- The number of key chains Matt buys -/
def num_keychains : ℕ := 10

/-- The price of a pack of 10 key chains -/
def price_pack_10 : ℚ := 20

/-- The price of a pack of 4 key chains -/
def price_pack_4 : ℚ := 12

/-- The amount Matt saves by choosing the cheaper option -/
def savings : ℚ := 20

theorem matt_keychains_purchase :
  num_keychains = 10 ∧
  (num_keychains : ℚ) * (price_pack_10 / 10) = 
    (num_keychains : ℚ) * (price_pack_4 / 4) - savings :=
by sorry

end NUMINAMATH_CALUDE_matt_keychains_purchase_l1851_185181


namespace NUMINAMATH_CALUDE_cookies_eaten_l1851_185150

theorem cookies_eaten (initial_cookies remaining_cookies : ℕ) 
  (h1 : initial_cookies = 28)
  (h2 : remaining_cookies = 7) :
  initial_cookies - remaining_cookies = 21 := by
sorry

end NUMINAMATH_CALUDE_cookies_eaten_l1851_185150


namespace NUMINAMATH_CALUDE_greatest_divisible_power_of_three_l1851_185168

theorem greatest_divisible_power_of_three (m : ℕ+) : 
  (∃ (k : ℕ), k = 2 ∧ (3^k : ℕ) ∣ (2^(3^m.val) + 1)) ∧
  (∀ (k : ℕ), k > 2 → ¬((3^k : ℕ) ∣ (2^(3^m.val) + 1))) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisible_power_of_three_l1851_185168


namespace NUMINAMATH_CALUDE_product_equality_implies_composite_sums_l1851_185196

theorem product_equality_implies_composite_sums (a b c d : ℕ) (h : a * b = c * d) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + b + c + d = x * y) ∧
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a^2 + b^2 + c^2 + d^2 = x * y) :=
by sorry

end NUMINAMATH_CALUDE_product_equality_implies_composite_sums_l1851_185196


namespace NUMINAMATH_CALUDE_count_congruent_integers_l1851_185182

theorem count_congruent_integers (n : ℕ) (m : ℕ) (a : ℕ) (b : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < n ∧ x % m = a) (Finset.range n)).card = b + 1 :=
by
  sorry

#check count_congruent_integers 1500 13 7 114

end NUMINAMATH_CALUDE_count_congruent_integers_l1851_185182


namespace NUMINAMATH_CALUDE_max_dot_product_in_trapezoid_l1851_185136

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is inside or on the boundary of a trapezoid -/
def isInTrapezoid (t : Trapezoid) (p : Point) : Prop := sorry

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Main theorem -/
theorem max_dot_product_in_trapezoid (t : Trapezoid) :
  t.A = Point.mk 0 0 →
  t.B = Point.mk 3 0 →
  t.C = Point.mk 2 2 →
  t.D = Point.mk 0 2 →
  let N := Point.mk 1 2
  ∀ M : Point, isInTrapezoid t M →
  dotProduct (Point.mk (M.x - t.A.x) (M.y - t.A.y)) (Point.mk (N.x - t.A.x) (N.y - t.A.y)) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_dot_product_in_trapezoid_l1851_185136


namespace NUMINAMATH_CALUDE_base_ten_to_base_seven_l1851_185115

theorem base_ten_to_base_seven : 
  ∃ (a b c d : ℕ), 
    947 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 5 ∧ c = 2 ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_to_base_seven_l1851_185115


namespace NUMINAMATH_CALUDE_product_of_smaller_numbers_l1851_185170

theorem product_of_smaller_numbers (A B C : ℝ) : 
  B = 10 → 
  C - B = B - A → 
  B * C = 115 → 
  A * B = 85 := by
sorry

end NUMINAMATH_CALUDE_product_of_smaller_numbers_l1851_185170


namespace NUMINAMATH_CALUDE_multi_digit_perfect_square_distinct_digits_l1851_185145

theorem multi_digit_perfect_square_distinct_digits :
  ∀ n : ℕ, n > 9 → (∃ m : ℕ, n = m^2) →
    ∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧
    ∃ k : ℕ, n = d₁ + 10 * k ∧ ∃ l : ℕ, k = d₂ + 10 * l :=
by sorry

end NUMINAMATH_CALUDE_multi_digit_perfect_square_distinct_digits_l1851_185145


namespace NUMINAMATH_CALUDE_dice_probability_l1851_185140

def probability_less_than_6 : ℚ := 1 / 2

def number_of_dice : ℕ := 6

def target_count : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dice_probability : 
  (choose number_of_dice target_count : ℚ) * probability_less_than_6^number_of_dice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1851_185140


namespace NUMINAMATH_CALUDE_right_triangle_area_l1851_185184

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1851_185184


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l1851_185105

theorem number_of_divisors_of_60 : ∃ (s : Finset Nat), ∀ d : Nat, d ∈ s ↔ d ∣ 60 ∧ d > 0 ∧ Finset.card s = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l1851_185105


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1851_185113

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

/-- Theorem: Given $35 principal and 4% simple annual interest, 
    the total amount owed after one year is $36.40 -/
theorem simple_interest_calculation :
  total_amount_owed 35 0.04 1 = 36.40 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1851_185113


namespace NUMINAMATH_CALUDE_macaroon_packing_l1851_185129

/-- The number of brown bags used to pack macaroons -/
def number_of_bags : ℕ := 4

/-- The total number of macaroons -/
def total_macaroons : ℕ := 12

/-- The weight of each macaroon in ounces -/
def macaroon_weight : ℕ := 5

/-- The remaining weight of macaroons after one bag is eaten, in ounces -/
def remaining_weight : ℕ := 45

theorem macaroon_packing :
  (total_macaroons % number_of_bags = 0) ∧
  (total_macaroons / number_of_bags * macaroon_weight = 
   total_macaroons * macaroon_weight - remaining_weight) →
  number_of_bags = 4 := by
sorry

end NUMINAMATH_CALUDE_macaroon_packing_l1851_185129


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1851_185161

theorem election_votes_theorem (emily_votes : ℕ) (emily_fraction : ℚ) (dexter_fraction : ℚ) :
  emily_votes = 48 →
  emily_fraction = 4 / 15 →
  dexter_fraction = 1 / 3 →
  ∃ (total_votes : ℕ),
    (emily_votes : ℚ) / total_votes = emily_fraction ∧
    total_votes = 180 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l1851_185161


namespace NUMINAMATH_CALUDE_prime_pair_from_quadratic_roots_l1851_185137

theorem prime_pair_from_quadratic_roots (p q : ℕ) (hp : p.Prime) (hq : q.Prime) 
  (x₁ x₂ : ℤ) (h_sum : x₁ + x₂ = -p) (h_prod : x₁ * x₂ = q) : p = 3 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_from_quadratic_roots_l1851_185137


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l1851_185192

/-- Triathlete's average speed for a multi-segment trip -/
theorem triathlete_average_speed (total_distance : ℝ) 
  (run_flat_speed run_uphill_speed run_downhill_speed swim_speed bike_speed : ℝ)
  (run_flat_distance run_uphill_distance run_downhill_distance swim_distance bike_distance : ℝ)
  (h1 : total_distance = run_flat_distance + run_uphill_distance + run_downhill_distance + swim_distance + bike_distance)
  (h2 : run_flat_speed > 0 ∧ run_uphill_speed > 0 ∧ run_downhill_speed > 0 ∧ swim_speed > 0 ∧ bike_speed > 0)
  (h3 : run_flat_distance > 0 ∧ run_uphill_distance > 0 ∧ run_downhill_distance > 0 ∧ swim_distance > 0 ∧ bike_distance > 0)
  (h4 : total_distance = 9)
  (h5 : run_flat_speed = 10)
  (h6 : run_uphill_speed = 6)
  (h7 : run_downhill_speed = 14)
  (h8 : swim_speed = 4)
  (h9 : bike_speed = 12)
  (h10 : run_flat_distance = 1)
  (h11 : run_uphill_distance = 1)
  (h12 : run_downhill_distance = 1)
  (h13 : swim_distance = 3)
  (h14 : bike_distance = 3) :
  ∃ (average_speed : ℝ), abs (average_speed - 0.1121) < 0.0001 ∧
    average_speed = total_distance / (run_flat_distance / run_flat_speed + 
                                      run_uphill_distance / run_uphill_speed + 
                                      run_downhill_distance / run_downhill_speed + 
                                      swim_distance / swim_speed + 
                                      bike_distance / bike_speed) / 60 :=
by sorry

end NUMINAMATH_CALUDE_triathlete_average_speed_l1851_185192


namespace NUMINAMATH_CALUDE_not_square_and_floor_sqrt_cube_divides_square_l1851_185107

theorem not_square_and_floor_sqrt_cube_divides_square (n : ℕ) :
  (∀ k : ℕ, n ≠ k^2) →
  (Nat.floor (Real.sqrt n))^3 ∣ n^2 →
  n = 2 ∨ n = 3 ∨ n = 8 ∨ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_not_square_and_floor_sqrt_cube_divides_square_l1851_185107


namespace NUMINAMATH_CALUDE_f_is_decreasing_l1851_185163

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom additivity : ∀ x y, f (x + y) = f x + f y
axiom negative_for_positive : ∀ x, x > 0 → f x < 0

-- State the theorem
theorem f_is_decreasing : 
  (∀ x y, x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_decreasing_l1851_185163


namespace NUMINAMATH_CALUDE_min_value_reciprocal_plus_x_l1851_185187

theorem min_value_reciprocal_plus_x (x : ℝ) (h : x > 0) : 
  4 / x + x ≥ 4 ∧ (4 / x + x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_plus_x_l1851_185187


namespace NUMINAMATH_CALUDE_antonov_candy_packs_l1851_185125

/-- Given a total number of candies and packs, calculate the number of candies per pack -/
def candies_per_pack (total_candies : ℕ) (total_packs : ℕ) : ℕ :=
  total_candies / total_packs

/-- Theorem: The number of candies per pack is 20 -/
theorem antonov_candy_packs : candies_per_pack 60 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_antonov_candy_packs_l1851_185125


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l1851_185120

theorem sum_a_b_equals_one (a b : ℝ) (h : Real.sqrt (a - b - 3) + |2 * a - 4| = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l1851_185120


namespace NUMINAMATH_CALUDE_winner_takes_eight_l1851_185185

/-- Represents the game state and rules --/
structure GameState where
  total_candies : ℕ
  winner_candies : ℕ
  loser_candies : ℕ
  winner_rounds : ℕ
  winner_takes : ℕ
  loser_takes : ℕ

/-- The theorem statement --/
theorem winner_takes_eight (game : GameState) : 
  game.total_candies = 55 ∧ 
  game.winner_candies = 25 ∧ 
  game.loser_candies = 30 ∧
  game.winner_rounds = 2 ∧
  game.winner_takes > game.loser_takes ∧
  game.winner_takes > 0 ∧
  game.loser_takes > 0 →
  game.winner_takes = 8 := by
  sorry

end NUMINAMATH_CALUDE_winner_takes_eight_l1851_185185


namespace NUMINAMATH_CALUDE_circle_transformation_l1851_185131

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_transformation (C : ℝ × ℝ) (h : C = (3, -4)) :
  (translate_right (reflect_x C) 5) = (8, 4) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l1851_185131


namespace NUMINAMATH_CALUDE_sum_59_28_rounded_equals_90_l1851_185112

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem sum_59_28_rounded_equals_90 : 
  round_to_nearest_ten (59 + 28) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_59_28_rounded_equals_90_l1851_185112


namespace NUMINAMATH_CALUDE_debate_team_group_size_l1851_185122

theorem debate_team_group_size :
  ∀ (boys girls groups : ℕ),
    boys = 26 →
    girls = 46 →
    groups = 8 →
    (boys + girls) / groups = 9 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l1851_185122


namespace NUMINAMATH_CALUDE_absolute_value_condition_l1851_185194

theorem absolute_value_condition (a : ℝ) : 
  (a ≤ 0 → |a - 2| ≥ 1) ∧ 
  ¬(|a - 2| ≥ 1 → a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_condition_l1851_185194


namespace NUMINAMATH_CALUDE_remainder_problem_l1851_185195

theorem remainder_problem (x : ℤ) : 
  (4 * x) % 7 = 6 → x % 7 = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1851_185195
