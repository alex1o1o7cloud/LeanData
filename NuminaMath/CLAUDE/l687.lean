import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_property_l687_68750

theorem perfect_square_property (n : ℕ) (hn : n ≥ 3) 
  (hx : ∃ x : ℕ, 1 + 3 * n = x ^ 2) : 
  ∃ a b c : ℕ, 1 + (3 * n + 3) / (a ^ 2 + b ^ 2 + c ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l687_68750


namespace NUMINAMATH_CALUDE_base_nine_subtraction_l687_68778

/-- Represents a number in base 9 --/
def BaseNine : Type := ℕ

/-- Converts a base 9 number to its decimal (base 10) representation --/
def to_decimal (n : BaseNine) : ℕ := sorry

/-- Converts a decimal (base 10) number to its base 9 representation --/
def from_decimal (n : ℕ) : BaseNine := sorry

/-- Subtracts two base 9 numbers --/
def base_nine_sub (a b : BaseNine) : BaseNine := sorry

/-- The main theorem to prove --/
theorem base_nine_subtraction :
  base_nine_sub (from_decimal 256) (from_decimal 143) = from_decimal 113 := by sorry

end NUMINAMATH_CALUDE_base_nine_subtraction_l687_68778


namespace NUMINAMATH_CALUDE_work_for_series_springs_l687_68798

/-- Work required to stretch a system of two springs in series -/
theorem work_for_series_springs (k₁ k₂ : ℝ) (x : ℝ) (h₁ : k₁ = 6000) (h₂ : k₂ = 12000) (h₃ : x = 0.1) :
  (1 / 2) * (1 / (1 / k₁ + 1 / k₂)) * x^2 = 20 := by
  sorry

#check work_for_series_springs

end NUMINAMATH_CALUDE_work_for_series_springs_l687_68798


namespace NUMINAMATH_CALUDE_third_consecutive_odd_integer_l687_68775

theorem third_consecutive_odd_integer (x : ℤ) : 
  (∀ n : ℤ, (x + 2*n) % 2 ≠ 0) →  -- x is odd
  3*x = 2*(x + 4) + 3 →          -- condition from the problem
  x + 4 = 15 :=                  -- third integer is 15
by sorry

end NUMINAMATH_CALUDE_third_consecutive_odd_integer_l687_68775


namespace NUMINAMATH_CALUDE_range_of_x_l687_68754

theorem range_of_x (x : ℝ) : (16 - x^2 ≥ 0) ↔ (-4 ≤ x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l687_68754


namespace NUMINAMATH_CALUDE_fraction_inequality_function_minimum_l687_68789

-- Problem 1
theorem fraction_inequality (c a b : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  a / (c - a) > b / (c - b) := by sorry

-- Problem 2
theorem function_minimum (x : ℝ) (h : x > 2) :
  x + 16 / (x - 2) ≥ 10 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_function_minimum_l687_68789


namespace NUMINAMATH_CALUDE_binomial_and_power_evaluation_l687_68741

theorem binomial_and_power_evaluation : 
  (Nat.choose 12 6 = 924) ∧ ((1 + 1 : ℕ)^12 = 4096) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_power_evaluation_l687_68741


namespace NUMINAMATH_CALUDE_revenue_decrease_l687_68766

theorem revenue_decrease (last_year_revenue : ℝ) : 
  let projected_revenue := 1.25 * last_year_revenue
  let actual_revenue := 0.6 * projected_revenue
  let decrease := projected_revenue - actual_revenue
  let percentage_decrease := (decrease / projected_revenue) * 100
  percentage_decrease = 40 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l687_68766


namespace NUMINAMATH_CALUDE_square_fraction_count_l687_68787

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, ∃ k : ℤ, n / (25 - n) = k^2 ∧ 25 - n ≠ 0) ∧ 
    S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_square_fraction_count_l687_68787


namespace NUMINAMATH_CALUDE_forint_bill_solution_exists_l687_68700

def is_valid_solution (x y z : ℕ) : Prop :=
  10 * x + 5 * y + z = 682 ∧ x = y + z

def is_one_of_solutions (x y z : ℕ) : Prop :=
  (x = 58 ∧ y = 11 ∧ z = 47) ∨
  (x = 54 ∧ y = 22 ∧ z = 32) ∨
  (x = 50 ∧ y = 33 ∧ z = 17) ∨
  (x = 46 ∧ y = 44 ∧ z = 2)

theorem forint_bill_solution_exists :
  ∃ x y z : ℕ, is_valid_solution x y z ∧ is_one_of_solutions x y z := by
  sorry

end NUMINAMATH_CALUDE_forint_bill_solution_exists_l687_68700


namespace NUMINAMATH_CALUDE_dispatch_plans_eq_180_l687_68786

/-- Represents the number of male officials -/
def num_males : ℕ := 5

/-- Represents the number of female officials -/
def num_females : ℕ := 3

/-- Represents the total number of officials -/
def total_officials : ℕ := num_males + num_females

/-- Represents the minimum number of officials in each group -/
def min_group_size : ℕ := 3

/-- Calculates the number of ways to divide officials into two groups -/
def dispatch_plans : ℕ := sorry

/-- Theorem stating that the number of dispatch plans is 180 -/
theorem dispatch_plans_eq_180 : dispatch_plans = 180 := by sorry

end NUMINAMATH_CALUDE_dispatch_plans_eq_180_l687_68786


namespace NUMINAMATH_CALUDE_choose_starters_with_triplet_l687_68723

/-- The number of players in the soccer team -/
def total_players : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of ways to choose 7 starters from 16 players with at least one triplet -/
def ways_to_choose_starters : ℕ := 9721

/-- Theorem stating that the number of ways to choose 7 starters from 16 players,
    including a set of triplets, such that at least one of the triplets is in the
    starting lineup, is equal to 9721 -/
theorem choose_starters_with_triplet :
  (Nat.choose num_triplets 1 * Nat.choose (total_players - num_triplets) (num_starters - 1) +
   Nat.choose num_triplets 2 * Nat.choose (total_players - num_triplets) (num_starters - 2) +
   Nat.choose num_triplets 3 * Nat.choose (total_players - num_triplets) (num_starters - 3)) =
  ways_to_choose_starters :=
by sorry

end NUMINAMATH_CALUDE_choose_starters_with_triplet_l687_68723


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l687_68733

/-- The ratio of the area of a square inscribed in an ellipse to the area of a square inscribed in a circle -/
theorem inscribed_square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse_square_area := (4 * a^2 * b^2) / (a^2 + b^2)
  let circle_square_area := 2 * b^2
  ellipse_square_area / circle_square_area = 2 * a^2 / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l687_68733


namespace NUMINAMATH_CALUDE_equation_solution_l687_68730

theorem equation_solution : ∃ s : ℚ, 
  (s^2 - 6*s + 8) / (s^2 - 9*s + 14) = (s^2 - 3*s - 18) / (s^2 - 2*s - 24) ∧ 
  s = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l687_68730


namespace NUMINAMATH_CALUDE_race_result_l687_68751

/-- Represents a participant in the race -/
structure Participant where
  position : ℝ
  speed : ℝ

/-- The race setup -/
structure Race where
  distance : ℝ
  a : Participant
  b : Participant
  c : Participant

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.distance = 60 ∧
  r.a.position = r.distance ∧
  r.b.position = r.distance - 10 ∧
  r.c.position = r.distance - 20

/-- Theorem stating the result of the race -/
theorem race_result (r : Race) :
  race_conditions r →
  (r.distance / r.b.speed - r.distance / r.c.speed) * r.c.speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_race_result_l687_68751


namespace NUMINAMATH_CALUDE_like_terms_sum_zero_l687_68729

theorem like_terms_sum_zero (a b : ℝ) (m n : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a^(m+1) * b^3 + (n-1) * a^2 * b^3 = 0) → (m = 1 ∧ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_zero_l687_68729


namespace NUMINAMATH_CALUDE_quadratic_factorization_l687_68714

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x, x^2 - 16*x + 60 = (x - a)*(x - b)) : 
  3*b - a = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l687_68714


namespace NUMINAMATH_CALUDE_prob_one_tail_theorem_l687_68780

/-- The probability of getting exactly one tail in 5 flips of a biased coin -/
def prob_one_tail_in_five_flips (p : ℝ) : ℝ :=
  5 * p * (1 - p)^4

/-- Theorem: The probability of getting exactly one tail in 5 flips of a biased coin -/
theorem prob_one_tail_theorem (p q : ℝ) 
  (h_prob : 0 ≤ p ∧ p ≤ 1) 
  (h_sum : p + q = 1) :
  prob_one_tail_in_five_flips p = 5 * p * q^4 :=
sorry

end NUMINAMATH_CALUDE_prob_one_tail_theorem_l687_68780


namespace NUMINAMATH_CALUDE_club_officer_selection_ways_l687_68761

def club_size : ℕ := 30
def num_officers : ℕ := 4

def ways_without_alice_bob : ℕ := 28 * 27 * 26 * 25
def ways_with_alice_bob : ℕ := 4 * 3 * 28 * 27

theorem club_officer_selection_ways :
  (ways_without_alice_bob + ways_with_alice_bob) = 500472 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_ways_l687_68761


namespace NUMINAMATH_CALUDE_power_greater_than_square_l687_68743

theorem power_greater_than_square (n : ℕ) (h : n ≥ 8) : 2^(n-1) > (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_square_l687_68743


namespace NUMINAMATH_CALUDE_divisor_of_p_l687_68719

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 40)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 100 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 150) :
  7 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_p_l687_68719


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l687_68755

theorem quadratic_root_theorem (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (a+b+c)*x + (ab+bc+ca)
  (f 2 = 0) → (∃ x, f x = 0 ∧ x ≠ 2) → (∃ x, f x = 0 ∧ x = a+b+c-2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l687_68755


namespace NUMINAMATH_CALUDE_smaller_rectangle_perimeter_is_9_l687_68703

/-- Represents a rectangle with its dimensions and division properties. -/
structure DividedRectangle where
  perimeter : ℝ
  verticalCuts : ℕ
  horizontalCuts : ℕ
  smallRectangles : ℕ
  totalCutLength : ℝ

/-- Calculates the perimeter of a smaller rectangle given a DividedRectangle. -/
def smallRectanglePerimeter (r : DividedRectangle) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating the perimeter of each smaller rectangle is 9 cm under given conditions. -/
theorem smaller_rectangle_perimeter_is_9 (r : DividedRectangle) 
    (h1 : r.perimeter = 96)
    (h2 : r.verticalCuts = 8)
    (h3 : r.horizontalCuts = 11)
    (h4 : r.smallRectangles = 108)
    (h5 : r.totalCutLength = 438) :
    smallRectanglePerimeter r = 9 := by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_perimeter_is_9_l687_68703


namespace NUMINAMATH_CALUDE_pet_center_cats_l687_68762

theorem pet_center_cats (initial_dogs : ℕ) (adopted_dogs : ℕ) (new_cats : ℕ) (final_total : ℕ) :
  initial_dogs = 36 →
  adopted_dogs = 20 →
  new_cats = 12 →
  final_total = 57 →
  ∃ initial_cats : ℕ,
    initial_cats = 29 ∧
    final_total = (initial_dogs - adopted_dogs) + (initial_cats + new_cats) :=
by sorry

end NUMINAMATH_CALUDE_pet_center_cats_l687_68762


namespace NUMINAMATH_CALUDE_digit_equation_solution_l687_68745

theorem digit_equation_solution :
  ∀ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 →
    100 * A + 10 * B + C = 3 * (A + B + C) + 294 →
    (A + B + C) * (100 * A + 10 * B + C) = 2295 →
    A = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l687_68745


namespace NUMINAMATH_CALUDE_odd_product_plus_one_is_odd_l687_68763

theorem odd_product_plus_one_is_odd (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  Odd (4 * p * q + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_product_plus_one_is_odd_l687_68763


namespace NUMINAMATH_CALUDE_circumcircle_equation_l687_68744

/-- Given a triangle ABC with sides defined by the equations:
    BC: x cos θ₁ + y sin θ₁ - p₁ = 0
    CA: x cos θ₂ + y sin θ₂ - p₂ = 0
    AB: x cos θ₃ + y sin θ₃ - p₃ = 0
    This theorem states that any point P(x, y) on the circumcircle of ABC
    satisfies the given equation. -/
theorem circumcircle_equation (θ₁ θ₂ θ₃ p₁ p₂ p₃ x y : ℝ) :
  (x * Real.cos θ₂ + y * Real.sin θ₂ - p₂) * (x * Real.cos θ₃ + y * Real.sin θ₃ - p₃) * Real.sin (θ₂ - θ₃) +
  (x * Real.cos θ₃ + y * Real.sin θ₃ - p₃) * (x * Real.cos θ₁ + y * Real.sin θ₁ - p₁) * Real.sin (θ₃ - θ₁) +
  (x * Real.cos θ₁ + y * Real.sin θ₁ - p₁) * (x * Real.cos θ₂ + y * Real.sin θ₂ - p₂) * Real.sin (θ₁ - θ₂) = 0 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l687_68744


namespace NUMINAMATH_CALUDE_ice_cube_ratio_l687_68767

def ice_cubes_in_glass : ℕ := 8
def number_of_trays : ℕ := 2
def spaces_per_tray : ℕ := 12

def total_ice_cubes : ℕ := number_of_trays * spaces_per_tray
def ice_cubes_in_pitcher : ℕ := total_ice_cubes - ice_cubes_in_glass

theorem ice_cube_ratio :
  (ice_cubes_in_pitcher : ℚ) / ice_cubes_in_glass = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_ice_cube_ratio_l687_68767


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l687_68720

theorem product_of_three_numbers (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ab : a * b = 45 * Real.rpow 3 (1/3))
  (h_ac : a * c = 75 * Real.rpow 3 (1/3))
  (h_bc : b * c = 27 * Real.rpow 3 (1/3)) :
  a * b * c = 135 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l687_68720


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_equation_negation_l687_68747

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem quadratic_equation_negation : 
  (¬∃ x : ℝ, x^2 + 2*x + 3 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_equation_negation_l687_68747


namespace NUMINAMATH_CALUDE_square_fence_perimeter_16_posts_l687_68725

/-- Calculates the outer perimeter of a square fence given the number of posts,
    post width, and gap between posts. -/
def squareFencePerimeter (numPosts : ℕ) (postWidth : ℚ) (gapWidth : ℚ) : ℚ :=
  let postsPerSide : ℕ := numPosts / 4
  let gapsPerSide : ℕ := postsPerSide - 1
  let sideLength : ℚ := (gapsPerSide : ℚ) * gapWidth + (postsPerSide : ℚ) * postWidth
  4 * sideLength

/-- The outer perimeter of a square fence with 16 posts, each 6 inches wide,
    and 4 feet between posts, is 56 feet. -/
theorem square_fence_perimeter_16_posts :
  squareFencePerimeter 16 (1/2) 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_16_posts_l687_68725


namespace NUMINAMATH_CALUDE_shopping_theorem_l687_68721

def shopping_problem (shoe_price : ℝ) (shoe_discount : ℝ) (shirt_price : ℝ) (num_shirts : ℕ) (final_discount : ℝ) : Prop :=
  let discounted_shoe_price := shoe_price * (1 - shoe_discount)
  let total_shirt_price := shirt_price * num_shirts
  let subtotal := discounted_shoe_price + total_shirt_price
  let final_price := subtotal * (1 - final_discount)
  final_price = 285

theorem shopping_theorem :
  shopping_problem 200 0.30 80 2 0.05 := by
  sorry

end NUMINAMATH_CALUDE_shopping_theorem_l687_68721


namespace NUMINAMATH_CALUDE_ratio_pq_qr_l687_68731

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the radius of the circle
def radius : ℝ := 2

-- Define the points P, Q, and R on the circle
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distance between two points
def distance : Point → Point → ℝ := sorry

-- Define the length of an arc
def arcLength : Point → Point → ℝ := sorry

-- State the theorem
theorem ratio_pq_qr (h1 : distance P Q = distance P R)
                    (h2 : distance P Q > radius)
                    (h3 : arcLength Q R = 2 * Real.pi) :
  distance P Q / arcLength Q R = 2 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ratio_pq_qr_l687_68731


namespace NUMINAMATH_CALUDE_wire_ratio_square_octagon_l687_68709

/-- The ratio of wire lengths for equal-area square and octagon -/
theorem wire_ratio_square_octagon (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a / 4)^2 = (1 + Real.sqrt 2) * (b / 8)^2 → a / b = Real.sqrt (2 * (1 + Real.sqrt 2)) / 2 := by
  sorry

#check wire_ratio_square_octagon

end NUMINAMATH_CALUDE_wire_ratio_square_octagon_l687_68709


namespace NUMINAMATH_CALUDE_inequality_proof_l687_68772

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l687_68772


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_max_on_interval_l687_68701

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Theorem for monotonically decreasing intervals
theorem f_monotone_decreasing (a : ℝ) :
  ∀ x, (x < -1 ∨ x > 3) → (∀ y, y > x → f a y < f a x) :=
sorry

-- Theorem for minimum and maximum values when a = -2
theorem f_min_max_on_interval :
  let a := -2
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f a y ≤ f a x) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a x ≥ -7) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a x ≤ 20) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a x = -7) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a x = 20) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_max_on_interval_l687_68701


namespace NUMINAMATH_CALUDE_distribution_proportion_l687_68708

theorem distribution_proportion (total : ℚ) (p q r s : ℚ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  s - p = 250 →
  p + q + r + s = total →
  q / r = 1 := by
  sorry

end NUMINAMATH_CALUDE_distribution_proportion_l687_68708


namespace NUMINAMATH_CALUDE_exists_unique_marking_scheme_l687_68706

/-- Represents a cell in the grid -/
structure Cell :=
  (row : Nat)
  (col : Nat)

/-- Represents a marking scheme for the grid -/
def MarkingScheme := Set Cell

/-- Represents a 10x10 sub-square in the grid -/
structure SubSquare :=
  (topLeft : Cell)

/-- Counts the number of marked cells in a sub-square -/
def countMarkedCells (scheme : MarkingScheme) (square : SubSquare) : Nat :=
  sorry

/-- Checks if all sub-squares have unique counts -/
def allSubSquaresUnique (scheme : MarkingScheme) : Prop :=
  sorry

/-- Main theorem: There exists a marking scheme where all sub-squares have unique counts -/
theorem exists_unique_marking_scheme :
  ∃ (scheme : MarkingScheme),
    (∀ c : Cell, c.row < 19 ∧ c.col < 19) →
    (∀ s : SubSquare, s.topLeft.row ≤ 9 ∧ s.topLeft.col ≤ 9) →
    allSubSquaresUnique scheme :=
  sorry

end NUMINAMATH_CALUDE_exists_unique_marking_scheme_l687_68706


namespace NUMINAMATH_CALUDE_window_treatment_cost_l687_68742

def number_of_windows : ℕ := 3
def cost_of_sheers : ℚ := 40
def cost_of_drapes : ℚ := 60

def total_cost : ℚ := number_of_windows * (cost_of_sheers + cost_of_drapes)

theorem window_treatment_cost : total_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_window_treatment_cost_l687_68742


namespace NUMINAMATH_CALUDE_system_solution_l687_68749

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  2 * x - 3 * y = 5 ∧ 4 * x - y = 5

-- Theorem statement
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 1 ∧ y = -1 :=
sorry

end NUMINAMATH_CALUDE_system_solution_l687_68749


namespace NUMINAMATH_CALUDE_airplane_seats_l687_68797

theorem airplane_seats : ∀ s : ℕ,
  s ≥ 30 →
  (30 : ℝ) + 0.4 * s + (3/5) * s ≤ s →
  s = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l687_68797


namespace NUMINAMATH_CALUDE_P_on_y_axis_after_move_l687_68734

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of P -/
def P : Point := ⟨3, 4⟩

/-- Function to move a point left by a given number of units -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  ⟨p.x - units, p.y⟩

/-- Predicate to check if a point is on the y-axis -/
def isOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- Theorem stating that P lands on the y-axis after moving 3 units left -/
theorem P_on_y_axis_after_move : isOnYAxis (moveLeft P 3) := by
  sorry

end NUMINAMATH_CALUDE_P_on_y_axis_after_move_l687_68734


namespace NUMINAMATH_CALUDE_x_range_l687_68793

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem x_range (x : ℝ) :
  (f (x - 2) > f 3) → -1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l687_68793


namespace NUMINAMATH_CALUDE_certain_positive_integer_value_l687_68796

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem certain_positive_integer_value :
  ∀ (i k m n : Nat),
    factorial 8 = 2^i * 3^k * 5^m * 7^n →
    i + k + m + n = 11 →
    n = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_positive_integer_value_l687_68796


namespace NUMINAMATH_CALUDE_pattern_36_l687_68716

-- Define a function that represents the pattern
def f (n : ℕ) : ℕ :=
  if n = 1 then 6
  else if n ≤ 5 then 360 + n
  else 3600 + n

-- State the theorem
theorem pattern_36 : f 36 = 3636 := by
  sorry

end NUMINAMATH_CALUDE_pattern_36_l687_68716


namespace NUMINAMATH_CALUDE_cards_drawn_l687_68753

theorem cards_drawn (total_cards : ℕ) (face_cards : ℕ) (prob : ℚ) (n : ℕ) : 
  total_cards = 52 →
  face_cards = 12 →
  prob = 12 / 52 →
  (face_cards : ℚ) / n = prob →
  n = total_cards :=
by sorry

end NUMINAMATH_CALUDE_cards_drawn_l687_68753


namespace NUMINAMATH_CALUDE_pasture_rent_is_175_l687_68773

/-- Represents the rent share of a person based on their oxen and months of grazing -/
structure RentShare where
  oxen : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given the rent shares and one known payment -/
def calculateTotalRent (shares : List RentShare) (knownShare : RentShare) (knownPayment : ℕ) : ℕ :=
  let totalOxenMonths := shares.foldl (fun acc s => acc + s.oxen * s.months) 0
  let knownShareOxenMonths := knownShare.oxen * knownShare.months
  (totalOxenMonths * knownPayment) / knownShareOxenMonths

/-- Theorem: The total rent of the pasture is 175 given the problem conditions -/
theorem pasture_rent_is_175 :
  let shares := [
    RentShare.mk 10 7,  -- A's share
    RentShare.mk 12 5,  -- B's share
    RentShare.mk 15 3   -- C's share
  ]
  let knownShare := RentShare.mk 15 3  -- C's share
  let knownPayment := 45  -- C's payment
  calculateTotalRent shares knownShare knownPayment = 175 := by
  sorry


end NUMINAMATH_CALUDE_pasture_rent_is_175_l687_68773


namespace NUMINAMATH_CALUDE_square_plus_fifteen_perfect_square_l687_68715

theorem square_plus_fifteen_perfect_square (n : ℤ) : 
  (∃ m : ℤ, n^2 + 15 = m^2) ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_fifteen_perfect_square_l687_68715


namespace NUMINAMATH_CALUDE_solve_system_l687_68765

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x + 4 * y = 0) 
  (eq2 : y - 3 = x) : 
  5 * y = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l687_68765


namespace NUMINAMATH_CALUDE_ryan_study_difference_l687_68707

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ

/-- The difference in hours between English and Chinese study time -/
def study_time_difference (schedule : StudySchedule) : ℤ :=
  schedule.english_hours - schedule.chinese_hours

/-- Theorem: Ryan spends 4 more hours on English than Chinese -/
theorem ryan_study_difference :
  ∀ (schedule : StudySchedule),
  schedule.english_hours = 6 →
  schedule.chinese_hours = 2 →
  study_time_difference schedule = 4 := by
sorry

end NUMINAMATH_CALUDE_ryan_study_difference_l687_68707


namespace NUMINAMATH_CALUDE_convex_polygon_division_theorem_l687_68768

-- Define a type for polygons
def Polygon : Type := Set (ℝ × ℝ)

-- Define a type for motions (transformations)
def Motion : Type := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a predicate for convex polygons
def IsConvex (p : Polygon) : Prop := sorry

-- Define a predicate for orientation-preserving motions
def IsOrientationPreserving (m : Motion) : Prop := sorry

-- Define a predicate for a polygon being dividable by a broken line into two polygons
def DividableByBrokenLine (p : Polygon) (p1 p2 : Polygon) : Prop := sorry

-- Define a predicate for a polygon being dividable by a segment into two polygons
def DividableBySegment (p : Polygon) (p1 p2 : Polygon) : Prop := sorry

-- Define a predicate for two polygons being transformable into each other by a motion
def Transformable (p1 p2 : Polygon) (m : Motion) : Prop := sorry

-- State the theorem
theorem convex_polygon_division_theorem (p : Polygon) :
  IsConvex p →
  (∃ (p1 p2 : Polygon) (m : Motion), 
    DividableByBrokenLine p p1 p2 ∧ 
    IsOrientationPreserving m ∧ 
    Transformable p1 p2 m) →
  (∃ (q1 q2 : Polygon) (n : Motion), 
    DividableBySegment p q1 q2 ∧ 
    IsOrientationPreserving n ∧ 
    Transformable q1 q2 n) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_division_theorem_l687_68768


namespace NUMINAMATH_CALUDE_equation_solution_l687_68732

theorem equation_solution : ∃ x : ℝ, (16 : ℝ) ^ (2 * x - 3) = (1 / 2 : ℝ) ^ (x + 8) ↔ x = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l687_68732


namespace NUMINAMATH_CALUDE_largest_three_digit_with_digit_product_8_l687_68735

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem largest_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → n ≤ 811 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_with_digit_product_8_l687_68735


namespace NUMINAMATH_CALUDE_no_prime_arrangement_with_natural_expression_l687_68710

theorem no_prime_arrangement_with_natural_expression :
  ¬ ∃ (p : ℕ → ℕ),
    (∀ n, Prime (p n)) ∧
    (∀ q : ℕ, Prime q → ∃ n, p n = q) ∧
    (∀ i : ℕ, ∃ k : ℕ, (p i * p (i + 1) - p (i + 2)^2) / (p i + p (i + 1)) = k) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_arrangement_with_natural_expression_l687_68710


namespace NUMINAMATH_CALUDE_max_andy_cookies_l687_68779

def total_cookies : ℕ := 30

def valid_distribution (andy_cookies : ℕ) : Prop :=
  andy_cookies + 3 * andy_cookies ≤ total_cookies

theorem max_andy_cookies :
  ∃ (max : ℕ), valid_distribution max ∧
    ∀ (n : ℕ), valid_distribution n → n ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_andy_cookies_l687_68779


namespace NUMINAMATH_CALUDE_min_cubes_needed_l687_68770

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := ℕ → ℕ → ℕ → Bool

/-- Checks if a cube is present at the given coordinates -/
def has_cube (grid : CubeGrid) (x y z : ℕ) : Prop := grid x y z = true

/-- Checks if the grid satisfies the adjacency condition -/
def satisfies_adjacency (grid : CubeGrid) : Prop :=
  ∀ x y z, has_cube grid x y z →
    (has_cube grid (x+1) y z ∨ has_cube grid (x-1) y z ∨
     has_cube grid x (y+1) z ∨ has_cube grid x (y-1) z ∨
     has_cube grid x y (z+1) ∨ has_cube grid x y (z-1))

/-- Checks if the grid matches the given front view -/
def matches_front_view (grid : CubeGrid) : Prop :=
  (has_cube grid 0 0 0) ∧ (has_cube grid 0 1 0) ∧ (has_cube grid 0 2 0) ∧
  (has_cube grid 1 0 0) ∧ (has_cube grid 1 1 0) ∧
  (has_cube grid 2 0 0) ∧ (has_cube grid 2 1 0)

/-- Checks if the grid matches the given side view -/
def matches_side_view (grid : CubeGrid) : Prop :=
  (has_cube grid 0 0 0) ∧ (has_cube grid 1 0 0) ∧ (has_cube grid 2 0 0) ∧
  (has_cube grid 2 0 1) ∧
  (has_cube grid 2 0 2)

/-- Counts the number of cubes in the grid -/
def count_cubes (grid : CubeGrid) : ℕ :=
  sorry -- Implementation omitted

/-- The main theorem to be proved -/
theorem min_cubes_needed :
  ∃ (grid : CubeGrid),
    satisfies_adjacency grid ∧
    matches_front_view grid ∧
    matches_side_view grid ∧
    count_cubes grid = 5 ∧
    (∀ (other_grid : CubeGrid),
      satisfies_adjacency other_grid →
      matches_front_view other_grid →
      matches_side_view other_grid →
      count_cubes other_grid ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_min_cubes_needed_l687_68770


namespace NUMINAMATH_CALUDE_work_completion_time_l687_68781

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 5

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 12

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

theorem work_completion_time :
  (y_worked / y_days) + (x_remaining / x_days) = 1 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l687_68781


namespace NUMINAMATH_CALUDE_contrapositive_of_zero_product_l687_68740

theorem contrapositive_of_zero_product (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) →
  (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_of_zero_product_l687_68740


namespace NUMINAMATH_CALUDE_water_level_decrease_l687_68782

def water_level_change (change : ℝ) : ℝ := change

theorem water_level_decrease (decrease : ℝ) : 
  water_level_change (-decrease) = -decrease :=
by sorry

end NUMINAMATH_CALUDE_water_level_decrease_l687_68782


namespace NUMINAMATH_CALUDE_john_annual_oil_change_cost_l687_68748

/-- Calculates the annual cost of oil changes for a driver --/
def annual_oil_change_cost (miles_per_month : ℕ) (miles_per_oil_change : ℕ) (free_changes_per_year : ℕ) (cost_per_change : ℕ) : ℕ :=
  let total_miles := miles_per_month * 12
  let total_changes := total_miles / miles_per_oil_change
  let paid_changes := total_changes - free_changes_per_year
  paid_changes * cost_per_change

/-- Theorem stating that John pays $150 a year for oil changes --/
theorem john_annual_oil_change_cost :
  annual_oil_change_cost 1000 3000 1 50 = 150 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_oil_change_cost_l687_68748


namespace NUMINAMATH_CALUDE_fruit_mix_kiwis_l687_68724

theorem fruit_mix_kiwis (total : ℕ) (s b o k : ℕ) : 
  total = 340 →
  s + b + o + k = total →
  s = 3 * b →
  o = 2 * k →
  k = 5 * s →
  k = 104 := by
  sorry

end NUMINAMATH_CALUDE_fruit_mix_kiwis_l687_68724


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l687_68705

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l687_68705


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l687_68783

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m
  (∃ x : ℝ, f x = 0) ↔ m ≤ 4 ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁^2 + x₂^2 + (x₁*x₂)^2 = 40 → m = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l687_68783


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l687_68759

/-- The volume of a rectangular parallelepiped -/
def volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a rectangular parallelepiped with width 15 cm, length 6 cm, and height 4 cm is 360 cubic centimeters -/
theorem rectangular_parallelepiped_volume :
  volume 15 6 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l687_68759


namespace NUMINAMATH_CALUDE_g_is_odd_l687_68712

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_g_is_odd_l687_68712


namespace NUMINAMATH_CALUDE_epidemic_competition_theorem_l687_68726

/-- Represents a participant in the competition -/
structure Participant where
  first_round_prob : ℚ
  second_round_prob : ℚ

/-- Calculates the probability of a participant winning both rounds -/
def win_prob (p : Participant) : ℚ :=
  p.first_round_prob * p.second_round_prob

/-- Calculates the probability of at least one participant winning -/
def at_least_one_wins (p1 p2 : Participant) : ℚ :=
  1 - (1 - win_prob p1) * (1 - win_prob p2)

theorem epidemic_competition_theorem 
  (A B : Participant)
  (h_A_first : A.first_round_prob = 5/6)
  (h_A_second : A.second_round_prob = 2/3)
  (h_B_first : B.first_round_prob = 3/5)
  (h_B_second : B.second_round_prob = 3/4) :
  win_prob A > win_prob B ∧ at_least_one_wins A B = 34/45 := by
  sorry

end NUMINAMATH_CALUDE_epidemic_competition_theorem_l687_68726


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l687_68711

/-- The probability that a randomly chosen point in a circle of radius 3 
    is closer to the center than to the boundary -/
theorem probability_closer_to_center (r : ℝ) (h : r = 3) : 
  (π * (r/2)^2) / (π * r^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l687_68711


namespace NUMINAMATH_CALUDE_tangent_circle_existence_and_radius_l687_68728

/-- Given three circles with radii r₁, r₂, r₃, where r₁ > r₂ and r₁ > r₃,
    there exists a circle touching the four tangents drawn as described,
    with radius (r₁ * r₂ * r₃) / (r₁ * (r₂ + r₃) - r₂ * r₃) -/
theorem tangent_circle_existence_and_radius 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ > r₂) 
  (h₂ : r₁ > r₃) 
  (h₃ : r₁ > 0) 
  (h₄ : r₂ > 0) 
  (h₅ : r₃ > 0) :
  ∃ (r : ℝ), r = (r₁ * r₂ * r₃) / (r₁ * (r₂ + r₃) - r₂ * r₃) ∧ 
  r > 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_existence_and_radius_l687_68728


namespace NUMINAMATH_CALUDE_bill_sunday_run_l687_68738

/-- Represents the miles run by Bill, Julia, and Mark over a weekend -/
structure WeekendRun where
  billSaturday : ℝ
  billSunday : ℝ
  juliaSunday : ℝ
  markSaturday : ℝ
  markSunday : ℝ

/-- Conditions for the weekend run -/
def weekendRunConditions (run : WeekendRun) : Prop :=
  run.billSunday = run.billSaturday + 4 ∧
  run.juliaSunday = 2 * run.billSunday ∧
  run.markSaturday = 5 ∧
  run.markSunday = run.markSaturday + 2 ∧
  run.billSaturday + run.billSunday + run.juliaSunday + run.markSaturday + run.markSunday = 50

/-- Theorem stating that under the given conditions, Bill ran 10.5 miles on Sunday -/
theorem bill_sunday_run (run : WeekendRun) (h : weekendRunConditions run) : 
  run.billSunday = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_bill_sunday_run_l687_68738


namespace NUMINAMATH_CALUDE_stratified_sample_medium_stores_l687_68791

/-- Given a population of stores with a known number of medium-sized stores,
    calculate the number of medium-sized stores in a stratified sample. -/
theorem stratified_sample_medium_stores
  (total_stores : ℕ)
  (medium_stores : ℕ)
  (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (medium_stores : ℚ) / total_stores * sample_size = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_medium_stores_l687_68791


namespace NUMINAMATH_CALUDE_no_even_rectangle_with_sum_120_l687_68746

/-- Represents a rectangle with positive even integer side lengths -/
structure EvenRectangle where
  length : ℕ
  width : ℕ
  length_positive : length > 0
  width_positive : width > 0
  length_even : Even length
  width_even : Even width

/-- Calculates the area of an EvenRectangle -/
def area (r : EvenRectangle) : ℕ := r.length * r.width

/-- Calculates the modified perimeter of an EvenRectangle -/
def modifiedPerimeter (r : EvenRectangle) : ℕ := 2 * (r.length + r.width) + 6

/-- Theorem stating that there's no EvenRectangle with A + P' = 120 -/
theorem no_even_rectangle_with_sum_120 :
  ∀ r : EvenRectangle, area r + modifiedPerimeter r ≠ 120 := by
  sorry

end NUMINAMATH_CALUDE_no_even_rectangle_with_sum_120_l687_68746


namespace NUMINAMATH_CALUDE_max_profit_is_1200_l687_68722

/-- Represents the cost and profit calculation for a shopping mall's purchasing plan. -/
structure ShoppingMall where
  cost_A : ℝ  -- Cost price of good A
  cost_B : ℝ  -- Cost price of good B
  sell_A : ℝ  -- Selling price of good A
  sell_B : ℝ  -- Selling price of good B
  total_units : ℕ  -- Total units to purchase

/-- Calculates the profit for a given purchasing plan. -/
def profit (sm : ShoppingMall) (units_A : ℕ) : ℝ :=
  let units_B := sm.total_units - units_A
  (sm.sell_A * units_A + sm.sell_B * units_B) - (sm.cost_A * units_A + sm.cost_B * units_B)

/-- Theorem stating that the maximum profit is $1200 under the given conditions. -/
theorem max_profit_is_1200 (sm : ShoppingMall) 
  (h1 : sm.cost_A + 3 * sm.cost_B = 240)
  (h2 : 2 * sm.cost_A + sm.cost_B = 130)
  (h3 : sm.sell_A = 40)
  (h4 : sm.sell_B = 90)
  (h5 : sm.total_units = 100)
  : ∃ (units_A : ℕ), 
    units_A ≥ 4 * (sm.total_units - units_A) ∧ 
    ∀ (x : ℕ), x ≥ 4 * (sm.total_units - x) → profit sm units_A ≥ profit sm x :=
by sorry

end NUMINAMATH_CALUDE_max_profit_is_1200_l687_68722


namespace NUMINAMATH_CALUDE_jeff_matches_won_l687_68758

/-- Represents the duration of the tennis competition in minutes -/
def total_playtime : ℕ := 225

/-- Represents the time in minutes it takes Jeff to score a point -/
def minutes_per_point : ℕ := 7

/-- Represents the minimum number of points required to win a match -/
def points_to_win : ℕ := 12

/-- Represents the break time in minutes between matches -/
def break_time : ℕ := 5

/-- Calculates the total number of points Jeff scored during the competition -/
def total_points : ℕ := total_playtime / minutes_per_point

/-- Calculates the duration of a single match in minutes, including playtime and break time -/
def match_duration : ℕ := points_to_win * minutes_per_point + break_time

/-- Represents the number of matches Jeff won during the competition -/
def matches_won : ℕ := total_playtime / match_duration

theorem jeff_matches_won : matches_won = 2 := by sorry

end NUMINAMATH_CALUDE_jeff_matches_won_l687_68758


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l687_68757

/-- Given two vectors a and b in ℝ², where a = (2, 3) and b = (k, -1),
    if a is perpendicular to b, then k = 3/2. -/
theorem perpendicular_vectors_k_value :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![k, -1]
  (∀ i, i < 2 → a i * b i = 0) →
  k = 3/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l687_68757


namespace NUMINAMATH_CALUDE_certain_number_problem_l687_68739

theorem certain_number_problem (x : ℝ) : 
  (10 + 20 + 60) / 3 = ((x + 40 + 25) / 3 + 5) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l687_68739


namespace NUMINAMATH_CALUDE_exactly_five_triangles_l687_68737

/-- A triangle with integral side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  sum_eq_8 : a + b + c = 8
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

/-- Count of distinct triangles with perimeter 8 -/
def count_triangles : ℕ := sorry

/-- The main theorem stating there are exactly 5 such triangles -/
theorem exactly_five_triangles : count_triangles = 5 := by sorry

end NUMINAMATH_CALUDE_exactly_five_triangles_l687_68737


namespace NUMINAMATH_CALUDE_balloon_count_l687_68736

/-- The number of gold balloons -/
def gold_balloons : ℕ := sorry

/-- The number of silver balloons -/
def silver_balloons : ℕ := sorry

/-- The number of black balloons -/
def black_balloons : ℕ := 150

theorem balloon_count : 
  (silver_balloons = 2 * gold_balloons) ∧ 
  (gold_balloons + silver_balloons + black_balloons = 573) → 
  gold_balloons = 141 :=
by sorry

end NUMINAMATH_CALUDE_balloon_count_l687_68736


namespace NUMINAMATH_CALUDE_yellow_balls_count_l687_68776

theorem yellow_balls_count (red white : ℕ) (a : ℝ) :
  red = 2 →
  white = 4 →
  (a / (red + white + a) = 1 / 4) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l687_68776


namespace NUMINAMATH_CALUDE_five_people_seven_chairs_arrangement_l687_68777

/-- The number of ways to arrange people in chairs with one person fixed -/
def arrangement_count (total_chairs : ℕ) (total_people : ℕ) (fixed_position : ℕ) : ℕ :=
  (total_chairs - 1).factorial / (total_chairs - total_people).factorial

/-- Theorem: Five people can be arranged in seven chairs with one person fixed in the middle in 360 ways -/
theorem five_people_seven_chairs_arrangement : 
  arrangement_count 7 5 4 = 360 := by
sorry

end NUMINAMATH_CALUDE_five_people_seven_chairs_arrangement_l687_68777


namespace NUMINAMATH_CALUDE_pages_difference_l687_68788

/-- The number of pages Person A reads per day -/
def pages_per_day_A : ℕ := 8

/-- The number of pages Person B reads per day -/
def pages_per_day_B : ℕ := 13

/-- The number of days we're considering -/
def days : ℕ := 7

/-- The total number of pages Person A reads in the given number of days -/
def total_pages_A : ℕ := pages_per_day_A * days

/-- The total number of pages Person B reads in the given number of days -/
def total_pages_B : ℕ := pages_per_day_B * days

theorem pages_difference : total_pages_B - total_pages_A = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l687_68788


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_hypotenuse_ratio_l687_68795

theorem right_triangle_perimeter_hypotenuse_ratio 
  (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a := 3*x + 3*y
  let b := 4*x
  let c := 4*y
  let perimeter := a + b + c
  (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2) →
  (perimeter / a = 7/3 ∨ perimeter / b = 56/25 ∨ perimeter / c = 56/25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_hypotenuse_ratio_l687_68795


namespace NUMINAMATH_CALUDE_both_runners_in_picture_probability_l687_68794

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Calculates the probability of both runners being in a picture -/
def probability_both_in_picture (rachel : Runner) (robert : Runner) : ℚ :=
  sorry

/-- Main theorem: The probability of both runners being in the picture is 3/16 -/
theorem both_runners_in_picture_probability :
  let rachel : Runner := { lapTime := 90, direction := true }
  let robert : Runner := { lapTime := 80, direction := false }
  probability_both_in_picture rachel robert = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_both_runners_in_picture_probability_l687_68794


namespace NUMINAMATH_CALUDE_max_distance_covered_l687_68799

/-- The maximum distance a person can cover in 6 hours, 
    given that they travel at 5 km/hr for half the distance 
    and 4 km/hr for the other half. -/
theorem max_distance_covered (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 6 →
  speed1 = 5 →
  speed2 = 4 →
  (total_time * speed1 * speed2) / (speed1 + speed2) = 120 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_covered_l687_68799


namespace NUMINAMATH_CALUDE_first_three_digits_after_decimal_l687_68713

theorem first_three_digits_after_decimal (n : ℕ) (x : ℝ) :
  n = 1200 →
  x = (10^n + 1)^(5/3) →
  ∃ (k : ℕ), x = k + 0.333 + r ∧ r < 0.001 :=
sorry

end NUMINAMATH_CALUDE_first_three_digits_after_decimal_l687_68713


namespace NUMINAMATH_CALUDE_books_left_to_read_l687_68774

theorem books_left_to_read 
  (total_books : ℕ) 
  (books_read : ℕ) 
  (h1 : total_books = 19) 
  (h2 : books_read = 4) : 
  total_books - books_read = 15 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l687_68774


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l687_68717

/-- The area of a right triangle with legs of length 3 and 5 is 7.5 -/
theorem right_triangle_area : Real → Prop :=
  fun a => 
    ∃ (b h : Real),
      b = 3 ∧
      h = 5 ∧
      a = (1 / 2) * b * h ∧
      a = 7.5

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l687_68717


namespace NUMINAMATH_CALUDE_divisibility_of_power_minus_one_l687_68756

theorem divisibility_of_power_minus_one (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n ^ (n - 1) : ℤ) - 1 = k * ((n - 1) ^ 2 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_power_minus_one_l687_68756


namespace NUMINAMATH_CALUDE_cola_price_is_three_l687_68727

/-- Represents the cost and quantity of drinks sold in a store --/
structure DrinkSales where
  cola_price : ℝ
  cola_quantity : ℕ
  juice_price : ℝ
  juice_quantity : ℕ
  water_price : ℝ
  water_quantity : ℕ
  total_earnings : ℝ

/-- Theorem stating that the cola price is $3 given the specific sales conditions --/
theorem cola_price_is_three (sales : DrinkSales)
  (h_juice_price : sales.juice_price = 1.5)
  (h_water_price : sales.water_price = 1)
  (h_cola_quantity : sales.cola_quantity = 15)
  (h_juice_quantity : sales.juice_quantity = 12)
  (h_water_quantity : sales.water_quantity = 25)
  (h_total_earnings : sales.total_earnings = 88) :
  sales.cola_price = 3 := by
  sorry

#check cola_price_is_three

end NUMINAMATH_CALUDE_cola_price_is_three_l687_68727


namespace NUMINAMATH_CALUDE_complex_argument_range_l687_68771

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + z⁻¹) = 1) :
  ∃ k : ℤ, k ∈ ({0, 1} : Set ℤ) ∧
  k * π + π / 2 - Real.arccos (3 / 4) / 2 ≤ Complex.arg z ∧
  Complex.arg z ≤ k * π + π / 2 + Real.arccos (3 / 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_argument_range_l687_68771


namespace NUMINAMATH_CALUDE_parcel_weight_sum_l687_68752

/-- Given three parcels with weights x, y, and z, prove that their total weight is 209 pounds. -/
theorem parcel_weight_sum (x y z : ℝ) 
  (h1 : x + y = 132)
  (h2 : y + z = 146)
  (h3 : z + x = 140) : 
  x + y + z = 209 := by
  sorry

end NUMINAMATH_CALUDE_parcel_weight_sum_l687_68752


namespace NUMINAMATH_CALUDE_arrangements_equal_78_l687_68704

/-- The number of different arrangements to select 2 workers for typesetting and 2 for printing
    from a group of 7 workers, where 5 are proficient in typesetting and 4 are proficient in printing. -/
def num_arrangements (total : ℕ) (typesetters : ℕ) (printers : ℕ) (typeset_needed : ℕ) (print_needed : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 78 -/
theorem arrangements_equal_78 :
  num_arrangements 7 5 4 2 2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_equal_78_l687_68704


namespace NUMINAMATH_CALUDE_girls_divisible_by_nine_l687_68769

theorem girls_divisible_by_nine (N : Nat) (m c d u : Nat) : 
  N < 10000 →
  N = 1000 * m + 100 * c + 10 * d + u →
  let B := m + c + d + u
  let G := N - B
  G % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_girls_divisible_by_nine_l687_68769


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l687_68790

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l687_68790


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l687_68785

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l687_68785


namespace NUMINAMATH_CALUDE_birds_on_fence_l687_68784

theorem birds_on_fence : ∃ x : ℕ, (2 * x + 10 = 50) ∧ (x = 20) :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l687_68784


namespace NUMINAMATH_CALUDE_quadratic_max_value_l687_68718

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem quadratic_max_value (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, f x m ≤ 1) ∧ 
  (∃ x ∈ Set.Icc 0 3, f x m = 1) → 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l687_68718


namespace NUMINAMATH_CALUDE_dress_hemming_time_l687_68764

/-- The time required to hem a dress given its length, stitch size, and stitching rate -/
theorem dress_hemming_time 
  (dress_length : ℝ) 
  (stitch_length : ℝ) 
  (stitches_per_minute : ℝ) 
  (h1 : dress_length = 3) -- dress length in feet
  (h2 : stitch_length = 1/4 / 12) -- stitch length in feet (1/4 inch converted to feet)
  (h3 : stitches_per_minute = 24) :
  dress_length / (stitch_length * stitches_per_minute) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dress_hemming_time_l687_68764


namespace NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l687_68792

theorem marks_lost_per_wrong_answer 
  (total_questions : ℕ)
  (marks_per_correct : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (h1 : total_questions = 60)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 160)
  (h4 : correct_answers = 44)
  : ℕ :=
by
  sorry

#check marks_lost_per_wrong_answer

end NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l687_68792


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_difference_l687_68760

theorem two_numbers_sum_product_difference (n : ℕ) (hn : n = 38) :
  ∃ x y : ℕ,
    1 ≤ x ∧ x < y ∧ y ≤ n ∧
    (n * (n + 1)) / 2 - x - y = x * y ∧
    y - x = 39 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_difference_l687_68760


namespace NUMINAMATH_CALUDE_work_done_stretching_spring_l687_68702

/-- Work done by stretching a spring -/
theorem work_done_stretching_spring
  (force : ℝ) (compression : ℝ) (stretch : ℝ)
  (hf : force = 10)
  (hc : compression = 0.1)
  (hs : stretch = 0.06)
  : (1/2) * (force / compression) * stretch^2 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_work_done_stretching_spring_l687_68702
