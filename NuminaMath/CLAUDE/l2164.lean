import Mathlib

namespace NUMINAMATH_CALUDE_punger_baseball_cards_l2164_216486

/-- Given the number of packs, cards per pack, and cards per page, 
    calculate the number of pages needed to store all cards. -/
def pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack + cards_per_page - 1) / cards_per_page

theorem punger_baseball_cards : 
  pages_needed 60 7 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_punger_baseball_cards_l2164_216486


namespace NUMINAMATH_CALUDE_flash_catches_ace_l2164_216465

/-- The distance Flash must run to catch Ace -/
def flashDistance (x v c y : ℝ) : ℝ := 2 * y

theorem flash_catches_ace (x v c y : ℝ) 
  (hx : x > 1) 
  (hc : c > 0) : 
  flashDistance x v c y = 2 * y := by
  sorry

#check flash_catches_ace

end NUMINAMATH_CALUDE_flash_catches_ace_l2164_216465


namespace NUMINAMATH_CALUDE_three_in_range_of_f_l2164_216476

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real b, 3 is in the range of f(x) = x^2 + bx - 1 -/
theorem three_in_range_of_f (b : ℝ) : ∃ x, f b x = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_range_of_f_l2164_216476


namespace NUMINAMATH_CALUDE_increase_decrease_calculation_l2164_216451

theorem increase_decrease_calculation (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) : 
  initial = 80 → 
  increase_percent = 150 → 
  decrease_percent = 20 → 
  (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) = 160 := by
sorry

end NUMINAMATH_CALUDE_increase_decrease_calculation_l2164_216451


namespace NUMINAMATH_CALUDE_half_work_completed_l2164_216470

/-- Represents the highway construction project -/
structure HighwayProject where
  initialMen : ℕ
  totalLength : ℝ
  initialDays : ℕ
  initialHoursPerDay : ℕ
  actualDays : ℕ
  additionalMen : ℕ
  newHoursPerDay : ℕ

/-- Calculates the fraction of work completed -/
def fractionCompleted (project : HighwayProject) : ℚ :=
  let initialManHours := project.initialMen * project.initialDays * project.initialHoursPerDay
  let actualManHours := project.initialMen * project.actualDays * project.initialHoursPerDay
  actualManHours / initialManHours

/-- Theorem stating that the fraction of work completed is 1/2 -/
theorem half_work_completed (project : HighwayProject) 
  (h1 : project.initialMen = 100)
  (h2 : project.totalLength = 2)
  (h3 : project.initialDays = 50)
  (h4 : project.initialHoursPerDay = 8)
  (h5 : project.actualDays = 25)
  (h6 : project.additionalMen = 60)
  (h7 : project.newHoursPerDay = 10)
  (h8 : (project.initialMen + project.additionalMen) * (project.initialDays - project.actualDays) * project.newHoursPerDay = project.initialMen * project.initialDays * project.initialHoursPerDay / 2) :
  fractionCompleted project = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_half_work_completed_l2164_216470


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2164_216446

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 2 ∧ ∃ (x' y' : ℝ), 4 < x' ∧ x' < 8 ∧ 8 < y' ∧ y' < 12 ∧ (⌊y'⌋ - ⌈x'⌉ : ℤ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2164_216446


namespace NUMINAMATH_CALUDE_calculate_expression_l2164_216422

theorem calculate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a*b^2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2164_216422


namespace NUMINAMATH_CALUDE_sugar_for_muffins_l2164_216459

/-- Given a recipe that requires 3 cups of sugar for 24 muffins,
    calculate the number of cups of sugar needed for 72 muffins. -/
theorem sugar_for_muffins (recipe_muffins : ℕ) (recipe_sugar : ℕ) (target_muffins : ℕ) :
  recipe_muffins = 24 →
  recipe_sugar = 3 →
  target_muffins = 72 →
  (target_muffins * recipe_sugar) / recipe_muffins = 9 :=
by
  sorry

#check sugar_for_muffins

end NUMINAMATH_CALUDE_sugar_for_muffins_l2164_216459


namespace NUMINAMATH_CALUDE_equation_solution_l2164_216439

theorem equation_solution : ∃! x : ℝ, -2 * x^2 = (4*x + 2) / (x + 4) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2164_216439


namespace NUMINAMATH_CALUDE_tuna_weight_l2164_216462

/-- A fish market scenario where we need to determine the weight of each tuna. -/
theorem tuna_weight (total_customers : ℕ) (num_tuna : ℕ) (pounds_per_customer : ℕ) (unserved_customers : ℕ) :
  total_customers = 100 →
  num_tuna = 10 →
  pounds_per_customer = 25 →
  unserved_customers = 20 →
  (total_customers - unserved_customers) * pounds_per_customer / num_tuna = 200 := by
sorry

end NUMINAMATH_CALUDE_tuna_weight_l2164_216462


namespace NUMINAMATH_CALUDE_max_consecutive_sum_is_six_l2164_216434

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The target sum -/
def target_sum : ℕ := 21

/-- The property that n consecutive integers sum to the target -/
def sum_to_target (n : ℕ) : Prop :=
  sum_first_n n = target_sum

/-- The maximum number of consecutive positive integers that sum to the target -/
def max_consecutive_sum : ℕ := 6

theorem max_consecutive_sum_is_six :
  (sum_to_target max_consecutive_sum) ∧
  (∀ k : ℕ, k > max_consecutive_sum → ¬(sum_to_target k)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_is_six_l2164_216434


namespace NUMINAMATH_CALUDE_distinct_arrangements_statistics_l2164_216479

def word_length : ℕ := 10
def letter_counts : List ℕ := [3, 2, 2, 1, 1]

theorem distinct_arrangements_statistics :
  (word_length.factorial) / ((letter_counts.map Nat.factorial).prod) = 75600 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_statistics_l2164_216479


namespace NUMINAMATH_CALUDE_gamma_cheaper_at_11_gamma_not_cheaper_at_10_min_shirts_for_gamma_cheaper_l2164_216469

/-- Represents the cost function for a t-shirt company -/
structure TShirtCompany where
  setupFee : ℕ
  costPerShirt : ℕ

/-- Calculates the total cost for a given number of shirts -/
def totalCost (company : TShirtCompany) (shirts : ℕ) : ℕ :=
  company.setupFee + company.costPerShirt * shirts

/-- The Acme T-Shirt Company -/
def acme : TShirtCompany := ⟨40, 10⟩

/-- The Beta T-Shirt Company -/
def beta : TShirtCompany := ⟨0, 15⟩

/-- The Gamma T-Shirt Company -/
def gamma : TShirtCompany := ⟨20, 12⟩

theorem gamma_cheaper_at_11 :
  totalCost gamma 11 < totalCost acme 11 ∧
  totalCost gamma 11 < totalCost beta 11 :=
sorry

theorem gamma_not_cheaper_at_10 :
  ¬(totalCost gamma 10 < totalCost acme 10 ∧
    totalCost gamma 10 < totalCost beta 10) :=
sorry

theorem min_shirts_for_gamma_cheaper : ℕ :=
  11

end NUMINAMATH_CALUDE_gamma_cheaper_at_11_gamma_not_cheaper_at_10_min_shirts_for_gamma_cheaper_l2164_216469


namespace NUMINAMATH_CALUDE_equation_solution_l2164_216415

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.07 * (25 + x) = 15.1 ∧ x = 111.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2164_216415


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_curve_l2164_216498

/-- The value of 'a' for which the asymptotes of the hyperbola x²/9 - y²/4 = 1 
    are precisely the two tangent lines of the curve y = ax² + 1/3 -/
theorem hyperbola_asymptotes_tangent_curve (a : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/4 = 1 → 
    ∃ k : ℝ, (y = k*x ∨ y = -k*x) ∧ 
    (∀ x₀ : ℝ, (k*x₀ = a*x₀^2 + 1/3 → 
      ∀ x : ℝ, k*x ≤ a*x^2 + 1/3) ∧
    (-k*x₀ = a*x₀^2 + 1/3 → 
      ∀ x : ℝ, -k*x ≤ a*x^2 + 1/3))) →
  a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_curve_l2164_216498


namespace NUMINAMATH_CALUDE_triangle_solutions_l2164_216416

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    this theorem proves that if a = 6, b = 6√3, and A = π/6,
    then there are two possible solutions for the triangle. -/
theorem triangle_solutions (a b c : ℝ) (A B C : ℝ) :
  a = 6 →
  b = 6 * Real.sqrt 3 →
  A = π / 6 →
  (B = π / 3 ∧ C = π / 2 ∧ c = 12) ∨
  (B = 2 * π / 3 ∧ C = π / 6 ∧ c = 6) :=
sorry

end NUMINAMATH_CALUDE_triangle_solutions_l2164_216416


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2164_216436

theorem root_sum_theorem (a b : ℝ) 
  (h1 : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + a*r1 + b = 0) ∧ (r2^2 + a*r2 + b = 0))
  (h2 : ∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + b*s1 + a = 0) ∧ (s2^2 + b*s2 + a = 0))
  (h3 : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
    ((t1^2 + a*t1 + b) * (t1^2 + b*t1 + a) = 0) ∧
    ((t2^2 + a*t2 + b) * (t2^2 + b*t2 + a) = 0) ∧
    ((t3^2 + a*t3 + b) * (t3^2 + b*t3 + a) = 0)) :
  t1 + t2 + t3 = -2 := by sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2164_216436


namespace NUMINAMATH_CALUDE_green_peaches_count_l2164_216455

/-- Represents a basket of peaches -/
structure Basket where
  total : Nat
  red : Nat
  green : Nat

/-- The number of green peaches in a basket -/
def greenPeaches (b : Basket) : Nat := b.green

/-- Theorem: Given a basket with 10 total peaches and 7 red peaches, 
    the number of green peaches is 3 -/
theorem green_peaches_count (b : Basket) 
  (h1 : b.total = 10) 
  (h2 : b.red = 7) 
  (h3 : b.green = b.total - b.red) : 
  greenPeaches b = 3 := by
  sorry

#check green_peaches_count

end NUMINAMATH_CALUDE_green_peaches_count_l2164_216455


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l2164_216400

def f (x : ℝ) : ℝ := (x - 2)^2

-- Property 1: f(x+2) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f (-x + 2)

-- Property 2: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
def is_decreasing_then_increasing (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x < y ∧ y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x ∧ x < y → f x < f y)

theorem f_satisfies_properties : 
  is_even_shifted f ∧ is_decreasing_then_increasing f :=
sorry

end NUMINAMATH_CALUDE_f_satisfies_properties_l2164_216400


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2164_216491

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 12 = 54 → Nat.gcd n 12 = 8 → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2164_216491


namespace NUMINAMATH_CALUDE_second_number_is_30_l2164_216484

theorem second_number_is_30 (a b c : ℚ) 
  (sum_eq : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8) :
  b = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_30_l2164_216484


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2164_216487

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 3) :
  (1/a + 1/b + 1/c) ≥ 3 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 1/x + 1/y + 1/z = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2164_216487


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2164_216464

theorem solution_of_linear_equation (a : ℚ) : 
  (∃ x y : ℚ, x = 2 ∧ y = 2 ∧ a * x + y = 5) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2164_216464


namespace NUMINAMATH_CALUDE_final_rope_length_l2164_216406

def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss : ℝ := 1.2
def num_knots : ℕ := rope_lengths.length - 1

theorem final_rope_length :
  (rope_lengths.sum - num_knots * knot_loss : ℝ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_rope_length_l2164_216406


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l2164_216480

theorem rectangle_width_proof (w : ℝ) (h1 : w > 0) : 
  (∃ l : ℝ, l > 0 ∧ l = 3 * w ∧ l + w = 3 * (l * w)) → w = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l2164_216480


namespace NUMINAMATH_CALUDE_divisibility_property_l2164_216426

theorem divisibility_property (m n p : ℕ) (h_prime : Nat.Prime p) 
  (h_order : m < n ∧ n < p) (h_div_m : p ∣ m^2 + 1) (h_div_n : p ∣ n^2 + 1) : 
  p ∣ m * n - 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2164_216426


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2164_216441

/-- Given a quadratic equation x^2 - 9x + 18 = 0, if its roots represent the base and legs
    of an isosceles triangle, then the perimeter of the triangle is 15. -/
theorem isosceles_triangle_perimeter (x : ℝ) : 
  x^2 - 9*x + 18 = 0 →
  ∃ (base leg : ℝ), 
    (x = base ∨ x = leg) ∧ 
    (base > 0 ∧ leg > 0) ∧
    (2*leg > base) ∧
    (base + 2*leg = 15) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2164_216441


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2164_216418

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2164_216418


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2164_216453

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 5*a + 5 = 0) → (b^2 - 5*b + 5 = 0) → a^2 + b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2164_216453


namespace NUMINAMATH_CALUDE_apple_distribution_l2164_216430

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem: There are 190 ways to distribute 30 apples among 3 people, with each person receiving at least 4 apples -/
theorem apple_distribution : distribution_ways 30 3 4 = 190 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l2164_216430


namespace NUMINAMATH_CALUDE_quadratic_binomial_square_l2164_216445

theorem quadratic_binomial_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 50*x + c = (x - a)^2) → c = 625 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_binomial_square_l2164_216445


namespace NUMINAMATH_CALUDE_mutual_win_exists_l2164_216485

/-- Represents the result of a match between two teams -/
inductive MatchResult
| Win
| Draw
| Loss

/-- Calculates points for a given match result -/
def points (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 2
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents a tournament with given number of teams -/
structure Tournament (n : Nat) where
  firstRound : Fin n → Fin n → MatchResult
  secondRound : Fin n → Fin n → MatchResult

/-- Calculates total points for a team after both rounds -/
def totalPoints (t : Tournament n) (team : Fin n) : Nat :=
  sorry

/-- Checks if all teams have different points after the first round -/
def allDifferentFirstRound (t : Tournament n) : Prop :=
  sorry

/-- Checks if all teams have the same points after both rounds -/
def allSameTotal (t : Tournament n) : Prop :=
  sorry

/-- Checks if there exists a pair of teams that have each won once against each other -/
def existsMutualWin (t : Tournament n) : Prop :=
  sorry

/-- Main theorem: If all teams have different points after the first round
    and the same total points after both rounds, then there exists a pair
    of teams that have each won once against each other -/
theorem mutual_win_exists (t : Tournament 20)
    (h1 : allDifferentFirstRound t)
    (h2 : allSameTotal t) :
    existsMutualWin t := by
  sorry

end NUMINAMATH_CALUDE_mutual_win_exists_l2164_216485


namespace NUMINAMATH_CALUDE_arrange_teachers_and_students_eq_24_l2164_216410

/-- The number of ways to arrange 2 teachers and 4 students in a row -/
def arrange_teachers_and_students : ℕ :=
  /- Two teachers must be in the middle -/
  let teacher_arrangements : ℕ := 2

  /- One specific student (A) cannot be at either end -/
  let student_A_positions : ℕ := 2

  /- Remaining three students can be arranged in the remaining positions -/
  let other_student_arrangements : ℕ := 6

  /- Total number of arrangements -/
  teacher_arrangements * student_A_positions * other_student_arrangements

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrange_teachers_and_students_eq_24 :
  arrange_teachers_and_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrange_teachers_and_students_eq_24_l2164_216410


namespace NUMINAMATH_CALUDE_A_in_third_quadrant_l2164_216478

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given point A -/
def A : Point :=
  { x := -2, y := -3 }

/-- Theorem stating that point A is in the third quadrant -/
theorem A_in_third_quadrant : isInThirdQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_A_in_third_quadrant_l2164_216478


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2164_216450

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2164_216450


namespace NUMINAMATH_CALUDE_trivia_team_size_l2164_216482

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : absent_members = 2)
  (h2 : points_per_member = 6)
  (h3 : total_points = 18) :
  ∃ (original_size : ℕ), 
    original_size * points_per_member - absent_members * points_per_member = total_points ∧ 
    original_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l2164_216482


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l2164_216458

theorem sugar_recipe_reduction :
  let original_recipe : ℚ := 27/4  -- 6 3/4 cups
  let reduced_recipe : ℚ := (1/3) * original_recipe
  reduced_recipe = 9/4  -- 2 1/4 cups
  := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l2164_216458


namespace NUMINAMATH_CALUDE_min_value_P_l2164_216408

theorem min_value_P (a b : ℝ) (h : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ t : ℝ, a * t^3 - t^2 + b * t - 1 = 0 ↔ t = x ∨ t = y ∨ t = z)) :
  ∀ P : ℝ, P = (5 * a^2 - 3 * a * b + 2) / (a^2 * (b - a)) → P ≥ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_P_l2164_216408


namespace NUMINAMATH_CALUDE_envelope_is_hyperbola_l2164_216489

/-- A family of straight lines forming right-angled triangles with area a^2 / 2 -/
def LineFamily (a : ℝ) := {l : Set (ℝ × ℝ) | ∃ α : ℝ, α > 0 ∧ l = {(x, y) | x + α^2 * y = α * a}}

/-- The envelope of the family of lines -/
def Envelope (a : ℝ) := {(x, y) : ℝ × ℝ | x * y = a^2 / 4}

/-- Theorem stating that the envelope of the line family is the hyperbola xy = a^2 / 4 -/
theorem envelope_is_hyperbola (a : ℝ) (h : a > 0) :
  Envelope a = {p : ℝ × ℝ | ∃ l ∈ LineFamily a, p ∈ l ∧ 
    ∀ l' ∈ LineFamily a, l ≠ l' → (∃ q ∈ l ∩ l', ∀ r ∈ l ∩ l', dist p q ≤ dist p r)} :=
sorry

end NUMINAMATH_CALUDE_envelope_is_hyperbola_l2164_216489


namespace NUMINAMATH_CALUDE_divisible_by_eleven_smallest_n_seven_l2164_216475

theorem divisible_by_eleven_smallest_n_seven (x : ℕ) : 
  (∃ k : ℕ, x = 11 * k) ∧ 
  (∀ m : ℕ, m < 7 → ¬(∃ j : ℕ, m * 11 = x)) ∧
  (∃ i : ℕ, 7 * 11 = x) →
  x = 77 := by sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_smallest_n_seven_l2164_216475


namespace NUMINAMATH_CALUDE_gcf_75_45_l2164_216460

theorem gcf_75_45 : Nat.gcd 75 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_75_45_l2164_216460


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l2164_216452

theorem president_and_committee_selection (n : ℕ) (k : ℕ) : 
  n = 10 → k = 3 → n * (Nat.choose (n - 1) k) = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_selection_l2164_216452


namespace NUMINAMATH_CALUDE_rounds_played_l2164_216423

def total_points : ℕ := 154
def points_per_round : ℕ := 11

theorem rounds_played (total : ℕ) (per_round : ℕ) (h1 : total = total_points) (h2 : per_round = points_per_round) :
  total / per_round = 14 := by
  sorry

end NUMINAMATH_CALUDE_rounds_played_l2164_216423


namespace NUMINAMATH_CALUDE_circle_symmetry_l2164_216492

-- Define the symmetry property
def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

-- Define the equation of circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y = 0

-- Define the equation of circle C'
def circle_C' (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 10

-- Theorem statement
theorem circle_symmetry :
  (∀ x y : ℝ, circle_C x y → circle_C (symmetric_point x y).1 (symmetric_point x y).2) →
  (∀ x y : ℝ, circle_C' x y ↔ circle_C (symmetric_point x y).1 (symmetric_point x y).2) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2164_216492


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_selection_l2164_216454

theorem stratified_sampling_second_year_selection
  (total_students : ℕ)
  (first_year_students : ℕ)
  (second_year_students : ℕ)
  (first_year_selected : ℕ)
  (h1 : total_students = 70)
  (h2 : first_year_students = 30)
  (h3 : second_year_students = 40)
  (h4 : first_year_selected = 6)
  (h5 : total_students = first_year_students + second_year_students) :
  (first_year_selected : ℚ) / first_year_students * second_year_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_selection_l2164_216454


namespace NUMINAMATH_CALUDE_arithmetic_mean_fraction_l2164_216449

theorem arithmetic_mean_fraction (x b : ℝ) (h : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fraction_l2164_216449


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2164_216463

/-- Represents a systematic sampling result -/
structure SystematicSample where
  first : Nat
  interval : Nat
  size : Nat

/-- Generates a sequence of numbers using systematic sampling -/
def generateSequence (sample : SystematicSample) : List Nat :=
  List.range sample.size |>.map (fun i => sample.first + i * sample.interval)

/-- Checks if a sequence is within the given range -/
def isWithinRange (seq : List Nat) (maxVal : Nat) : Prop :=
  seq.all (· ≤ maxVal)

theorem systematic_sampling_result :
  let classSize : Nat := 50
  let sampleSize : Nat := 5
  let result : List Nat := [5, 15, 25, 35, 45]
  ∃ (sample : SystematicSample),
    sample.size = sampleSize ∧
    sample.interval = classSize / sampleSize ∧
    generateSequence sample = result ∧
    isWithinRange result classSize :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2164_216463


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l2164_216495

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 7, -10]

theorem matrix_sum_equality : A + B = !![(-2 : ℤ), 5; 7, -5] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l2164_216495


namespace NUMINAMATH_CALUDE_real_roots_of_equation_l2164_216497

theorem real_roots_of_equation :
  ∀ x : ℝ, x^4 + x^2 - 20 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_equation_l2164_216497


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l2164_216499

/-- Represents a parabola in 2D space -/
structure Parabola where
  equation : ℝ → ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { equation := fun x => p.equation (x - h) + v }

/-- The original parabola y = 3x² -/
def original_parabola : Parabola :=
  { equation := fun x => 3 * x^2 }

/-- The shifted parabola -/
def shifted_parabola : Parabola :=
  shift_parabola original_parabola 1 2

theorem shifted_parabola_equation :
  shifted_parabola.equation = fun x => 3 * (x - 1)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l2164_216499


namespace NUMINAMATH_CALUDE_hexagon_height_is_six_l2164_216481

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a hexagon --/
structure Hexagon where
  height : ℝ

/-- Given a 9x16 rectangle that can be cut into two congruent hexagons
    and repositioned to form a different rectangle, 
    prove that the height of each hexagon is 6 --/
theorem hexagon_height_is_six 
  (original : Rectangle)
  (new : Rectangle)
  (hex1 hex2 : Hexagon)
  (h1 : original.width = 16 ∧ original.height = 9)
  (h2 : hex1 = hex2)
  (h3 : original.width * original.height = new.width * new.height)
  (h4 : new.width = new.height)
  (h5 : hex1.height + hex2.height = new.height)
  : hex1.height = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_height_is_six_l2164_216481


namespace NUMINAMATH_CALUDE_ed_lost_marbles_l2164_216417

theorem ed_lost_marbles (ed_initial : ℕ → ℕ) (doug_initial : ℕ) 
  (h1 : ed_initial doug_initial = doug_initial + 30)
  (h2 : ed_initial doug_initial - 21 = 91)
  (h3 : 91 = doug_initial + 9) : 
  ed_initial doug_initial - 91 = 21 :=
by sorry

end NUMINAMATH_CALUDE_ed_lost_marbles_l2164_216417


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l2164_216443

/-- Given a point P(x,6) on the terminal side of angle θ with cos θ = -4/5, prove that x = -8 -/
theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (x, 6) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) →
  Real.cos θ = -4/5 →
  x = -8 :=
by sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l2164_216443


namespace NUMINAMATH_CALUDE_two_sevens_numeral_l2164_216427

/-- Given two sevens in a numeral with a difference of 69930 between their place values,
    prove that the numeral is 7700070. -/
theorem two_sevens_numeral (A B : ℕ) : 
  A - B = 69930 →
  A = 10 * B →
  A = 77700 ∧ B = 7770 ∧ 7700070 = 7 * A + 7 * B :=
by sorry

end NUMINAMATH_CALUDE_two_sevens_numeral_l2164_216427


namespace NUMINAMATH_CALUDE_projection_problem_l2164_216402

/-- Given two vectors that project onto the same vector, prove the resulting projection vector --/
theorem projection_problem (a b v p : ℝ × ℝ) : 
  a = (-3, 2) →
  b = (4, 5) →
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ p = k₂ • v) →
  p = (-69/58, 161/58) := by
  sorry

end NUMINAMATH_CALUDE_projection_problem_l2164_216402


namespace NUMINAMATH_CALUDE_find_r_l2164_216444

theorem find_r (a b c r : ℝ) 
  (h1 : a * (b - c) / (b * (c - a)) = r)
  (h2 : b * (c - a) / (c * (b - a)) = r)
  (h3 : r > 0) :
  r = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l2164_216444


namespace NUMINAMATH_CALUDE_fraction_simplification_l2164_216493

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) :
  b / (a * b + b) = 1 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2164_216493


namespace NUMINAMATH_CALUDE_triangle_circumradius_l2164_216467

/-- The circumradius of a triangle with sides 12, 10, and 7 is 6 units. -/
theorem triangle_circumradius (a b c : ℝ) (h_a : a = 12) (h_b : b = 10) (h_c : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * area)
  R = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l2164_216467


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_1645_l2164_216496

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_9999_seconds_to_1645 :
  let initial_time : Time := ⟨16, 45, 0⟩
  let seconds_to_add : Nat := 9999
  let final_time : Time := addSeconds initial_time seconds_to_add
  final_time = ⟨19, 31, 39⟩ := by sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_1645_l2164_216496


namespace NUMINAMATH_CALUDE_church_members_count_l2164_216473

theorem church_members_count :
  ∀ (total adults children : ℕ),
  adults = (40 * total) / 100 →
  children = total - adults →
  children = adults + 24 →
  total = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_church_members_count_l2164_216473


namespace NUMINAMATH_CALUDE_pipe_filling_time_l2164_216457

/-- Given two pipes A and B that fill a tank, where:
    - Pipe A fills the tank in t minutes
    - Pipe B fills the tank 3 times as fast as Pipe A
    - Both pipes together fill the tank in 3 minutes
    Then, Pipe A takes 12 minutes to fill the tank alone. -/
theorem pipe_filling_time (t : ℝ) 
  (hA : t > 0)  -- Pipe A's filling time is positive
  (hB : t / 3 > 0)  -- Pipe B's filling time is positive
  (h_both : 1 / t + 1 / (t / 3) = 1 / 3)  -- Combined filling rate equals 1/3
  : t = 12 := by
  sorry


end NUMINAMATH_CALUDE_pipe_filling_time_l2164_216457


namespace NUMINAMATH_CALUDE_students_taking_both_languages_l2164_216442

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ) :
  total = 94 →
  french = 41 →
  german = 22 →
  neither = 40 →
  ∃ (both : ℕ), both = 9 ∧ total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_taking_both_languages_l2164_216442


namespace NUMINAMATH_CALUDE_park_trees_l2164_216437

theorem park_trees (blackbirds_per_tree : ℕ) (magpies : ℕ) (total_birds : ℕ) :
  blackbirds_per_tree = 3 →
  magpies = 13 →
  total_birds = 34 →
  ∃ trees : ℕ, trees * blackbirds_per_tree + magpies = total_birds ∧ trees = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_park_trees_l2164_216437


namespace NUMINAMATH_CALUDE_intersection_count_l2164_216421

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 15

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := (num_x_points.choose 2) * (num_y_points.choose 2)

/-- Theorem stating the maximum number of intersection points -/
theorem intersection_count :
  max_intersections = 4725 := by sorry

end NUMINAMATH_CALUDE_intersection_count_l2164_216421


namespace NUMINAMATH_CALUDE_revenue_condition_l2164_216440

def initial_price : ℝ := 50
def initial_sales : ℝ := 300
def revenue_threshold : ℝ := 15950

def monthly_revenue (x : ℝ) : ℝ := (initial_price - x) * (initial_sales + 10 * x)

theorem revenue_condition (x : ℝ) :
  monthly_revenue x ≥ revenue_threshold ↔ (x = 9 ∨ x = 11) :=
sorry

end NUMINAMATH_CALUDE_revenue_condition_l2164_216440


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l2164_216472

theorem mean_equality_implies_y_equals_three :
  let mean1 := (7 + 11 + 19) / 3
  let mean2 := (16 + 18 + y) / 3
  mean1 = mean2 →
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l2164_216472


namespace NUMINAMATH_CALUDE_well_digging_payment_l2164_216403

/-- The total payment for two workers digging a well over three days -/
def total_payment (hourly_rate : ℕ) (day1_hours day2_hours day3_hours : ℕ) (num_workers : ℕ) : ℕ :=
  hourly_rate * (day1_hours + day2_hours + day3_hours) * num_workers

/-- Theorem stating that the total payment for the given scenario is $660 -/
theorem well_digging_payment :
  total_payment 10 10 8 15 2 = 660 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l2164_216403


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2164_216456

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 145 →
  bridge_length = 230 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2164_216456


namespace NUMINAMATH_CALUDE_f_zero_eq_five_l2164_216447

/-- Given a function f such that f(x-2) = 2^x - x + 3 for all x, prove that f(0) = 5 -/
theorem f_zero_eq_five (f : ℝ → ℝ) (h : ∀ x, f (x - 2) = 2^x - x + 3) : f 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_eq_five_l2164_216447


namespace NUMINAMATH_CALUDE_slope_of_cutting_line_l2164_216483

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℚ

/-- Checks if a line cuts a parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { v1 := ⟨4, 20⟩
  , v2 := ⟨4, 56⟩
  , v3 := ⟨13, 81⟩
  , v4 := ⟨13, 45⟩ }

/-- The theorem to be proved -/
theorem slope_of_cutting_line :
  ∃ (l : Line), cutsIntoCongruentPolygons specificParallelogram l ∧ l.slope = 53 / 9 :=
sorry

end NUMINAMATH_CALUDE_slope_of_cutting_line_l2164_216483


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2164_216420

/-- 
Given a right circular cone with diameter 16 units and altitude 20 units, 
and an inscribed right circular cylinder with height equal to its diameter 
and coinciding axis with the cone, the radius of the cylinder is 40/9 units.
-/
theorem inscribed_cylinder_radius 
  (cone_diameter : ℝ) 
  (cone_altitude : ℝ) 
  (cylinder_radius : ℝ) :
  cone_diameter = 16 →
  cone_altitude = 20 →
  cylinder_radius * 2 = cylinder_radius * 2 →  -- Height equals diameter
  (cone_altitude - cylinder_radius * 2) / cylinder_radius = 5 / 2 →  -- Similar triangles ratio
  cylinder_radius = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2164_216420


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l2164_216409

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : z₁ + z₂ = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l2164_216409


namespace NUMINAMATH_CALUDE_factor_expression_l2164_216494

theorem factor_expression (x : ℝ) : 75 * x^11 + 200 * x^22 = 25 * x^11 * (3 + 8 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2164_216494


namespace NUMINAMATH_CALUDE_product_of_terms_l2164_216432

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), ∀ n, b n = b₁ * r^(n - 1)

/-- Main theorem -/
theorem product_of_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n ≠ 0) →
  2 * (a 2) - (a 7)^2 + 2 * (a 12) = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 3 * b 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_terms_l2164_216432


namespace NUMINAMATH_CALUDE_circle_C_and_line_l_properties_l2164_216404

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the line m: y = -2x + 4
def line_m (x : ℝ) : ℝ := -2 * x + 4

-- Define circle P
def circle_P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 5 * p.1^2 + 5 * p.2^2 - 16 * p.1 - 8 * p.2 + 12 = 0}

theorem circle_C_and_line_l_properties :
  ∃ (center : ℝ × ℝ) (P Q : ℝ × ℝ) (k : ℝ),
    center.2 = line_y_eq_x center.1 ∧
    point_A ∈ circle_C ∧
    point_B ∈ circle_C ∧
    P ∈ circle_C ∧
    Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧
    Q.2 = line_l k Q.1 ∧
    dot_product P Q = -2 →
    (∀ (p : ℝ × ℝ), p ∈ circle_C ↔ p.1^2 + p.2^2 = 4) ∧
    k = 0 ∧
    ∃ (E F : ℝ × ℝ),
      E ∈ circle_C ∧
      F ∈ circle_C ∧
      E.2 = line_m E.1 ∧
      F.2 = line_m F.1 ∧
      (2, 0) ∈ circle_P :=
by sorry

end NUMINAMATH_CALUDE_circle_C_and_line_l_properties_l2164_216404


namespace NUMINAMATH_CALUDE_armstrong_made_quote_l2164_216428

-- Define the type for astronauts
inductive Astronaut : Type
| Apollo : Astronaut
| MichaelCollins : Astronaut
| Armstrong : Astronaut
| Aldrin : Astronaut

-- Define the famous quote
def famous_quote : String := "That's one small step for man, one giant leap for mankind."

-- Define the property of making the quote on the Moon
def made_quote_on_moon (a : Astronaut) : Prop := 
  a = Astronaut.Armstrong ∧ ∃ (quote : String), quote = famous_quote

-- Theorem stating that Armstrong made the famous quote on the Moon
theorem armstrong_made_quote : 
  ∃ (a : Astronaut), made_quote_on_moon a :=
sorry

end NUMINAMATH_CALUDE_armstrong_made_quote_l2164_216428


namespace NUMINAMATH_CALUDE_gcd_689_1021_l2164_216425

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_689_1021_l2164_216425


namespace NUMINAMATH_CALUDE_no_130_consecutive_numbers_with_900_divisors_l2164_216431

theorem no_130_consecutive_numbers_with_900_divisors :
  ¬ ∃ (n : ℕ), ∀ (k : ℕ), k ∈ Finset.range 130 →
    (Nat.divisors (n + k)).card = 900 :=
sorry

end NUMINAMATH_CALUDE_no_130_consecutive_numbers_with_900_divisors_l2164_216431


namespace NUMINAMATH_CALUDE_polar_equation_perpendicular_line_l2164_216490

/-- The polar equation of a line passing through (2,0) and perpendicular to the polar axis -/
theorem polar_equation_perpendicular_line (ρ θ : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ (x y : ℝ), x = 2 → y = ρ * Real.sin θ) →
  ρ * Real.cos θ = 2 :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_perpendicular_line_l2164_216490


namespace NUMINAMATH_CALUDE_store_purchase_exists_l2164_216488

theorem store_purchase_exists :
  ∃ (P L E : ℕ), 0.45 * (P : ℝ) + 0.35 * (L : ℝ) + 0.30 * (E : ℝ) = 7.80 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_exists_l2164_216488


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2164_216471

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + m = 0 ∧ y^2 - 6*y + m = 0) → m < 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2164_216471


namespace NUMINAMATH_CALUDE_average_k_for_quadratic_roots_l2164_216448

theorem average_k_for_quadratic_roots (k : ℤ) : 
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ (a, b) => a + b)
  let distinct_k_values := k_values.eraseDups
  (distinct_k_values.sum / distinct_k_values.length : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_k_for_quadratic_roots_l2164_216448


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2164_216433

theorem arithmetic_expression_equality : 70 + (105 / 15) + (19 * 11) - 250 - (360 / 12) = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2164_216433


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2164_216474

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := m^2 - 1 + (m + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) →
  (m = 1 ∧ (1 : ℂ) / (1 + z) = (1 : ℂ) / 5 - (2 : ℂ) / 5 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2164_216474


namespace NUMINAMATH_CALUDE_problem_statement_l2164_216477

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem problem_statement :
  (∀ a, a ∈ M → a ∈ N) ∧ 
  (∃ a, a ∈ M ∧ a ∉ N) ∧
  (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
  (¬(∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2164_216477


namespace NUMINAMATH_CALUDE_f_3_equals_7_l2164_216468

-- Define the function f
def f : ℝ → ℝ := fun x => 2*x + 1

-- State the theorem
theorem f_3_equals_7 : f 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_7_l2164_216468


namespace NUMINAMATH_CALUDE_hexagonangulo_19_requires_59_l2164_216413

/-- A hexagonângulo is a shape formed by triangles -/
structure Hexagonangulo where
  triangles : ℕ
  perimeter : ℕ

/-- Calculates the number of unit triangles needed to form a triangle of given side length -/
def trianglesInLargerTriangle (side : ℕ) : ℕ := side^2

/-- Constructs a hexagonângulo with given perimeter using unit triangles -/
def constructHexagonangulo (p : ℕ) : Hexagonangulo :=
  { triangles := 
      4 * trianglesInLargerTriangle 2 + 
      3 * trianglesInLargerTriangle 3 + 
      1 * trianglesInLargerTriangle 4,
    perimeter := p }

/-- Theorem: A hexagonângulo with perimeter 19 requires 59 unit triangles -/
theorem hexagonangulo_19_requires_59 : 
  (constructHexagonangulo 19).triangles = 59 := by sorry

end NUMINAMATH_CALUDE_hexagonangulo_19_requires_59_l2164_216413


namespace NUMINAMATH_CALUDE_exam_average_l2164_216461

theorem exam_average (total_candidates : ℕ) (passed_candidates : ℕ) (passed_avg : ℝ) (failed_avg : ℝ) :
  total_candidates = 120 →
  passed_candidates = 100 →
  passed_avg = 39 →
  failed_avg = 15 →
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := passed_candidates * passed_avg + failed_candidates * failed_avg
  let overall_avg := total_marks / total_candidates
  overall_avg = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l2164_216461


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l2164_216407

theorem arithmetic_square_root_of_one_fourth :
  let x : ℚ := 1/2
  (x * x = 1/4) ∧ (∀ y : ℚ, y * y = 1/4 → y = x ∨ y = -x) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l2164_216407


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2164_216429

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2164_216429


namespace NUMINAMATH_CALUDE_exponent_division_l2164_216435

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^4 / x = x^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2164_216435


namespace NUMINAMATH_CALUDE_distance_sum_constant_l2164_216412

theorem distance_sum_constant (a b x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) :
  |x - a| + |x - b| = 50 :=
by
  sorry

#check distance_sum_constant

end NUMINAMATH_CALUDE_distance_sum_constant_l2164_216412


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2164_216438

theorem arithmetic_computation : -9 * 5 - (-7 * -2) + (-11 * -6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2164_216438


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2164_216414

theorem polynomial_expansion (x : ℝ) : 
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2164_216414


namespace NUMINAMATH_CALUDE_smallest_unstuck_perimeter_l2164_216405

/-- A rectangle inscribed in a larger rectangle. -/
structure InscribedRectangle where
  outer_width : ℝ
  outer_height : ℝ
  inner_width : ℝ
  inner_height : ℝ
  is_inscribed : inner_width ≤ outer_width ∧ inner_height ≤ outer_height

/-- An unstuck inscribed rectangle can be rotated slightly within the larger rectangle. -/
def is_unstuck (r : InscribedRectangle) : Prop := sorry

/-- The perimeter of a rectangle. -/
def perimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem to be proved. -/
theorem smallest_unstuck_perimeter :
  ∃ (r : InscribedRectangle),
    r.outer_width = 8 ∧
    r.outer_height = 6 ∧
    is_unstuck r ∧
    (∀ (s : InscribedRectangle),
      s.outer_width = 8 ∧
      s.outer_height = 6 ∧
      is_unstuck s →
      perimeter r.inner_width r.inner_height ≤ perimeter s.inner_width s.inner_height) ∧
    perimeter r.inner_width r.inner_height = Real.sqrt 448 := by sorry

end NUMINAMATH_CALUDE_smallest_unstuck_perimeter_l2164_216405


namespace NUMINAMATH_CALUDE_annette_sara_weight_difference_l2164_216419

/-- Given the weights of combinations of people, prove that Annette weighs 8 pounds more than Sara. -/
theorem annette_sara_weight_difference 
  (annette caitlin sara bob : ℝ) 
  (h1 : annette + caitlin = 95)
  (h2 : caitlin + sara = 87)
  (h3 : annette + sara = 97)
  (h4 : caitlin + bob = 100)
  (h5 : annette + caitlin + bob = 155) :
  annette - sara = 8 := by
sorry

end NUMINAMATH_CALUDE_annette_sara_weight_difference_l2164_216419


namespace NUMINAMATH_CALUDE_i_cubed_eq_neg_i_l2164_216424

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- State the theorem
theorem i_cubed_eq_neg_i : i^3 = -i := by sorry

end NUMINAMATH_CALUDE_i_cubed_eq_neg_i_l2164_216424


namespace NUMINAMATH_CALUDE_cos_difference_special_l2164_216401

theorem cos_difference_special (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_special_l2164_216401


namespace NUMINAMATH_CALUDE_unique_solution_xy_l2164_216411

theorem unique_solution_xy : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 3) + (y - 3)) ∧ 
  x = 1 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l2164_216411


namespace NUMINAMATH_CALUDE_equation_solution_l2164_216466

theorem equation_solution : ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ↔ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2164_216466
