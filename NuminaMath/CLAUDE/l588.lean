import Mathlib

namespace NUMINAMATH_CALUDE_iodine131_electrons_l588_58845

structure Atom where
  atomicMass : ℕ
  protonNumber : ℕ

def numberOfNeutrons (a : Atom) : ℕ := a.atomicMass - a.protonNumber

def numberOfElectrons (a : Atom) : ℕ := a.protonNumber

def iodine131 : Atom := ⟨131, 53⟩

theorem iodine131_electrons : numberOfElectrons iodine131 = 53 := by
  sorry

end NUMINAMATH_CALUDE_iodine131_electrons_l588_58845


namespace NUMINAMATH_CALUDE_cos_105_cos_45_plus_sin_105_sin_45_l588_58828

theorem cos_105_cos_45_plus_sin_105_sin_45 :
  Real.cos (105 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (105 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_cos_45_plus_sin_105_sin_45_l588_58828


namespace NUMINAMATH_CALUDE_order_of_abc_l588_58885

theorem order_of_abc : 
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l588_58885


namespace NUMINAMATH_CALUDE_flour_needed_l588_58873

theorem flour_needed (total : ℝ) (added : ℝ) (needed : ℝ) :
  total = 8.5 ∧ added = 2.25 ∧ needed = total - added → needed = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l588_58873


namespace NUMINAMATH_CALUDE_vector_on_line_l588_58842

/-- Given distinct vectors a and b in a vector space V over ℝ,
    prove that the vector (1/2)a + (1/2)b lies on the line passing through a and b. -/
theorem vector_on_line {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/2 : ℝ) • a + (1/2 : ℝ) • b = a + t • (b - a) :=
sorry

end NUMINAMATH_CALUDE_vector_on_line_l588_58842


namespace NUMINAMATH_CALUDE_min_value_of_f_l588_58835

/-- The function f(x) = -x^3 + 3x^2 + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- Theorem: Given f(x) = -x^3 + 3x^2 + 9x + a, where a is a constant,
    and the maximum value of f(x) in the interval [-2, 2] is 20,
    the minimum value of f(x) in the interval [-2, 2] is -7. -/
theorem min_value_of_f (a : ℝ) (h : ∃ x ∈ Set.Icc (-2) 2, f a x = 20 ∧ ∀ y ∈ Set.Icc (-2) 2, f a y ≤ 20) :
  ∃ x ∈ Set.Icc (-2) 2, f a x = -7 ∧ ∀ y ∈ Set.Icc (-2) 2, f a y ≥ -7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l588_58835


namespace NUMINAMATH_CALUDE_ratio_evaluation_l588_58847

theorem ratio_evaluation : (2^121 * 3^123) / 6^122 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l588_58847


namespace NUMINAMATH_CALUDE_flashlight_problem_l588_58819

/-- Represents the minimum number of attempts needed to guarantee a flashlight lights up -/
def min_attempts (total_batteries : ℕ) (good_batteries : ℕ) : ℕ :=
  if total_batteries = 2 * good_batteries - 1 
  then good_batteries + 1
  else if total_batteries = 2 * good_batteries 
  then good_batteries + 3
  else 0  -- undefined for other cases

/-- Theorem for the flashlight problem -/
theorem flashlight_problem (n : ℕ) (h : n > 2) :
  (min_attempts (2 * n + 1) (n + 1) = n + 2) ∧
  (min_attempts (2 * n) n = n + 3) := by
  sorry

#check flashlight_problem

end NUMINAMATH_CALUDE_flashlight_problem_l588_58819


namespace NUMINAMATH_CALUDE_square_root_of_81_l588_58898

theorem square_root_of_81 : 
  {x : ℝ | x^2 = 81} = {9, -9} := by sorry

end NUMINAMATH_CALUDE_square_root_of_81_l588_58898


namespace NUMINAMATH_CALUDE_triangle_mn_length_l588_58849

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let BC := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  AB = 5 ∧ AC = 4 ∧ BC = 6

-- Define the angle bisector and point X
def angleBisector (t : Triangle) : ℝ × ℝ → Prop := sorry

-- Define points M and N
def pointM (t : Triangle) : ℝ × ℝ := sorry
def pointN (t : Triangle) : ℝ × ℝ := sorry

-- Define parallel lines
def isParallel (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem triangle_mn_length (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : ∃ X, angleBisector t X ∧ X.1 ∈ Set.Icc t.A.1 t.B.1 ∧ X.2 = t.A.2)
  (h3 : isParallel (X, pointM t) (t.A, t.C))
  (h4 : isParallel (X, pointN t) (t.B, t.C)) :
  let MN := Real.sqrt ((pointM t).1 - (pointN t).1)^2 + ((pointM t).2 - (pointN t).2)^2
  MN = 3 * Real.sqrt 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_mn_length_l588_58849


namespace NUMINAMATH_CALUDE_smallest_consecutive_non_primes_l588_58860

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_non_primes (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 5 → ¬(is_prime (start + i))

theorem smallest_consecutive_non_primes :
  ∃ (n : ℕ), n > 90 ∧ n < 96 ∧ consecutive_non_primes n ∧
  ∀ m : ℕ, m > 90 ∧ m < 96 ∧ consecutive_non_primes m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_non_primes_l588_58860


namespace NUMINAMATH_CALUDE_book_pages_count_l588_58874

theorem book_pages_count (total_notebooks : ℕ) (sum_of_four_pages : ℕ) : 
  total_notebooks = 12 ∧ sum_of_four_pages = 338 → 
  ∃ (total_pages : ℕ), 
    total_pages = 288 ∧
    (total_pages / total_notebooks : ℚ) + 1 + 
    (total_pages / total_notebooks : ℚ) + 2 + 
    (total_pages / 3 : ℚ) - 1 + 
    (total_pages / 3 : ℚ) = sum_of_four_pages := by
  sorry

#check book_pages_count

end NUMINAMATH_CALUDE_book_pages_count_l588_58874


namespace NUMINAMATH_CALUDE_problem_solution_l588_58886

theorem problem_solution (x y : ℝ) 
  (h1 : x = 52) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y + 100 = 540000) : 
  y = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l588_58886


namespace NUMINAMATH_CALUDE_final_savings_after_expense_increase_l588_58843

/-- Calculates the final savings after expense increase -/
def finalSavings (salary : ℝ) (initialSavingsRate : ℝ) (expenseIncreaseRate : ℝ) : ℝ :=
  let initialExpenses := salary * (1 - initialSavingsRate)
  let newExpenses := initialExpenses * (1 + expenseIncreaseRate)
  salary - newExpenses

/-- Theorem stating that given the problem conditions, the final savings is 250 -/
theorem final_savings_after_expense_increase :
  finalSavings 6250 0.2 0.2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_final_savings_after_expense_increase_l588_58843


namespace NUMINAMATH_CALUDE_robin_extra_drinks_l588_58801

/-- Calculates the number of extra drinks given the quantities bought and consumed --/
def extra_drinks (sodas_bought energy_bought smoothies_bought
                  sodas_drunk energy_drunk smoothies_drunk : ℕ) : ℕ :=
  (sodas_bought + energy_bought + smoothies_bought) -
  (sodas_drunk + energy_drunk + smoothies_drunk)

/-- Theorem stating that Robin has 32 extra drinks --/
theorem robin_extra_drinks :
  extra_drinks 22 15 12 6 9 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_robin_extra_drinks_l588_58801


namespace NUMINAMATH_CALUDE_geometric_sequence_values_l588_58836

theorem geometric_sequence_values (a b c : ℝ) : 
  (∃ q : ℝ, q ≠ 0 ∧ 2 * q = a ∧ a * q = b ∧ b * q = c ∧ c * q = 32) → 
  ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_values_l588_58836


namespace NUMINAMATH_CALUDE_minimum_score_for_average_increase_miguel_minimum_score_l588_58817

def current_scores : List ℕ := [92, 88, 76, 84, 90]
def desired_increase : ℕ := 4

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem minimum_score_for_average_increase 
  (scores : List ℕ) 
  (increase : ℕ) 
  (min_score : ℕ) : Prop :=
  let current_avg := average scores
  let new_scores := scores ++ [min_score]
  let new_avg := average new_scores
  new_avg ≥ current_avg + increase ∧
  ∀ (score : ℕ), score < min_score → 
    average (scores ++ [score]) < current_avg + increase

theorem miguel_minimum_score : 
  minimum_score_for_average_increase current_scores desired_increase 110 := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_for_average_increase_miguel_minimum_score_l588_58817


namespace NUMINAMATH_CALUDE_rounded_number_accuracy_l588_58853

/-- Given a number 5.60 × 10^5 rounded to the nearest whole number,
    prove that it is accurate to the thousandth place. -/
theorem rounded_number_accuracy (n : ℝ) (h : n = 5.60 * 10^5) :
  ∃ (m : ℕ), |n - m| ≤ 5 * 10^2 :=
sorry

end NUMINAMATH_CALUDE_rounded_number_accuracy_l588_58853


namespace NUMINAMATH_CALUDE_brian_cards_left_l588_58897

/-- Given that Brian has 76 cards initially and Wayne takes 59 cards away,
    prove that Brian will have 17 cards left. -/
theorem brian_cards_left (initial_cards : ℕ) (cards_taken : ℕ) (cards_left : ℕ) : 
  initial_cards = 76 → cards_taken = 59 → cards_left = initial_cards - cards_taken → cards_left = 17 := by
  sorry

end NUMINAMATH_CALUDE_brian_cards_left_l588_58897


namespace NUMINAMATH_CALUDE_eugene_model_house_l588_58876

/-- Eugene's model house building problem --/
theorem eugene_model_house (toothpicks_per_card : ℕ) (cards_in_deck : ℕ) 
  (boxes_used : ℕ) (toothpicks_per_box : ℕ) : 
  toothpicks_per_card = 75 →
  cards_in_deck = 52 →
  boxes_used = 6 →
  toothpicks_per_box = 450 →
  cards_in_deck - (boxes_used * toothpicks_per_box) / toothpicks_per_card = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_eugene_model_house_l588_58876


namespace NUMINAMATH_CALUDE_basketball_team_starters_l588_58839

theorem basketball_team_starters : Nat.choose 16 8 = 12870 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l588_58839


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l588_58899

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l588_58899


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l588_58820

theorem quadratic_roots_to_coefficients :
  ∀ (p q : ℝ),
    (∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = -2 ∨ x = 3) →
    p = -1 ∧ q = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l588_58820


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_4_75_l588_58834

/-- Represents the dimensions of a rectangular yard -/
structure YardDimensions where
  length : ℝ
  width : ℝ

/-- Represents the parallel sides of the rectangular remainder -/
structure RemainderSides where
  side1 : ℝ
  side2 : ℝ

/-- Calculates the fraction of the yard occupied by flower beds -/
def flowerBedFraction (yard : YardDimensions) (remainder : RemainderSides) : ℚ :=
  sorry

/-- Theorem statement -/
theorem flower_bed_fraction_is_4_75 
  (yard : YardDimensions)
  (remainder : RemainderSides)
  (h1 : yard.length = 30)
  (h2 : yard.width = 10)
  (h3 : remainder.side1 = 30)
  (h4 : remainder.side2 = 22) :
  flowerBedFraction yard remainder = 4/75 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_4_75_l588_58834


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l588_58825

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 
    (x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0) ∧
    (x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0) ∧
    (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l588_58825


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l588_58837

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (4 / (x - 1) = 3 / x) ↔ x = -3 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l588_58837


namespace NUMINAMATH_CALUDE_broken_tree_height_l588_58807

/-- 
Given a tree that broke and fell across a road, this theorem proves that 
if the breadth of the road is 12 m and the tree broke at a height of 16 m, 
then the original height of the tree is 36 m.
-/
theorem broken_tree_height 
  (breadth : ℝ) 
  (broken_height : ℝ) 
  (h_breadth : breadth = 12) 
  (h_broken : broken_height = 16) : 
  ∃ (original_height : ℝ), original_height = 36 := by
  sorry

end NUMINAMATH_CALUDE_broken_tree_height_l588_58807


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l588_58812

/-- Given that real numbers 4, m, 9 form a geometric sequence,
    prove that the eccentricity of the conic section x^2/m + y^2 = 1
    is either √30/6 or √7 -/
theorem conic_section_eccentricity (m : ℝ) :
  (4 * m = m * 9) →
  let e := if m > 0
           then Real.sqrt (1 - m / 6) / Real.sqrt (m / 6)
           else Real.sqrt (1 + 6 / m) / 1
  (e = Real.sqrt 30 / 6 ∨ e = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l588_58812


namespace NUMINAMATH_CALUDE_miser_knight_theorem_l588_58865

theorem miser_knight_theorem (N : ℕ) (h2 : ∀ (a b : ℕ), a + b = 2 → N % a = 0 ∧ N % b = 0)
  (h3 : ∀ (a b c : ℕ), a + b + c = 3 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0)
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = 4 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0 ∧ N % d = 0)
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = 5 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0 ∧ N % d = 0 ∧ N % e = 0) :
  N % 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_miser_knight_theorem_l588_58865


namespace NUMINAMATH_CALUDE_distance_to_line_rational_l588_58870

/-- The distance from any lattice point to the line 3x - 4y + 4 = 0 is rational -/
theorem distance_to_line_rational (a b : ℤ) : ∃ (q : ℚ), q = |4 * b - 3 * a - 4| / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_rational_l588_58870


namespace NUMINAMATH_CALUDE_overlapping_rectangles_perimeter_l588_58854

/-- The perimeter of a shape formed by two overlapping rectangles -/
theorem overlapping_rectangles_perimeter :
  ∀ (length width : ℝ),
  length = 7 →
  width = 3 →
  (2 * (length + width)) * 2 - 2 * width = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_overlapping_rectangles_perimeter_l588_58854


namespace NUMINAMATH_CALUDE_candy_cost_problem_l588_58871

/-- The cost per pound of the first type of candy -/
def first_candy_cost : ℝ := sorry

/-- The weight of the first type of candy in pounds -/
def first_candy_weight : ℝ := 10

/-- The weight of the second type of candy in pounds -/
def second_candy_weight : ℝ := 20

/-- The cost per pound of the second type of candy -/
def second_candy_cost : ℝ := 5

/-- The cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 30

theorem candy_cost_problem :
  first_candy_cost * first_candy_weight + 
  second_candy_cost * second_candy_weight = 
  mixture_cost * total_weight ∧
  first_candy_cost = 8 := by sorry

end NUMINAMATH_CALUDE_candy_cost_problem_l588_58871


namespace NUMINAMATH_CALUDE_nancy_quarters_l588_58800

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The total amount Nancy saved in dollars -/
def total_saved : ℚ := 3

/-- The number of quarters Nancy saved -/
def num_quarters : ℕ := 12

theorem nancy_quarters :
  (quarter_value * num_quarters : ℚ) = total_saved := by sorry

end NUMINAMATH_CALUDE_nancy_quarters_l588_58800


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l588_58823

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -300 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l588_58823


namespace NUMINAMATH_CALUDE_special_function_upper_bound_l588_58894

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

/-- The main theorem -/
theorem special_function_upper_bound 
  (f : ℝ → ℝ) (h : SpecialFunction f) : 
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by sorry

end NUMINAMATH_CALUDE_special_function_upper_bound_l588_58894


namespace NUMINAMATH_CALUDE_line_intersection_canonical_equations_l588_58813

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations 
  (x y z : ℝ) : 
  (6*x - 5*y + 3*z + 8 = 0) ∧ (6*x + 5*y - 4*z + 4 = 0) →
  ∃ (t : ℝ), x = 5*t - 1 ∧ y = 42*t + 2/5 ∧ z = 60*t :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_equations_l588_58813


namespace NUMINAMATH_CALUDE_first_floor_rooms_l588_58883

theorem first_floor_rooms : ∃ (x : ℕ), x > 0 ∧ 
  (6 * (x - 1) = 5 * x + 4) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_floor_rooms_l588_58883


namespace NUMINAMATH_CALUDE_inequality_proof_l588_58833

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l588_58833


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l588_58827

theorem min_value_sum_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a ≥ 12 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a = 12 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l588_58827


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l588_58844

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 4*ρ*(Real.sin θ) + 7 = 0

/-- Line C₂ in polar coordinates -/
def C₂ (θ : ℝ) : Prop :=
  Real.tan θ = Real.sqrt 3

/-- Theorem stating the sum of reciprocals of distances to intersection points -/
theorem intersection_distance_sum :
  ∀ ρ₁ ρ₂ θ : ℝ,
  C₁ ρ₁ θ → C₁ ρ₂ θ → C₂ θ →
  1 / ρ₁ + 1 / ρ₂ = (2 + 2 * Real.sqrt 3) / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l588_58844


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l588_58875

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x^2 * x^(1/2))^(1/4) = x^(5/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l588_58875


namespace NUMINAMATH_CALUDE_g_zero_iff_a_eq_seven_fifths_l588_58891

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(a) = 0 if and only if a = 7/5 -/
theorem g_zero_iff_a_eq_seven_fifths :
  ∀ a : ℝ, g a = 0 ↔ a = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_iff_a_eq_seven_fifths_l588_58891


namespace NUMINAMATH_CALUDE_ellipse_trajectory_l588_58824

-- Define the focal points
def F1 : ℝ × ℝ := (3, 0)
def F2 : ℝ × ℝ := (-3, 0)

-- Define the distance sum constant
def distanceSum : ℝ := 10

-- Define the equation of the ellipse
def isOnEllipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 25) + (P.2^2 / 16) = 1

-- Theorem statement
theorem ellipse_trajectory :
  ∀ P : ℝ × ℝ, 
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = distanceSum →
  isOnEllipse P :=
sorry

end NUMINAMATH_CALUDE_ellipse_trajectory_l588_58824


namespace NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l588_58881

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l588_58881


namespace NUMINAMATH_CALUDE_range_of_c_l588_58863

-- Define the triangular pyramid
structure TriangularPyramid where
  -- Base edges
  base_edge1 : ℝ
  base_edge2 : ℝ
  base_edge3 : ℝ
  -- Side edges opposite to base edges
  side_edge1 : ℝ
  side_edge2 : ℝ
  side_edge3 : ℝ

-- Define the specific triangular pyramid from the problem
def specificPyramid (c : ℝ) : TriangularPyramid :=
  { base_edge1 := 1
  , base_edge2 := 1
  , base_edge3 := c
  , side_edge1 := 1
  , side_edge2 := c
  , side_edge3 := c }

-- Theorem stating the range of c
theorem range_of_c :
  ∀ c : ℝ, (∃ p : TriangularPyramid, p = specificPyramid c) →
  (Real.sqrt 5 - 1) / 2 < c ∧ c < (Real.sqrt 5 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_c_l588_58863


namespace NUMINAMATH_CALUDE_initial_order_size_l588_58810

/-- The number of cogs produced per hour in the initial phase -/
def initial_rate : ℕ := 36

/-- The number of cogs produced per hour in the second phase -/
def second_rate : ℕ := 60

/-- The number of additional cogs produced in the second phase -/
def additional_cogs : ℕ := 60

/-- The overall average output in cogs per hour -/
def average_output : ℝ := 45

/-- The theorem stating that the initial order was for 60 cogs -/
theorem initial_order_size :
  ∃ x : ℕ, 
    (x + additional_cogs) / (x / initial_rate + 1 : ℝ) = average_output →
    x = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_order_size_l588_58810


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l588_58892

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : 1 / (a 2 * a 4) + 2 / (a 4 * a 4) + 1 / (a 4 * a 6) = 81) :
  1 / a 3 + 1 / a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l588_58892


namespace NUMINAMATH_CALUDE_triangle_side_range_l588_58803

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- acute triangle
  A + B + C = π ∧ -- sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- positive sides
  (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C ∧ -- given equation
  a = Real.sqrt 3 -- given value of a
  →
  5 < b^2 + c^2 ∧ b^2 + c^2 ≤ 6 := by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l588_58803


namespace NUMINAMATH_CALUDE_water_displaced_by_sphere_l588_58889

/-- The volume of water displaced by a completely submerged sphere -/
theorem water_displaced_by_sphere (diameter : ℝ) (volume_displaced : ℝ) :
  diameter = 8 →
  volume_displaced = (4/3) * Real.pi * (diameter/2)^3 →
  volume_displaced = (256/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_by_sphere_l588_58889


namespace NUMINAMATH_CALUDE_trees_survived_vs_died_l588_58811

theorem trees_survived_vs_died (initial_trees dead_trees : ℕ) : 
  initial_trees = 11 → dead_trees = 2 → 
  (initial_trees - dead_trees) - dead_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_trees_survived_vs_died_l588_58811


namespace NUMINAMATH_CALUDE_bike_ride_distance_l588_58877

/-- Calculates the total distance traveled in a 3-hour bike ride given specific conditions -/
theorem bike_ride_distance (second_hour_distance : ℝ) 
  (h1 : second_hour_distance = 12)
  (h2 : second_hour_distance = 1.2 * (second_hour_distance / 1.2))
  (h3 : 1.25 * second_hour_distance = 15) : 
  (second_hour_distance / 1.2) + second_hour_distance + (1.25 * second_hour_distance) = 37 := by
  sorry

#check bike_ride_distance

end NUMINAMATH_CALUDE_bike_ride_distance_l588_58877


namespace NUMINAMATH_CALUDE_locus_of_centers_is_hyperbola_l588_58851

/-- A circle with center (x, y) and radius R that touches the diameter of circle k -/
structure TouchingCircle where
  x : ℝ
  y : ℝ
  R : ℝ
  touches_diameter : (-r : ℝ) ≤ x ∧ x ≤ r
  non_negative_y : y ≥ 0
  tangent_to_diameter : R = y

/-- The locus of centers of circles touching the diameter of k and with closest point at distance R from k -/
def locus_of_centers (r : ℝ) (c : TouchingCircle) : Prop :=
  (c.y - 2*r/3)^2 / (r/3)^2 - c.x^2 / (r/Real.sqrt 3)^2 = 1

theorem locus_of_centers_is_hyperbola (r : ℝ) (h : r > 0) :
  ∀ c : TouchingCircle, locus_of_centers r c ↔ 
    c.R = 2 * c.y ∧ 
    Real.sqrt (c.x^2 + c.y^2) = r - 2 * c.y ∧
    r ≥ 3 * c.y :=
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_is_hyperbola_l588_58851


namespace NUMINAMATH_CALUDE_alisha_todd_ratio_l588_58895

/-- Represents the number of gumballs given to each person and the total purchased -/
structure GumballDistribution where
  total : ℕ
  todd : ℕ
  alisha : ℕ
  bobby : ℕ
  remaining : ℕ

/-- Defines the conditions of the gumball distribution problem -/
def gumball_problem (g : GumballDistribution) : Prop :=
  g.total = 45 ∧
  g.todd = 4 ∧
  g.bobby = 4 * g.alisha - 5 ∧
  g.remaining = 6 ∧
  g.total = g.todd + g.alisha + g.bobby + g.remaining

/-- Theorem stating the ratio of gumballs given to Alisha vs Todd -/
theorem alisha_todd_ratio (g : GumballDistribution) 
  (h : gumball_problem g) : g.alisha = 2 * g.todd := by
  sorry


end NUMINAMATH_CALUDE_alisha_todd_ratio_l588_58895


namespace NUMINAMATH_CALUDE_quadratic_negative_value_condition_l588_58869

/-- Given a quadratic function f(x) = x^2 + mx + 1, 
    this theorem states that there exists a positive x₀ such that f(x₀) < 0 
    if and only if m < -2 -/
theorem quadratic_negative_value_condition (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + m*x₀ + 1 < 0) ↔ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_value_condition_l588_58869


namespace NUMINAMATH_CALUDE_original_celery_cost_l588_58818

def original_order : ℝ := 25
def new_tomatoes : ℝ := 2.20
def old_tomatoes : ℝ := 0.99
def new_lettuce : ℝ := 1.75
def old_lettuce : ℝ := 1.00
def new_celery : ℝ := 2.00
def delivery_tip : ℝ := 8.00
def new_total : ℝ := 35

theorem original_celery_cost :
  ∃ (old_celery : ℝ),
    old_celery = 0.04 ∧
    original_order = old_tomatoes + old_lettuce + old_celery ∧
    new_total = new_tomatoes + new_lettuce + new_celery + delivery_tip :=
by sorry

end NUMINAMATH_CALUDE_original_celery_cost_l588_58818


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l588_58864

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l588_58864


namespace NUMINAMATH_CALUDE_factor_and_divisor_relations_l588_58804

theorem factor_and_divisor_relations : 
  (∃ n : ℤ, 45 = 5 * n) ∧ 
  (209 % 19 = 0 ∧ 95 % 19 = 0) ∧ 
  (∃ m : ℤ, 180 = 9 * m) := by
sorry


end NUMINAMATH_CALUDE_factor_and_divisor_relations_l588_58804


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l588_58809

-- Define the propositions
variable (P Q : Prop)

-- Define the original implication
def original_statement : Prop := P → Q

-- Define the contrapositive
def contrapositive : Prop := ¬Q → ¬P

-- Define the disjunction form
def disjunction_form : Prop := ¬P ∨ Q

-- Theorem stating the equivalence of the three forms
theorem equivalence_of_statements :
  (original_statement P Q) ↔ (contrapositive P Q) ∧ (disjunction_form P Q) :=
sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l588_58809


namespace NUMINAMATH_CALUDE_base_eight_unique_for_729_l588_58868

/-- Represents a number in base b with digits d₃d₂d₁d₀ --/
def BaseRepresentation (b : ℕ) (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * b^3 + d₂ * b^2 + d₁ * b + d₀

/-- Checks if a number is in XYXY format --/
def IsXYXY (d₃ d₂ d₁ d₀ : ℕ) : Prop :=
  d₃ = d₁ ∧ d₂ = d₀ ∧ d₃ ≠ d₂

theorem base_eight_unique_for_729 :
  ∃! b : ℕ, 6 ≤ b ∧ b ≤ 9 ∧
    ∃ X Y : ℕ, X ≠ Y ∧
      BaseRepresentation b X Y X Y = 729 ∧
      IsXYXY X Y X Y :=
by sorry

end NUMINAMATH_CALUDE_base_eight_unique_for_729_l588_58868


namespace NUMINAMATH_CALUDE_chemical_mixture_percentage_l588_58862

/-- Given two solutions x and y with different compositions of chemicals a and b,
    and a mixture of these solutions, prove that the percentage of chemical a
    in the mixture is 12%. -/
theorem chemical_mixture_percentage : 
  let x_percent_a : ℝ := 10  -- Percentage of chemical a in solution x
  let x_percent_b : ℝ := 90  -- Percentage of chemical b in solution x
  let y_percent_a : ℝ := 20  -- Percentage of chemical a in solution y
  let y_percent_b : ℝ := 80  -- Percentage of chemical b in solution y
  let mixture_percent_x : ℝ := 80  -- Percentage of solution x in the mixture
  let mixture_percent_y : ℝ := 20  -- Percentage of solution y in the mixture

  -- Ensure percentages add up to 100%
  x_percent_a + x_percent_b = 100 →
  y_percent_a + y_percent_b = 100 →
  mixture_percent_x + mixture_percent_y = 100 →

  -- Calculate the percentage of chemical a in the mixture
  (mixture_percent_x * x_percent_a + mixture_percent_y * y_percent_a) / 100 = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_chemical_mixture_percentage_l588_58862


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l588_58882

theorem imaginary_part_of_one_over_one_plus_i :
  Complex.im (1 / (1 + Complex.I)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l588_58882


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_equals_one_l588_58815

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

/-- Given vectors a and b, prove that if they are parallel, then m = 1 -/
theorem parallel_vectors_imply_m_equals_one (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, m + 1)
  are_parallel a b → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_equals_one_l588_58815


namespace NUMINAMATH_CALUDE_hat_number_problem_l588_58866

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem hat_number_problem (alice_number bob_number : ℕ) : 
  alice_number ∈ Finset.range 50 →
  bob_number ∈ Finset.range 50 →
  alice_number ≠ bob_number →
  alice_number ≠ 1 →
  bob_number < alice_number →
  is_prime bob_number →
  bob_number < 10 →
  is_perfect_square (100 * bob_number + alice_number) →
  alice_number = 24 ∧ bob_number = 3 :=
by sorry

end NUMINAMATH_CALUDE_hat_number_problem_l588_58866


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l588_58890

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 2 * y - 4 = 0

-- Define the parabola equations
def parabola_equation_1 (x y : ℝ) : Prop := y^2 = 16 * x
def parabola_equation_2 (x y : ℝ) : Prop := x^2 = -8 * y

-- Define a parabola type
structure Parabola where
  focus : ℝ × ℝ
  is_on_line : line_equation focus.1 focus.2

-- Theorem statement
theorem parabola_standard_equation (p : Parabola) :
  (∃ x y : ℝ, parabola_equation_1 x y) ∨ (∃ x y : ℝ, parabola_equation_2 x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l588_58890


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l588_58859

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 189)
  (h3 : downstream_time = 7)
  : ∃ (boat_speed : ℝ), boat_speed = 22 ∧ 
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l588_58859


namespace NUMINAMATH_CALUDE_root_inequality_l588_58861

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) := exp x + x - 2
def g (x : ℝ) := log x + x - 2

-- State the theorem
theorem root_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) : f a < f 1 ∧ f 1 < f b := by
  sorry

end

end NUMINAMATH_CALUDE_root_inequality_l588_58861


namespace NUMINAMATH_CALUDE_runner_picture_probability_l588_58857

/-- Rachel's lap time in seconds -/
def rachel_lap_time : ℕ := 100

/-- Robert's lap time in seconds -/
def robert_lap_time : ℕ := 70

/-- Duration of the observation period in seconds -/
def observation_period : ℕ := 60

/-- Fraction of the track captured in the picture -/
def picture_fraction : ℚ := 1/5

/-- Time when the picture is taken (in seconds after start) -/
def picture_time : ℕ := 720  -- 12 minutes

theorem runner_picture_probability :
  let rachel_position := picture_time % rachel_lap_time
  let robert_position := robert_lap_time - (picture_time % robert_lap_time)
  let rachel_in_picture := rachel_position ≤ (rachel_lap_time * picture_fraction / 2) ∨
                           rachel_position ≥ rachel_lap_time - (rachel_lap_time * picture_fraction / 2)
  let robert_in_picture := robert_position ≤ (robert_lap_time * picture_fraction / 2) ∨
                           robert_position ≥ robert_lap_time - (robert_lap_time * picture_fraction / 2)
  (∃ t : ℕ, t ≥ picture_time ∧ t < picture_time + observation_period ∧
            rachel_in_picture ∧ robert_in_picture) →
  (1 : ℚ) / 16 = ↑(Nat.card {t : ℕ | t ≥ picture_time ∧ t < picture_time + observation_period ∧
                              rachel_in_picture ∧ robert_in_picture}) / observation_period :=
by sorry

end NUMINAMATH_CALUDE_runner_picture_probability_l588_58857


namespace NUMINAMATH_CALUDE_transformed_point_sum_l588_58821

/-- Given a function g : ℝ → ℝ such that g(8) = 5, 
    prove that (8/3, 14/9) is on the graph of 3y = g(3x)/3 + 3 
    and that the sum of its coordinates is 38/9 -/
theorem transformed_point_sum (g : ℝ → ℝ) (h : g 8 = 5) : 
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end NUMINAMATH_CALUDE_transformed_point_sum_l588_58821


namespace NUMINAMATH_CALUDE_class_size_difference_l588_58878

theorem class_size_difference (students : ℕ) (teachers : ℕ) (enrollments : List ℕ) : 
  students = 120 →
  teachers = 6 →
  enrollments = [60, 30, 15, 5, 5, 5] →
  (enrollments.sum = students) →
  (enrollments.length = teachers) →
  let t : ℚ := (enrollments.sum : ℚ) / teachers
  let s : ℚ := (enrollments.map (λ x => x * x)).sum / students
  t - s = -20 := by
  sorry

#check class_size_difference

end NUMINAMATH_CALUDE_class_size_difference_l588_58878


namespace NUMINAMATH_CALUDE_problem_statement_l588_58830

theorem problem_statement : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l588_58830


namespace NUMINAMATH_CALUDE_bicycle_exchange_point_exists_l588_58852

/-- Represents the problem of finding the optimal bicycle exchange point --/
theorem bicycle_exchange_point_exists :
  ∃ x : ℝ, 0 < x ∧ x < 20 ∧
  (x / 10 + (20 - x) / 4 = (20 - x) / 8 + x / 5) := by
  sorry

#check bicycle_exchange_point_exists

end NUMINAMATH_CALUDE_bicycle_exchange_point_exists_l588_58852


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l588_58888

/-- Proves that given specific conditions for a round trip, the return speed is 37.5 mph -/
theorem round_trip_speed_calculation (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) :
  distance = 150 →
  speed_ab = 75 →
  avg_speed = 50 →
  (2 * distance) / (distance / speed_ab + distance / ((2 * distance) / (2 * distance / avg_speed - distance / speed_ab))) = avg_speed →
  (2 * distance) / (2 * distance / avg_speed - distance / speed_ab) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l588_58888


namespace NUMINAMATH_CALUDE_range_of_a_l588_58848

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (Set.compl A ∪ B a = U) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l588_58848


namespace NUMINAMATH_CALUDE_two_consecutive_count_l588_58805

/-- Represents the number of balls in the box -/
def n : ℕ := 5

/-- Represents the number of people drawing balls -/
def k : ℕ := 3

/-- Counts the number of ways to draw balls with exactly two consecutive numbers -/
def count_two_consecutive (n k : ℕ) : ℕ :=
  sorry

theorem two_consecutive_count :
  count_two_consecutive n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_two_consecutive_count_l588_58805


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l588_58831

/-- Represents Cherry's delivery service earnings --/
def cherry_earnings : ℝ → ℝ → ℝ → ℝ → ℝ := λ price_small price_large num_small num_large =>
  (price_small * num_small + price_large * num_large) * 7

/-- Theorem stating Cherry's weekly earnings --/
theorem cherry_weekly_earnings :
  let price_small := 2.5
  let price_large := 4
  let num_small := 4
  let num_large := 2
  cherry_earnings price_small price_large num_small num_large = 126 :=
by sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l588_58831


namespace NUMINAMATH_CALUDE_square_root_of_four_l588_58884

theorem square_root_of_four :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_square_root_of_four_l588_58884


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l588_58840

theorem polynomial_division_theorem (x : ℝ) :
  x^5 + 8 = (x + 2) * (x^4 - 2*x^3 + 4*x^2 - 8*x + 16) + (-24) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l588_58840


namespace NUMINAMATH_CALUDE_prob_two_red_is_two_fifths_l588_58816

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- The number of balls drawn from the bag -/
def num_drawn : ℕ := 2

/-- The probability of drawing two red balls -/
def prob_two_red : ℚ := (num_red_balls.choose num_drawn : ℚ) / (total_balls.choose num_drawn)

theorem prob_two_red_is_two_fifths : prob_two_red = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_is_two_fifths_l588_58816


namespace NUMINAMATH_CALUDE_solution_set_l588_58806

/-- A function f : ℝ → ℝ satisfying certain properties -/
axiom f : ℝ → ℝ

/-- The derivative of f -/
axiom f' : ℝ → ℝ

/-- f(x-1) is an odd function -/
axiom f_odd : ∀ x, f ((-x) - 1) = -f (x - 1)

/-- For x < -1, (x+1)[f(x) + (x+1)f'(x)] < 0 -/
axiom f_property : ∀ x, x < -1 → (x + 1) * (f x + (x + 1) * f' x) < 0

/-- The solution set for xf(x-1) > f(0) is (-1, 1) -/
theorem solution_set : 
  {x : ℝ | x * f (x - 1) > f 0} = Set.Ioo (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_l588_58806


namespace NUMINAMATH_CALUDE_video_game_lives_l588_58893

theorem video_game_lives (initial_players : Nat) (quitting_players : Nat) (lives_per_player : Nat) : 
  initial_players = 20 → quitting_players = 10 → lives_per_player = 7 → 
  (initial_players - quitting_players) * lives_per_player = 70 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l588_58893


namespace NUMINAMATH_CALUDE_light_bulb_probability_l588_58850

/-- The number of screw-in light bulbs in the box -/
def screwIn : ℕ := 3

/-- The number of bayonet light bulbs in the box -/
def bayonet : ℕ := 5

/-- The total number of light bulbs in the box -/
def totalBulbs : ℕ := screwIn + bayonet

/-- The number of draws -/
def draws : ℕ := 5

/-- The probability of drawing all screw-in light bulbs by the 5th draw -/
def probability : ℚ := 3 / 28

theorem light_bulb_probability :
  probability = (Nat.choose screwIn (screwIn - 1) * Nat.choose bayonet (draws - screwIn) * Nat.factorial (draws - 1)) /
                (Nat.choose totalBulbs draws * Nat.factorial draws) :=
sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l588_58850


namespace NUMINAMATH_CALUDE_profit_is_24000_l588_58841

def initial_value : ℝ := 150000
def depreciation_rate : ℝ := 0.22
def selling_price : ℝ := 115260
def years : ℕ := 2

def value_after_years (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 - rate) ^ years

def profit (initial : ℝ) (rate : ℝ) (years : ℕ) (selling_price : ℝ) : ℝ :=
  selling_price - value_after_years initial rate years

theorem profit_is_24000 :
  profit initial_value depreciation_rate years selling_price = 24000 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_24000_l588_58841


namespace NUMINAMATH_CALUDE_determinant_transformation_l588_58896

theorem determinant_transformation (p q r s : ℝ) 
  (h : Matrix.det !![p, q; r, s] = 6) : 
  Matrix.det !![p, 9*p + 4*q; r, 9*r + 4*s] = 24 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l588_58896


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_l588_58855

theorem twenty_five_percent_less_than_eighty (x : ℝ) : x = 40 ↔ 80 - 0.25 * 80 = x + 0.5 * x := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_l588_58855


namespace NUMINAMATH_CALUDE_max_value_expression_l588_58826

theorem max_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  (1 / (a^2 - 4*a + 9)) + (1 / (b^2 - 4*b + 9)) + (1 / (c^2 - 4*c + 9)) ≤ 7/18 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l588_58826


namespace NUMINAMATH_CALUDE_local_max_implies_neg_local_min_l588_58856

open Function Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define x₀ as a non-zero real number
variable (x₀ : ℝ)
variable (hx₀ : x₀ ≠ 0)

-- Define that x₀ is a local maximum point of f
def IsLocalMaxAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀

-- Define local minimum
def IsLocalMinAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x

-- State the theorem
theorem local_max_implies_neg_local_min
  (h : IsLocalMaxAt f x₀) :
  IsLocalMinAt (fun x ↦ -f (-x)) (-x₀) :=
sorry

end NUMINAMATH_CALUDE_local_max_implies_neg_local_min_l588_58856


namespace NUMINAMATH_CALUDE_rahul_work_time_l588_58822

theorem rahul_work_time (meena_time : ℝ) (combined_time : ℝ) (rahul_time : ℝ) : 
  meena_time = 10 →
  combined_time = 10 / 3 →
  1 / rahul_time + 1 / meena_time = 1 / combined_time →
  rahul_time = 5 := by
sorry

end NUMINAMATH_CALUDE_rahul_work_time_l588_58822


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_line_through_1_2_parallel_to_x_axis_l588_58832

/-- A line parallel to the x-axis passing through a point (x₀, y₀) has the equation y = y₀ -/
theorem line_parallel_to_x_axis (x₀ y₀ : ℝ) :
  let line := {(x, y) : ℝ × ℝ | y = y₀}
  (∀ (x : ℝ), (x, y₀) ∈ line) ∧ ((x₀, y₀) ∈ line) → 
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = y₀ :=
by sorry

/-- The equation of the line passing through (1,2) and parallel to the x-axis is y = 2 -/
theorem line_through_1_2_parallel_to_x_axis :
  let line := {(x, y) : ℝ × ℝ | y = 2}
  (∀ (x : ℝ), (x, 2) ∈ line) ∧ ((1, 2) ∈ line) → 
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_line_through_1_2_parallel_to_x_axis_l588_58832


namespace NUMINAMATH_CALUDE_wendy_first_day_miles_l588_58808

theorem wendy_first_day_miles (total_miles second_day_miles third_day_miles : ℕ) 
  (h1 : total_miles = 493)
  (h2 : second_day_miles = 223)
  (h3 : third_day_miles = 145) :
  total_miles - (second_day_miles + third_day_miles) = 125 := by
  sorry

end NUMINAMATH_CALUDE_wendy_first_day_miles_l588_58808


namespace NUMINAMATH_CALUDE_magnitude_of_vector_2_neg1_l588_58887

/-- The magnitude of a 2D vector (2, -1) is √5 -/
theorem magnitude_of_vector_2_neg1 :
  let a : Fin 2 → ℝ := ![2, -1]
  Real.sqrt ((a 0) ^ 2 + (a 1) ^ 2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_2_neg1_l588_58887


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_range_l588_58872

theorem sqrt_x_minus_5_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_range_l588_58872


namespace NUMINAMATH_CALUDE_total_population_l588_58879

/-- Represents the number of boys, girls, and teachers in a school -/
structure School where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls
  t : ℕ  -- number of teachers

/-- The conditions of the school population -/
def school_conditions (s : School) : Prop :=
  s.b = 4 * s.g ∧ s.g = 5 * s.t

/-- The theorem stating that the total population is 26 times the number of teachers -/
theorem total_population (s : School) (h : school_conditions s) : 
  s.b + s.g + s.t = 26 * s.t := by
  sorry

end NUMINAMATH_CALUDE_total_population_l588_58879


namespace NUMINAMATH_CALUDE_rectangular_stadium_length_l588_58880

theorem rectangular_stadium_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 800) 
  (h2 : breadth = 300) : 
  2 * (breadth + 100) = perimeter :=
by sorry

end NUMINAMATH_CALUDE_rectangular_stadium_length_l588_58880


namespace NUMINAMATH_CALUDE_root_set_equivalence_l588_58802

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- The set of real roots of f(x) = 0 -/
def rootSet (a : ℝ) : Set ℝ := {x : ℝ | f a x = 0}

/-- The set of real roots of f(f(x)) = 0 -/
def composedRootSet (a : ℝ) : Set ℝ := {x : ℝ | f a (f a x) = 0}

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem root_set_equivalence :
  ∀ a : ℝ, (rootSet a = composedRootSet a ∧ rootSet a ≠ ∅) ↔ 0 ≤ a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_root_set_equivalence_l588_58802


namespace NUMINAMATH_CALUDE_constant_sum_through_P_l588_58846

/-- The function f(x) = x³ + 3x² + x -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + x

/-- The point P on the graph of f -/
def P : ℝ × ℝ := (-1, f (-1))

theorem constant_sum_through_P :
  ∃ (y : ℝ), ∀ (x₁ x₂ : ℝ),
    x₁ ≠ -1 → x₂ ≠ -1 →
    (x₂ - (-1)) * (f x₁ - f (-1)) = (x₁ - (-1)) * (f x₂ - f (-1)) →
    f x₁ + f x₂ = y :=
  sorry

end NUMINAMATH_CALUDE_constant_sum_through_P_l588_58846


namespace NUMINAMATH_CALUDE_sum_of_roots_l588_58829

theorem sum_of_roots (k d x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ →
  (4 * x₁^2 - k * x₁ = d) →
  (4 * x₂^2 - k * x₂ = d) →
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l588_58829


namespace NUMINAMATH_CALUDE_heptagon_internal_angles_sum_heptagon_internal_angles_sum_is_540_l588_58838

/-- The sum of internal angles of a heptagon, excluding the central point when divided into triangles -/
theorem heptagon_internal_angles_sum : ℝ :=
  let n : ℕ := 7  -- number of vertices in the heptagon
  let polygon_angle_sum : ℝ := (n - 2) * 180
  let central_angle_sum : ℝ := 360
  polygon_angle_sum - central_angle_sum

/-- Proof that the sum of internal angles of a heptagon, excluding the central point, is 540 degrees -/
theorem heptagon_internal_angles_sum_is_540 :
  heptagon_internal_angles_sum = 540 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_internal_angles_sum_heptagon_internal_angles_sum_is_540_l588_58838


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l588_58814

theorem min_value_quadratic_form (x y : ℤ) (h : (x, y) ≠ (0, 0)) :
  |5 * x^2 + 11 * x * y - 5 * y^2| ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l588_58814


namespace NUMINAMATH_CALUDE_exist_non_adjacent_colors_l588_58858

/-- Represents a coloring of a 50x50 square --/
def Coloring := Fin 50 → Fin 50 → Fin 100

/-- No single-color domino exists in the coloring --/
def NoSingleColorDomino (c : Coloring) : Prop :=
  ∀ i j, (i < 49 → c i j ≠ c (i+1) j) ∧ (j < 49 → c i j ≠ c i (j+1))

/-- All colors are present in the coloring --/
def AllColorsPresent (c : Coloring) : Prop :=
  ∀ color, ∃ i j, c i j = color

/-- Two colors are not adjacent if they don't appear next to each other anywhere --/
def ColorsNotAdjacent (c : Coloring) (color1 color2 : Fin 100) : Prop :=
  ∀ i j, (i < 49 → (c i j ≠ color1 ∨ c (i+1) j ≠ color2) ∧ (c i j ≠ color2 ∨ c (i+1) j ≠ color1)) ∧
         (j < 49 → (c i j ≠ color1 ∨ c i (j+1) ≠ color2) ∧ (c i j ≠ color2 ∨ c i (j+1) ≠ color1))

/-- Main theorem: There exist two non-adjacent colors in any valid coloring --/
theorem exist_non_adjacent_colors (c : Coloring) 
  (h1 : NoSingleColorDomino c) (h2 : AllColorsPresent c) : 
  ∃ color1 color2, ColorsNotAdjacent c color1 color2 := by
  sorry

end NUMINAMATH_CALUDE_exist_non_adjacent_colors_l588_58858


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l588_58867

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l588_58867
