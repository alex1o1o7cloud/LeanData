import Mathlib

namespace NUMINAMATH_CALUDE_tony_fish_problem_l1656_165683

/-- The number of fish Tony has after a given number of years -/
def fish_count (initial : ℕ) (years : ℕ) : ℕ :=
  initial + years

theorem tony_fish_problem (x : ℕ) :
  fish_count x 5 = 7 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_tony_fish_problem_l1656_165683


namespace NUMINAMATH_CALUDE_quadratic_negative_range_l1656_165614

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_negative_range (a b c : ℝ) :
  (quadratic a b c (-1) = 4) →
  (quadratic a b c 0 = 0) →
  (∃ n, quadratic a b c 1 = n) →
  (∃ m, quadratic a b c 2 = m) →
  (quadratic a b c 3 = 4) →
  (∀ x : ℝ, quadratic a b c x < 0 ↔ 0 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_range_l1656_165614


namespace NUMINAMATH_CALUDE_tate_had_32_tickets_l1656_165682

/-- The number of tickets Tate and Peyton have together -/
def total_tickets : ℕ := 51

/-- The number of additional tickets Tate buys -/
def additional_tickets : ℕ := 2

/-- Tate's initial number of tickets -/
def tate_initial_tickets : ℕ → Prop := λ t => 
  ∃ (tate_total peyton_total : ℕ),
    tate_total = t + additional_tickets ∧
    peyton_total = tate_total / 2 ∧
    tate_total + peyton_total = total_tickets

theorem tate_had_32_tickets : tate_initial_tickets 32 := by
  sorry

end NUMINAMATH_CALUDE_tate_had_32_tickets_l1656_165682


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1656_165664

theorem unique_solution_floor_equation :
  ∃! b : ℝ, b + ⌊b⌋ = 22.6 ∧ b = 11.6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1656_165664


namespace NUMINAMATH_CALUDE_three_300deg_arcs_must_intersect_l1656_165632

/-- A great circle arc on a sphere -/
structure GreatCircleArc where
  /-- The angle of the arc in degrees -/
  angle : ℝ

/-- A configuration of three great circle arcs on a sphere -/
structure ThreeArcsConfiguration where
  arc1 : GreatCircleArc
  arc2 : GreatCircleArc
  arc3 : GreatCircleArc

/-- Predicate to check if two arcs intersect -/
def intersect (a b : GreatCircleArc) : Prop := sorry

/-- Theorem: It's impossible to place 3 great circle arcs of 300° each on a sphere with no common points -/
theorem three_300deg_arcs_must_intersect (config : ThreeArcsConfiguration) :
  config.arc1.angle = 300 ∧ config.arc2.angle = 300 ∧ config.arc3.angle = 300 →
  intersect config.arc1 config.arc2 ∨ intersect config.arc2 config.arc3 ∨ intersect config.arc3 config.arc1 :=
by sorry

end NUMINAMATH_CALUDE_three_300deg_arcs_must_intersect_l1656_165632


namespace NUMINAMATH_CALUDE_mango_ratio_l1656_165690

theorem mango_ratio (total_mangoes : ℕ) (unripe_fraction : ℚ) (kept_unripe : ℕ) 
  (mangoes_per_jar : ℕ) (jars_made : ℕ) : 
  total_mangoes = 54 →
  unripe_fraction = 2/3 →
  kept_unripe = 16 →
  mangoes_per_jar = 4 →
  jars_made = 5 →
  (total_mangoes * (1 - unripe_fraction) : ℚ) / total_mangoes = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_mango_ratio_l1656_165690


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1656_165647

theorem more_girls_than_boys :
  ∀ (girls boys : ℕ),
    girls > boys →
    girls + boys = 41 →
    girls = 22 →
    girls - boys = 3 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1656_165647


namespace NUMINAMATH_CALUDE_bicycle_distance_l1656_165661

/-- The distance traveled by a bicycle in 30 minutes, given that it travels 1/2 as fast as a motorcycle going 90 miles per hour -/
theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time : ℝ) :
  motorcycle_speed = 90 →
  bicycle_speed_ratio = 1/2 →
  time = 1/2 →
  bicycle_speed_ratio * motorcycle_speed * time = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_distance_l1656_165661


namespace NUMINAMATH_CALUDE_A_intersect_B_l1656_165686

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1656_165686


namespace NUMINAMATH_CALUDE_pencil_distribution_l1656_165623

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 1001 →
  num_students = 91 →
  (num_pens % num_students = 0) →
  (num_pencils % num_students = 0) →
  ∃ k : ℕ, num_pencils = 91 * k :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1656_165623


namespace NUMINAMATH_CALUDE_sum_pascal_triangle_rows_8_to_10_l1656_165646

-- Define a function to calculate the sum of interior numbers for a given row
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Define the sum of interior numbers for rows 8, 9, and 10
def sumRows8to10 : ℕ := sumInteriorNumbers 8 + sumInteriorNumbers 9 + sumInteriorNumbers 10

-- Theorem statement
theorem sum_pascal_triangle_rows_8_to_10 : sumRows8to10 = 890 := by
  sorry

end NUMINAMATH_CALUDE_sum_pascal_triangle_rows_8_to_10_l1656_165646


namespace NUMINAMATH_CALUDE_line_length_difference_is_correct_l1656_165606

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.678934

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := 3.33457689

/-- The difference between the white and blue line lengths -/
def line_length_difference : ℝ := white_line_length - blue_line_length

/-- Theorem stating that the difference between the white and blue line lengths is 4.34435711 inches -/
theorem line_length_difference_is_correct : 
  line_length_difference = 4.34435711 := by sorry

end NUMINAMATH_CALUDE_line_length_difference_is_correct_l1656_165606


namespace NUMINAMATH_CALUDE_divide_by_ten_equals_two_l1656_165635

theorem divide_by_ten_equals_two : ∃ x : ℚ, x * 5 = 100 ∧ x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_ten_equals_two_l1656_165635


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1656_165624

theorem negation_of_universal_statement :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ↔ ∃ a b : ℝ, a ≤ b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1656_165624


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l1656_165667

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area_from_polar_equation (θ : ℝ) :
  let r := 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
      ∃ θ, x = r * Real.cos θ ∧ y = r * Real.sin θ) ∧
    π * radius^2 = 25 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l1656_165667


namespace NUMINAMATH_CALUDE_luke_sticker_problem_l1656_165678

/-- The number of stickers Luke used to decorate the greeting card -/
def stickers_used_for_card (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given_to_sister : ℕ) (final : ℕ) : ℕ :=
  initial + bought + birthday - given_to_sister - final

/-- Theorem stating the number of stickers Luke used for the greeting card -/
theorem luke_sticker_problem :
  stickers_used_for_card 20 12 20 5 39 = 8 := by
  sorry

end NUMINAMATH_CALUDE_luke_sticker_problem_l1656_165678


namespace NUMINAMATH_CALUDE_equal_roots_count_l1656_165658

/-- The number of real values of p for which the roots of x^2 - px + p^2 = 0 are equal is 1 -/
theorem equal_roots_count (p : ℝ) : ∃! p, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! x, x^2 - p*x + p^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_count_l1656_165658


namespace NUMINAMATH_CALUDE_percentage_absent_l1656_165693

/-- Given a class of 50 students with 45 present, prove that 10% are absent. -/
theorem percentage_absent (total : ℕ) (present : ℕ) (h1 : total = 50) (h2 : present = 45) :
  (total - present) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_absent_l1656_165693


namespace NUMINAMATH_CALUDE_limit_of_f_l1656_165698

open Real

noncomputable def f (x : ℝ) : ℝ :=
  tan (cos x + sin ((x - 1) / (x + 1)) * cos ((x + 1) / (x - 1)))

theorem limit_of_f :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - tan (cos 1)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_f_l1656_165698


namespace NUMINAMATH_CALUDE_work_hours_difference_l1656_165650

def hours_week1 : ℕ := 35
def hours_week2 : ℕ := 35
def hours_week3 : ℕ := 48
def hours_week4 : ℕ := 48

theorem work_hours_difference : 
  (hours_week3 + hours_week4) - (hours_week1 + hours_week2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_difference_l1656_165650


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1656_165608

/-- The probability of forming a convex quadrilateral by selecting 4 chords at random from 8 points on a circle -/
theorem convex_quadrilateral_probability (n : ℕ) (k : ℕ) : 
  n = 8 → k = 4 → (Nat.choose n 2).choose k / (Nat.choose n k) = 2 / 585 :=
by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1656_165608


namespace NUMINAMATH_CALUDE_custom_op_five_three_l1656_165674

/-- Custom binary operation " defined as m " n = n^2 - m -/
def custom_op (m n : ℤ) : ℤ := n^2 - m

/-- Theorem stating that 5 " 3 = 4 -/
theorem custom_op_five_three : custom_op 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_five_three_l1656_165674


namespace NUMINAMATH_CALUDE_P_homogeneous_P_sum_condition_P_initial_condition_P_unique_l1656_165657

/-- A homogeneous polynomial of degree n in x and y satisfying specific conditions -/
def P (n : ℕ+) : ℝ → ℝ → ℝ := fun x y ↦ (x + y) ^ (n.val - 1) * (x - 2*y)

/-- P is a homogeneous polynomial of degree n -/
theorem P_homogeneous (n : ℕ+) (t x y : ℝ) : 
  P n (t * x) (t * y) = t ^ n.val * P n x y := by sorry

/-- P satisfies the given sum condition -/
theorem P_sum_condition (n : ℕ+) (a b c : ℝ) :
  P n (a + b) c + P n (b + c) a + P n (c + a) b = 0 := by sorry

/-- P satisfies the given initial condition -/
theorem P_initial_condition (n : ℕ+) : P n 1 0 = 1 := by sorry

/-- P is the unique polynomial satisfying all conditions -/
theorem P_unique (n : ℕ+) (Q : ℝ → ℝ → ℝ) 
  (h_homogeneous : ∀ t x y, Q (t * x) (t * y) = t ^ n.val * Q x y)
  (h_sum : ∀ a b c, Q (a + b) c + Q (b + c) a + Q (c + a) b = 0)
  (h_initial : Q 1 0 = 1) :
  Q = P n := by sorry

end NUMINAMATH_CALUDE_P_homogeneous_P_sum_condition_P_initial_condition_P_unique_l1656_165657


namespace NUMINAMATH_CALUDE_negative_one_third_m_meets_requirements_l1656_165648

/-- Represents an algebraic expression -/
inductive AlgebraicExpression
  | Fraction : ℚ → String → AlgebraicExpression
  | Mixed : ℕ → ℚ → String → AlgebraicExpression
  | Division : String → String → AlgebraicExpression
  | Multiplication : String → ℕ → AlgebraicExpression

/-- Checks if an algebraic expression meets the writing requirements -/
def meetsWritingRequirements (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.Fraction _ _ => true
  | _ => false

/-- The theorem stating that -1/3m meets the writing requirements -/
theorem negative_one_third_m_meets_requirements :
  meetsWritingRequirements (AlgebraicExpression.Fraction (-1/3) "m") :=
by sorry

end NUMINAMATH_CALUDE_negative_one_third_m_meets_requirements_l1656_165648


namespace NUMINAMATH_CALUDE_pizza_order_count_l1656_165675

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 2) (h2 : total_slices = 28) :
  total_slices / slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_count_l1656_165675


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_property_l1656_165615

theorem polynomial_roots_sum_property (x₁ x₂ : ℝ) (h₁ : x₁^2 - 6*x₁ + 1 = 0) (h₂ : x₂^2 - 6*x₂ + 1 = 0) :
  ∀ n : ℕ, ∃ k : ℤ, (x₁^n + x₂^n = k) ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_property_l1656_165615


namespace NUMINAMATH_CALUDE_max_eccentricity_sum_l1656_165652

/-- Given an ellipse with eccentricity e₁ and a hyperbola with eccentricity e₂ sharing the same foci,
    if the minor axis of the ellipse is three times the length of the conjugate axis of the hyperbola,
    then the maximum value of 1/e₁ + 1/e₂ is 10/3. -/
theorem max_eccentricity_sum (e₁ e₂ : ℝ) (h_ellipse : 0 < e₁ ∧ e₁ < 1) (h_hyperbola : e₂ > 1)
  (h_foci : ∃ (c : ℝ), c > 0 ∧ c^2 * e₁^2 = c^2 * e₂^2)
  (h_axes : ∃ (b₁ b₂ : ℝ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ = 3 * b₂) :
  (∀ x y : ℝ, x > 0 ∧ y > 1 → 1/x + 1/y ≤ 10/3) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 1 ∧ 1/x + 1/y = 10/3) :=
sorry

end NUMINAMATH_CALUDE_max_eccentricity_sum_l1656_165652


namespace NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l1656_165604

def f (n : ℕ+) : ℤ := (n : ℤ)^3 - 9*(n : ℤ)^2 + 23*(n : ℤ) - 17

theorem not_prime_for_all_positive_n : ∀ n : ℕ+, ¬(Nat.Prime (Int.natAbs (f n))) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l1656_165604


namespace NUMINAMATH_CALUDE_five_equal_angles_l1656_165631

theorem five_equal_angles (rays : ℕ) (angle : ℝ) : 
  rays = 5 → 
  rays * angle = 360 → 
  angle = 72 := by
sorry

end NUMINAMATH_CALUDE_five_equal_angles_l1656_165631


namespace NUMINAMATH_CALUDE_company_bonus_fund_l1656_165691

theorem company_bonus_fund : ∃ (n : ℕ) (initial_fund : ℕ), 
  (60 * n - 10 = initial_fund) ∧ 
  (50 * n + 110 = initial_fund) ∧ 
  (initial_fund = 710) := by
  sorry

end NUMINAMATH_CALUDE_company_bonus_fund_l1656_165691


namespace NUMINAMATH_CALUDE_upper_limit_is_1575_l1656_165628

def upper_limit (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧
  (∃ (S : Finset ℕ), S.card = 8 ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ m ∧ 25 ∣ x ∧ 35 ∣ x) ∧
    (∀ y : ℕ, y ≥ 1 ∧ y ≤ m ∧ 25 ∣ y ∧ 35 ∣ y → y ∈ S)) ∧
  (∀ k < m, ¬∃ (T : Finset ℕ), T.card = 8 ∧
    (∀ x ∈ T, x ≥ 1 ∧ x ≤ k ∧ 25 ∣ x ∧ 35 ∣ x) ∧
    (∀ y : ℕ, y ≥ 1 ∧ y ≤ k ∧ 25 ∣ y ∧ 35 ∣ y → y ∈ T))

theorem upper_limit_is_1575 : upper_limit 1575 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_is_1575_l1656_165628


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_l1656_165654

theorem complex_root_modulus_one (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ 6 ∣ (n + 2) :=
sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_l1656_165654


namespace NUMINAMATH_CALUDE_chameleon_count_denis_chameleons_l1656_165677

theorem chameleon_count : ℕ → Prop :=
  fun total : ℕ =>
    ∃ (initial_brown : ℕ),
      let initial_red : ℕ := 5 * initial_brown
      let final_brown : ℕ := initial_brown - 2
      let final_red : ℕ := initial_red + 2
      (final_red = 8 * final_brown) ∧
      (total = initial_brown + initial_red) ∧
      (total = 36)

theorem denis_chameleons : chameleon_count 36 := by
  sorry

end NUMINAMATH_CALUDE_chameleon_count_denis_chameleons_l1656_165677


namespace NUMINAMATH_CALUDE_license_plate_increase_l1656_165656

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible license plates in the old scheme -/
def old_scheme_count : ℕ := num_letters * (num_digits ^ 5)

/-- The number of possible license plates in the new scheme -/
def new_scheme_count : ℕ := (num_letters ^ 2) * (num_digits ^ 4)

/-- The ratio of new scheme count to old scheme count -/
def license_plate_ratio : ℚ := new_scheme_count / old_scheme_count

theorem license_plate_increase : license_plate_ratio = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l1656_165656


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1656_165611

def A : Set ℝ := {1, 4}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1656_165611


namespace NUMINAMATH_CALUDE_zero_in_interval_l1656_165649

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ x₀ ∈ Set.Ico 2 3 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1656_165649


namespace NUMINAMATH_CALUDE_brads_running_speed_l1656_165639

/-- Calculates Brad's running speed given the problem conditions -/
theorem brads_running_speed 
  (maxwell_speed : ℝ) 
  (total_distance : ℝ) 
  (maxwell_time : ℝ) 
  (brad_delay : ℝ) 
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 14)
  (h3 : maxwell_time = 2)
  (h4 : brad_delay = 1) : 
  (total_distance - maxwell_speed * maxwell_time) / (maxwell_time - brad_delay) = 6 := by
  sorry

#check brads_running_speed

end NUMINAMATH_CALUDE_brads_running_speed_l1656_165639


namespace NUMINAMATH_CALUDE_largest_n_with_difference_seven_l1656_165684

/-- A function that checks if a list of natural numbers contains two numbers with a difference of 7 -/
def hasDifferenceSeven (list : List Nat) : Prop :=
  ∃ x y, x ∈ list ∧ y ∈ list ∧ x ≠ y ∧ (x - y = 7 ∨ y - x = 7)

/-- A function that checks if all selections of 50 numbers from 1 to n satisfy the difference condition -/
def allSelectionsHaveDifferenceSeven (n : Nat) : Prop :=
  ∀ list : List Nat, list.Nodup → list.length = 50 → (∀ x ∈ list, x ≥ 1 ∧ x ≤ n) →
    hasDifferenceSeven list

/-- The main theorem stating that 98 is the largest number satisfying the condition -/
theorem largest_n_with_difference_seven :
  allSelectionsHaveDifferenceSeven 98 ∧
  ¬(allSelectionsHaveDifferenceSeven 99) := by
  sorry

#check largest_n_with_difference_seven

end NUMINAMATH_CALUDE_largest_n_with_difference_seven_l1656_165684


namespace NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l1656_165666

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

theorem f_monotonicity_and_minimum (m : ℝ) (h : m > -1) :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 3 → f x > f y) ∧
  (∀ x ∈ Set.Icc (-1) m, 
    f x ≥ (if m ≤ 3 then f m else -25)) ∧
  (if m ≤ 3 
   then ∀ x ∈ Set.Icc (-1) m, f x ≥ m^3 - 3*m^2 - 9*m + 2
   else ∀ x ∈ Set.Icc (-1) m, f x ≥ -25) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l1656_165666


namespace NUMINAMATH_CALUDE_arithmetic_mean_negative_seven_to_six_l1656_165697

def arithmetic_mean (a b : Int) : ℚ :=
  let n := b - a + 1
  let sum := (a + b) * n / 2
  sum / n

theorem arithmetic_mean_negative_seven_to_six :
  arithmetic_mean (-7) 6 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_negative_seven_to_six_l1656_165697


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1656_165694

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1656_165694


namespace NUMINAMATH_CALUDE_symmetry_about_one_zero_a_symmetry_about_one_zero_b_symmetry_about_one_zero_c_l1656_165643

variable (f : ℝ → ℝ)

-- Statement 1
theorem symmetry_about_one_zero_a (h : ∀ x : ℝ, f (x + 2) = -f (-x)) :
  ∀ x : ℝ, f (x + 1) = -f (1 - x) := by sorry

-- Statement 2
theorem symmetry_about_one_zero_b :
  ∀ x : ℝ, f x = -(-f (2 - x)) := by sorry

-- Statement 3
theorem symmetry_about_one_zero_c :
  ∀ x : ℝ, f (-1 + x) - f (1 - x) = -(f (-1 + (2 - x)) - f (1 - (2 - x))) := by sorry

end NUMINAMATH_CALUDE_symmetry_about_one_zero_a_symmetry_about_one_zero_b_symmetry_about_one_zero_c_l1656_165643


namespace NUMINAMATH_CALUDE_probability_factor_less_than_10_l1656_165627

def factors_of_120 : Finset ℕ := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}

def factors_less_than_10 : Finset ℕ := {1, 2, 3, 4, 5, 6, 8, 9}

theorem probability_factor_less_than_10 : 
  (factors_less_than_10.card : ℚ) / (factors_of_120.card : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_10_l1656_165627


namespace NUMINAMATH_CALUDE_correct_volunteer_assignment_l1656_165641

/-- The number of ways to select and assign volunteers to tour groups --/
def assignVolunteers (totalVolunteers femaleVolunteers tourGroups : ℕ) : ℕ :=
  let maleVolunteers := totalVolunteers - femaleVolunteers
  let totalCombinations := Nat.choose totalVolunteers tourGroups
  let allFemaleCombinations := Nat.choose femaleVolunteers tourGroups
  let allMaleCombinations := Nat.choose maleVolunteers tourGroups
  let validCombinations := totalCombinations - allFemaleCombinations - allMaleCombinations
  validCombinations * Nat.factorial tourGroups

/-- Theorem stating the correct number of ways to assign volunteers --/
theorem correct_volunteer_assignment :
  assignVolunteers 10 4 3 = 576 :=
by sorry

end NUMINAMATH_CALUDE_correct_volunteer_assignment_l1656_165641


namespace NUMINAMATH_CALUDE_train_length_is_140_meters_l1656_165620

-- Define the given conditions
def train_speed : Real := 45 -- km/hr
def bridge_crossing_time : Real := 30 -- seconds
def bridge_length : Real := 235 -- meters

-- Define the theorem
theorem train_length_is_140_meters :
  let speed_mps := train_speed * 1000 / 3600 -- Convert km/hr to m/s
  let total_distance := speed_mps * bridge_crossing_time
  let train_length := total_distance - bridge_length
  train_length = 140 := by sorry

end NUMINAMATH_CALUDE_train_length_is_140_meters_l1656_165620


namespace NUMINAMATH_CALUDE_function_inequality_implies_domain_restriction_l1656_165618

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem function_inequality_implies_domain_restriction
  (h_increasing : Increasing f)
  (h_inequality : ∀ x, f x < f (2*x - 3)) :
  ∀ x, f x < f (2*x - 3) → x > 3 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_domain_restriction_l1656_165618


namespace NUMINAMATH_CALUDE_integral_multiple_equals_2400_l1656_165607

theorem integral_multiple_equals_2400 : ∃ (x : ℤ), x = 4 * 595 ∧ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_integral_multiple_equals_2400_l1656_165607


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1656_165617

theorem polar_to_cartesian (r : ℝ) (θ : ℝ) (x y : ℝ) :
  r = 2 ∧ θ = 5 * π / 6 →
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  x = -Real.sqrt 3 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1656_165617


namespace NUMINAMATH_CALUDE_great_m_conference_teams_l1656_165651

/-- The number of games played when each team in a conference plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of teams in the GREAT M conference -/
def num_teams : ℕ := 8

theorem great_m_conference_teams :
  num_games num_teams = 28 ∧ num_teams > 0 := by
  sorry

#eval num_games num_teams -- Should output 28

end NUMINAMATH_CALUDE_great_m_conference_teams_l1656_165651


namespace NUMINAMATH_CALUDE_parking_lot_bikes_l1656_165602

theorem parking_lot_bikes (cars : ℕ) (total_wheels : ℕ) (car_wheels : ℕ) (bike_wheels : ℕ) : 
  cars = 14 → total_wheels = 66 → car_wheels = 4 → bike_wheels = 2 → 
  (total_wheels - cars * car_wheels) / bike_wheels = 5 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_bikes_l1656_165602


namespace NUMINAMATH_CALUDE_robins_gum_increase_l1656_165633

/-- Given Robin's initial and final gum counts, prove the number of pieces her brother gave her. -/
theorem robins_gum_increase (initial final brother_gave : ℕ) 
  (h1 : initial = 63)
  (h2 : final = 159)
  (h3 : final = initial + brother_gave) :
  brother_gave = 96 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_increase_l1656_165633


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1656_165603

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -4 < x ∧ x < 2}
def solution_set_2 : Set ℝ := {x | (-2/3 ≤ x ∧ x < 1/3) ∨ (1 < x ∧ x ≤ 2)}

-- Theorem for the first inequality
theorem inequality_solution_1 : 
  {x : ℝ | |x - 1| + |x + 3| < 6} = solution_set_1 := by sorry

-- Theorem for the second inequality
theorem inequality_solution_2 :
  {x : ℝ | 1 < |3*x - 2| ∧ |3*x - 2| < 4} = solution_set_2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1656_165603


namespace NUMINAMATH_CALUDE_standard_colony_requirements_l1656_165670

/-- Represents the type of culture medium -/
inductive CultureMedium
| Liquid
| Solid

/-- Represents the number of initial bacteria -/
inductive InitialBacteria
| One
| Many

/-- Defines a bacterial colony -/
structure BacterialColony where
  origin : InitialBacteria
  medium : CultureMedium
  visible_mass : Bool
  single_mother_cell : Bool
  significant_for_identification : Bool

/-- Defines a standard bacterial colony -/
def is_standard_colony (colony : BacterialColony) : Prop :=
  colony.origin = InitialBacteria.One ∧
  colony.medium = CultureMedium.Solid ∧
  colony.visible_mass ∧
  colony.single_mother_cell ∧
  colony.significant_for_identification

theorem standard_colony_requirements :
  ∀ (colony : BacterialColony),
    colony.visible_mass ∧
    colony.single_mother_cell ∧
    colony.significant_for_identification →
    is_standard_colony colony ↔
      colony.origin = InitialBacteria.One ∧
      colony.medium = CultureMedium.Solid :=
by sorry

end NUMINAMATH_CALUDE_standard_colony_requirements_l1656_165670


namespace NUMINAMATH_CALUDE_infinite_solutions_cubic_equation_l1656_165605

theorem infinite_solutions_cubic_equation :
  ∀ k : ℕ+, ∃ a b c : ℕ+,
    (a : ℤ)^3 + 1990 * (b : ℤ)^3 = (c : ℤ)^4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_cubic_equation_l1656_165605


namespace NUMINAMATH_CALUDE_solution_set_min_sum_reciprocals_equality_condition_l1656_165663

-- Define the inequality
def inequality (x : ℝ) : Prop := |x + 1| + |2*x - 1| ≤ 3

-- Theorem 1: Solution set of the inequality
theorem solution_set : 
  {x : ℝ | inequality x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Minimum value of the sum of reciprocals
theorem min_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 2) :
  1/a + 1/b + 1/c ≥ 9/2 := by sorry

-- Theorem 3: Condition for equality
theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 2) :
  1/a + 1/b + 1/c = 9/2 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_sum_reciprocals_equality_condition_l1656_165663


namespace NUMINAMATH_CALUDE_irrational_floor_inequality_l1656_165659

theorem irrational_floor_inequality : ∃ (a b : ℝ), 
  (a > 1) ∧ (b > 1) ∧ 
  Irrational a ∧ Irrational b ∧
  (∀ (m n : ℕ), m > 0 → n > 0 → ⌊a^m⌋ ≠ ⌊b^n⌋) :=
by
  -- Let a = 2 + √3 and b = (5 + √17)/2
  let a := 2 + Real.sqrt 3
  let b := (5 + Real.sqrt 17) / 2
  
  -- Prove the existence
  use a, b
  
  sorry -- Skip the actual proof

end NUMINAMATH_CALUDE_irrational_floor_inequality_l1656_165659


namespace NUMINAMATH_CALUDE_point_coordinates_l1656_165626

/-- A point in the second quadrant with given distances from axes -/
structure PointInSecondQuadrant where
  x : ℝ
  y : ℝ
  in_second_quadrant : x < 0 ∧ y > 0
  distance_from_x_axis : |y| = 2
  distance_from_y_axis : |x| = 5

/-- The coordinates of the point are (-5, 2) -/
theorem point_coordinates (P : PointInSecondQuadrant) : P.x = -5 ∧ P.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1656_165626


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1656_165609

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 8*x - 48 = 0 → (∃ y : ℝ, y^2 + 8*y - 48 = 0 ∧ y ≠ x) → x ≥ -12 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1656_165609


namespace NUMINAMATH_CALUDE_total_travel_time_is_58_hours_l1656_165669

/-- Represents the travel times between cities -/
structure TravelTimes where
  newOrleansToNewYork : ℝ
  newYorkToSanFrancisco : ℝ
  layoverInNewYork : ℝ

/-- The total travel time from New Orleans to San Francisco -/
def totalTravelTime (t : TravelTimes) : ℝ :=
  t.newOrleansToNewYork + t.layoverInNewYork + t.newYorkToSanFrancisco

/-- Theorem stating the total travel time is 58 hours -/
theorem total_travel_time_is_58_hours (t : TravelTimes) 
  (h1 : t.newOrleansToNewYork = 3/4 * t.newYorkToSanFrancisco)
  (h2 : t.newYorkToSanFrancisco = 24)
  (h3 : t.layoverInNewYork = 16) : 
  totalTravelTime t = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_is_58_hours_l1656_165669


namespace NUMINAMATH_CALUDE_super_lucky_years_l1656_165645

def is_super_lucky_year (Y : ℕ) : Prop :=
  ∃ (m d : ℕ), 
    1 ≤ m ∧ m ≤ 12 ∧
    1 ≤ d ∧ d ≤ 31 ∧
    m + d = 24 ∧
    m * d = 2 * ((Y % 100) / 10 + Y % 10)

theorem super_lucky_years : 
  is_super_lucky_year 2076 ∧ 
  is_super_lucky_year 2084 ∧ 
  ¬is_super_lucky_year 2070 ∧ 
  ¬is_super_lucky_year 2081 ∧ 
  ¬is_super_lucky_year 2092 :=
sorry

end NUMINAMATH_CALUDE_super_lucky_years_l1656_165645


namespace NUMINAMATH_CALUDE_edge_probability_first_20_rows_l1656_165662

/-- Represents Pascal's Triangle up to a certain number of rows -/
structure PascalTriangle (n : ℕ) where
  total_elements : ℕ
  edge_numbers : ℕ

/-- The probability of selecting an edge number from Pascal's Triangle -/
def edge_probability (pt : PascalTriangle 20) : ℚ :=
  pt.edge_numbers / pt.total_elements

/-- Theorem: The probability of selecting an edge number from the first 20 rows of Pascal's Triangle is 13/70 -/
theorem edge_probability_first_20_rows :
  ∃ (pt : PascalTriangle 20),
    pt.total_elements = 210 ∧
    pt.edge_numbers = 39 ∧
    edge_probability pt = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_edge_probability_first_20_rows_l1656_165662


namespace NUMINAMATH_CALUDE_man_speed_l1656_165655

/-- The speed of a man running opposite to a bullet train --/
theorem man_speed (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ) :
  train_length = 200 →
  train_speed = 69 →
  passing_time = 10 →
  ∃ (man_speed : ℝ), 
    (man_speed ≥ 2.9 ∧ man_speed ≤ 3.1) ∧
    (train_length / passing_time = train_speed * (1000 / 3600) + man_speed) :=
by sorry

end NUMINAMATH_CALUDE_man_speed_l1656_165655


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1656_165640

theorem triangle_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 4) (h2 : b = 4 * Real.sqrt 3) (h3 : A = 30 * π / 180) :
  let B := Real.arcsin ((b * Real.sin A) / a)
  B = 60 * π / 180 ∨ B = 120 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1656_165640


namespace NUMINAMATH_CALUDE_roots_of_unity_real_roots_l1656_165679

theorem roots_of_unity_real_roots (n : ℕ) : 
  ¬ (∀ z : ℂ, z^n = 1 → (z.im = 0 → z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_unity_real_roots_l1656_165679


namespace NUMINAMATH_CALUDE_ribbon_length_l1656_165680

theorem ribbon_length (A : ℝ) (π_estimate : ℝ) (extra : ℝ) : 
  A = 154 → π_estimate = 22 / 7 → extra = 2 →
  ∃ (r : ℝ), 
    A = π_estimate * r^2 ∧ 
    2 * π_estimate * r + extra = 46 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_l1656_165680


namespace NUMINAMATH_CALUDE_quadratic_intersection_with_x_axis_l1656_165621

theorem quadratic_intersection_with_x_axis (a b c : ℝ) :
  ∃ x : ℝ, (x - a) * (x - b) - c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_with_x_axis_l1656_165621


namespace NUMINAMATH_CALUDE_solve_diamond_equation_l1656_165695

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ :=
  a / b

-- Axioms for the diamond operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) * c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- Theorem to prove
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 504 (diamond 12 x) = 50 → x = 25 / 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_diamond_equation_l1656_165695


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1656_165630

theorem sum_of_two_numbers : ∃ (x y : ℝ), 
  3 * x - y = 20 ∧ 
  y = 17 ∧ 
  x + y = 29.333333333333332 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1656_165630


namespace NUMINAMATH_CALUDE_rectangle_area_l1656_165689

/-- Rectangle ABCD with point E on CD and point F on AC -/
structure Rectangle :=
  (A B C D E F : ℝ × ℝ)

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem rectangle_area (rect : Rectangle) : 
  -- Point E is located one-third of the way along side CD
  rect.E.1 = rect.D.1 + (rect.C.1 - rect.D.1) / 3 ∧
  rect.E.2 = rect.D.2 →
  -- AB is twice the length of BC
  rect.A.1 - rect.B.1 = 2 * (rect.B.2 - rect.C.2) →
  -- Line BE intersects diagonal AC at point F
  ∃ t : ℝ, rect.F = (1 - t) • rect.A + t • rect.C ∧
          ∃ s : ℝ, rect.F = (1 - s) • rect.B + s • rect.E →
  -- The area of triangle BFE is 18
  triangleArea rect.B rect.F rect.E = 18 →
  -- The area of rectangle ABCD is 108
  (rect.A.1 - rect.D.1) * (rect.A.2 - rect.D.2) = 108 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1656_165689


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1656_165600

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (1 : ℂ) / ((1 + Complex.I)^2 + 1) + Complex.I
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1656_165600


namespace NUMINAMATH_CALUDE_sequence_calculation_l1656_165613

def x (n : ℕ) : ℕ := n^2 + n
def y (n : ℕ) : ℕ := 2 * n^2
def z (n : ℕ) : ℕ := n^3
def t (n : ℕ) : ℕ := 2^n

theorem sequence_calculation :
  (x 1 = 2 ∧ x 2 = 6 ∧ x 3 = 12 ∧ x 4 = 20) ∧
  (y 1 = 2 ∧ y 2 = 8 ∧ y 3 = 18 ∧ y 4 = 32) ∧
  (z 1 = 1 ∧ z 2 = 8 ∧ z 3 = 27 ∧ z 4 = 64) ∧
  (t 1 = 2 ∧ t 2 = 4 ∧ t 3 = 8 ∧ t 4 = 16) := by
  sorry

end NUMINAMATH_CALUDE_sequence_calculation_l1656_165613


namespace NUMINAMATH_CALUDE_last_segment_speed_is_90_l1656_165653

-- Define the problem parameters
def total_distance : ℝ := 150
def total_time : ℝ := 135
def first_segment_time : ℝ := 45
def second_segment_time : ℝ := 45
def last_segment_time : ℝ := 45
def first_segment_speed : ℝ := 50
def second_segment_speed : ℝ := 60

-- Define the theorem
theorem last_segment_speed_is_90 :
  let last_segment_speed := 
    (total_distance - (first_segment_speed * first_segment_time / 60 + 
                       second_segment_speed * second_segment_time / 60)) / 
    (last_segment_time / 60)
  last_segment_speed = 90 := by sorry

end NUMINAMATH_CALUDE_last_segment_speed_is_90_l1656_165653


namespace NUMINAMATH_CALUDE_garbs_are_morks_and_plogs_l1656_165676

-- Define the sets
variable (U : Type) -- Universe set
variable (Mork Plog Snark Garb : Set U)

-- Define the given conditions
variable (h1 : Mork ⊆ Plog)    -- All Morks are Plogs
variable (h2 : Snark ⊆ Mork)   -- All Snarks are Morks
variable (h3 : Garb ⊆ Plog)    -- All Garbs are Plogs
variable (h4 : Garb ⊆ Snark)   -- All Garbs are Snarks

-- Theorem to prove
theorem garbs_are_morks_and_plogs : Garb ⊆ Mork ∩ Plog :=
sorry

end NUMINAMATH_CALUDE_garbs_are_morks_and_plogs_l1656_165676


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l1656_165637

theorem geometric_mean_minimum (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (hgm : z^2 = x*y) : 
  (Real.log z)/(4*Real.log x) + (Real.log z)/(Real.log y) ≥ 9/8 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l1656_165637


namespace NUMINAMATH_CALUDE_petya_friend_count_l1656_165672

/-- Represents the number of friends a student has -/
def FriendCount := Fin 29

/-- Represents a student in the class -/
def Student := Fin 29

/-- The function that maps each student to their friend count -/
def friendCount : Student → FriendCount := sorry

/-- Petya is represented by the last student in the enumeration -/
def petya : Student := ⟨28, sorry⟩

theorem petya_friend_count :
  (∀ (s1 s2 : Student), s1 ≠ s2 → friendCount s1 ≠ friendCount s2) →
  friendCount petya = ⟨14, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_petya_friend_count_l1656_165672


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1656_165601

theorem algebraic_expression_equality (x y : ℝ) : 
  x + 2*y + 1 = 3 → 2*x + 4*y + 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1656_165601


namespace NUMINAMATH_CALUDE_NaCl_moles_formed_l1656_165616

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String

-- Define the reaction equation
structure ReactionEquation where
  reactants : List (ChemicalSpecies × ℕ)
  products : List (ChemicalSpecies × ℕ)

-- Define the available reactants
def available_reactants : List (ChemicalSpecies × ℚ) :=
  [(⟨"NaOH"⟩, 2), (⟨"Cl2"⟩, 1)]

-- Define the balanced equation
def balanced_equation : ReactionEquation :=
  { reactants := [(⟨"NaOH"⟩, 2), (⟨"Cl2"⟩, 1)]
  , products := [(⟨"NaCl"⟩, 2), (⟨"H2O"⟩, 1)] }

-- Define the function to calculate the moles of product formed
def moles_of_product_formed (product : ChemicalSpecies) (eq : ReactionEquation) (reactants : List (ChemicalSpecies × ℚ)) : ℚ :=
  sorry

-- Theorem statement
theorem NaCl_moles_formed :
  moles_of_product_formed ⟨"NaCl"⟩ balanced_equation available_reactants = 2 :=
sorry

end NUMINAMATH_CALUDE_NaCl_moles_formed_l1656_165616


namespace NUMINAMATH_CALUDE_train_speed_l1656_165681

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 240) (h2 : time = 16) :
  length / time = 15 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1656_165681


namespace NUMINAMATH_CALUDE_area_of_EFGH_l1656_165642

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle EFGH formed by three identical smaller rectangles -/
def area_EFGH (small : Rectangle) : ℝ :=
  (2 * small.width) * small.length

theorem area_of_EFGH : 
  ∀ (small : Rectangle),
  small.width = 7 →
  small.length = 3 * small.width →
  area_EFGH small = 294 := by
sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l1656_165642


namespace NUMINAMATH_CALUDE_half_power_inequality_l1656_165629

theorem half_power_inequality (a b : ℝ) (h : a > b) : (1/2 : ℝ)^b > (1/2 : ℝ)^a := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l1656_165629


namespace NUMINAMATH_CALUDE_stones_sent_away_l1656_165636

theorem stones_sent_away (original_stones : ℕ) (stones_left : ℕ) (stones_sent : ℕ) : 
  original_stones = 78 → stones_left = 15 → stones_sent = original_stones - stones_left → stones_sent = 63 := by
  sorry

end NUMINAMATH_CALUDE_stones_sent_away_l1656_165636


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l1656_165699

theorem quadratic_constant_term 
  (x : ℝ) 
  (some_number : ℝ) 
  (h1 : x = 0.5) 
  (h2 : 2 * x^2 + 9 * x + some_number = 0) : 
  some_number = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l1656_165699


namespace NUMINAMATH_CALUDE_manager_salary_is_220000_l1656_165622

/-- Represents the average salary of managers at Plutarch Enterprises -/
def manager_salary : ℝ := 220000

/-- Theorem stating that the average salary of managers at Plutarch Enterprises is $220,000 -/
theorem manager_salary_is_220000 
  (marketer_percent : ℝ) (engineer_percent : ℝ) (sales_percent : ℝ) (manager_percent : ℝ)
  (marketer_salary : ℝ) (engineer_salary : ℝ) (sales_salary : ℝ) (total_avg_salary : ℝ)
  (h1 : marketer_percent = 0.60)
  (h2 : engineer_percent = 0.20)
  (h3 : sales_percent = 0.10)
  (h4 : manager_percent = 0.10)
  (h5 : marketer_salary = 50000)
  (h6 : engineer_salary = 80000)
  (h7 : sales_salary = 70000)
  (h8 : total_avg_salary = 75000)
  (h9 : marketer_percent + engineer_percent + sales_percent + manager_percent = 1) :
  manager_salary = 220000 := by
  sorry

#check manager_salary_is_220000

end NUMINAMATH_CALUDE_manager_salary_is_220000_l1656_165622


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1656_165638

/-- A rectangle with perimeter 40 and area 100 has sides of length 10 -/
theorem rectangle_max_area (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧  -- x and y are positive (implicitly defining a rectangle)
  2 * (x + y) = 40 ∧  -- perimeter is 40
  x * y = 100  -- area is 100
  → x = 10 ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1656_165638


namespace NUMINAMATH_CALUDE_all_points_fit_l1656_165696

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the square
def square : Set Point :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ max (|x| + |y|) (|x - y|) ≤ 2}

-- Define a translation
def translate (p : Point) (v : ℝ × ℝ) : Point :=
  (p.1 + v.1, p.2 + v.2)

-- Define the property that any three points can be translated to fit in the square
def any_three_fit (S : Set Point) :=
  ∀ (p1 p2 p3 : Point), p1 ∈ S → p2 ∈ S → p3 ∈ S →
    ∃ (v : ℝ × ℝ), {translate p1 v, translate p2 v, translate p3 v} ⊆ square

-- Theorem statement
theorem all_points_fit (S : Set Point) (h : any_three_fit S) :
  ∃ (v : ℝ × ℝ), ∀ (p : Point), p ∈ S → translate p v ∈ square :=
sorry

end NUMINAMATH_CALUDE_all_points_fit_l1656_165696


namespace NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1656_165612

/-- Calculates the number of pages read on the fourth day given the reading pattern for a book -/
theorem pages_read_on_fourth_day
  (total_pages : ℕ)
  (day1_pages : ℕ)
  (day2_multiplier : ℕ)
  (day3_additional : ℕ)
  (h1 : total_pages = 354)
  (h2 : day1_pages = 63)
  (h3 : day2_multiplier = 2)
  (h4 : day3_additional = 10) :
  total_pages - (day1_pages + day2_multiplier * day1_pages + (day2_multiplier * day1_pages + day3_additional)) = 29 :=
by sorry

end NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1656_165612


namespace NUMINAMATH_CALUDE_perfect_square_coefficient_l1656_165619

/-- A quadratic trinomial in a and b with coefficient m -/
def quadratic_trinomial (a b : ℝ) (m : ℝ) : ℝ := a^2 + m*a*b + b^2

/-- Definition of a perfect square trinomial -/
def is_perfect_square_trinomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ), ∀ (x y : ℝ), f x y = (g x + y)^2 ∨ f x y = (g x - y)^2

/-- If a^2 + mab + b^2 is a perfect square trinomial, then m = 2 or m = -2 -/
theorem perfect_square_coefficient (m : ℝ) :
  is_perfect_square_trinomial (quadratic_trinomial · · m) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_coefficient_l1656_165619


namespace NUMINAMATH_CALUDE_solution_triples_l1656_165625

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 16*x + 60

theorem solution_triples : 
  ∀ x y z : ℤ, 
    f x = y ∧ f y = z ∧ f z = x → 
      (x = 3 ∧ y = 3 ∧ z = 3) ∨ 
      (x = -4 ∧ y = -4 ∧ z = -4) ∨ 
      (x = 5 ∧ y = 5 ∧ z = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_triples_l1656_165625


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1656_165671

theorem modulus_of_complex_fraction : 
  let z : ℂ := (4 - 3*I) / (2 - I)
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1656_165671


namespace NUMINAMATH_CALUDE_waiter_tables_theorem_l1656_165644

/-- Given the initial number of customers, the number of customers who left,
    and the number of people at each remaining table, calculate the number of
    tables with customers remaining. -/
def remaining_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) : ℕ :=
  (initial_customers - customers_left) / people_per_table

theorem waiter_tables_theorem (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ)
    (h1 : initial_customers ≥ customers_left)
    (h2 : people_per_table > 0)
    (h3 : (initial_customers - customers_left) % people_per_table = 0) :
    remaining_tables initial_customers customers_left people_per_table =
    (initial_customers - customers_left) / people_per_table :=
  by sorry

end NUMINAMATH_CALUDE_waiter_tables_theorem_l1656_165644


namespace NUMINAMATH_CALUDE_parallelogram_area_gt_one_l1656_165665

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four lattice points -/
structure Parallelogram where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint
  v4 : LatticePoint

/-- Checks if a point is inside or on the sides of a parallelogram -/
def isInsideOrOnSides (p : LatticePoint) (pg : Parallelogram) : Prop :=
  sorry

/-- Calculates the area of a parallelogram -/
def area (pg : Parallelogram) : ℝ :=
  sorry

/-- Theorem: The area of a parallelogram with vertices at lattice points 
    and at least one additional lattice point inside or on its sides is greater than 1 -/
theorem parallelogram_area_gt_one (pg : Parallelogram) 
  (h : ∃ p : LatticePoint, p ≠ pg.v1 ∧ p ≠ pg.v2 ∧ p ≠ pg.v3 ∧ p ≠ pg.v4 ∧ isInsideOrOnSides p pg) : 
  area pg > 1 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_gt_one_l1656_165665


namespace NUMINAMATH_CALUDE_cards_eaten_ratio_l1656_165688

/-- Given that Benny bought 4 new baseball cards and has 34 cards left after his dog ate some,
    prove that the ratio of cards eaten by the dog to the total number of cards before
    the dog ate them is (X - 30) / (X + 4), where X is the number of cards Benny had
    before buying the new ones. -/
theorem cards_eaten_ratio (X : ℕ) : 
  let cards_bought : ℕ := 4
  let cards_left : ℕ := 34
  let total_before_eating : ℕ := X + cards_bought
  let cards_eaten : ℕ := total_before_eating - cards_left
  cards_eaten / total_before_eating = (X - 30) / (X + 4) :=
by sorry

end NUMINAMATH_CALUDE_cards_eaten_ratio_l1656_165688


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1656_165668

/-- Given a train with length 280.0224 meters that crosses a post in 25.2 seconds, 
    its speed is 40.0032 km/hr. -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) 
    (h1 : train_length = 280.0224) 
    (h2 : crossing_time = 25.2) : 
  (train_length / 1000) / (crossing_time / 3600) = 40.0032 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1656_165668


namespace NUMINAMATH_CALUDE_transport_probabilities_l1656_165673

/-- Transportation method probabilities -/
structure TransportProb where
  train : ℝ
  ship : ℝ
  car : ℝ
  airplane : ℝ
  sum_to_one : train + ship + car + airplane = 1
  all_nonneg : train ≥ 0 ∧ ship ≥ 0 ∧ car ≥ 0 ∧ airplane ≥ 0

/-- Given probabilities for each transportation method -/
def given_probs : TransportProb where
  train := 0.3
  ship := 0.1
  car := 0.2
  airplane := 0.4
  sum_to_one := by norm_num
  all_nonneg := by norm_num

/-- Theorem stating the probabilities of combined events -/
theorem transport_probabilities (p : TransportProb) :
  p.train + p.airplane = 0.7 ∧ 1 - p.ship = 0.9 := by sorry

end NUMINAMATH_CALUDE_transport_probabilities_l1656_165673


namespace NUMINAMATH_CALUDE_message_decoding_l1656_165634

-- Define the Russian alphabet
def RussianAlphabet : List Char := ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

-- Define the digit groups
def DigitGroups : List (List Nat) := [[1], [3, 7, 8], [0, 4, 5, 6]]

-- Define the encoding function
def encode (groups : List (List Nat)) (alphabet : List Char) : Nat → Option Char := sorry

-- Define the decoding function
def decode (groups : List (List Nat)) (alphabet : List Char) : List Nat → String := sorry

-- Theorem statement
theorem message_decoding :
  decode DigitGroups RussianAlphabet [8, 7, 3, 1, 4, 6, 5, 0, 7, 3, 8, 1] = "НАУКА" ∧
  encode DigitGroups RussianAlphabet 8 = some 'Н' ∧
  encode DigitGroups RussianAlphabet 7 = some 'А' ∧
  encode DigitGroups RussianAlphabet 3 = some 'У' ∧
  encode DigitGroups RussianAlphabet 1 = some 'К' ∧
  encode DigitGroups RussianAlphabet 4 = some 'А' ∧
  encode DigitGroups RussianAlphabet 6 = none ∧
  encode DigitGroups RussianAlphabet 5 = some 'К' ∧
  encode DigitGroups RussianAlphabet 0 = none := by
  sorry

end NUMINAMATH_CALUDE_message_decoding_l1656_165634


namespace NUMINAMATH_CALUDE_tim_seashells_l1656_165685

/-- The number of seashells Tim found initially -/
def initial_seashells : ℕ := 679

/-- The number of seashells Tim gave to Sara -/
def seashells_given : ℕ := 172

/-- The number of seashells Tim has after giving some to Sara -/
def remaining_seashells : ℕ := initial_seashells - seashells_given

theorem tim_seashells : remaining_seashells = 507 := by sorry

end NUMINAMATH_CALUDE_tim_seashells_l1656_165685


namespace NUMINAMATH_CALUDE_cos_tan_identity_l1656_165660

theorem cos_tan_identity : 
  (Real.cos (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / Real.cos (50 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_tan_identity_l1656_165660


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1656_165687

theorem trigonometric_equation_solution (z : ℝ) : 
  2 * (Real.cos z) * (Real.sin (3 * Real.pi / 2 - z))^3 - 
  5 * (Real.sin z)^2 * (Real.cos z)^2 + 
  (Real.sin z) * (Real.cos (3 * Real.pi / 2 + z))^3 = 
  Real.cos (2 * z) → 
  ∃ (n : ℤ), z = Real.pi / 3 * (3 * ↑n + 1) ∨ z = Real.pi / 3 * (3 * ↑n - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1656_165687


namespace NUMINAMATH_CALUDE_rhombus_area_l1656_165692

/-- The area of a rhombus with side length 4 and an interior angle of 45 degrees is 8√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1656_165692


namespace NUMINAMATH_CALUDE_perfect_square_base7_b_values_l1656_165610

/-- A number in base 7 of the form ac4b where a ≠ 0 and 0 ≤ b < 7 -/
structure Base7Number where
  a : ℕ
  c : ℕ
  b : ℕ
  a_nonzero : a ≠ 0
  b_range : b < 7

/-- Convert a Base7Number to its decimal representation -/
def to_decimal (n : Base7Number) : ℕ :=
  343 * n.a + 49 * n.c + 28 + n.b

/-- Predicate to check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_square_base7_b_values (n : Base7Number) :
  is_perfect_square (to_decimal n) → n.b = 0 ∨ n.b = 1 ∨ n.b = 4 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_base7_b_values_l1656_165610
