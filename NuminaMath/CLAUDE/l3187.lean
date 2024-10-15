import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_inequality_l3187_318706

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) : Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3187_318706


namespace NUMINAMATH_CALUDE_marissa_initial_ribbon_l3187_318795

/-- The amount of ribbon used per box in feet -/
def ribbon_per_box : ℝ := 0.7

/-- The number of boxes Marissa tied -/
def num_boxes : ℕ := 5

/-- The amount of ribbon left after tying all boxes in feet -/
def ribbon_left : ℝ := 1

/-- The initial amount of ribbon Marissa had in feet -/
def initial_ribbon : ℝ := ribbon_per_box * num_boxes + ribbon_left

theorem marissa_initial_ribbon :
  initial_ribbon = 4.5 := by sorry

end NUMINAMATH_CALUDE_marissa_initial_ribbon_l3187_318795


namespace NUMINAMATH_CALUDE_alex_shirts_l3187_318724

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3) 
  (h2 : ben = joe + 8) 
  (h3 : ben = 15) : 
  alex = 4 := by
sorry

end NUMINAMATH_CALUDE_alex_shirts_l3187_318724


namespace NUMINAMATH_CALUDE_sector_area_l3187_318745

theorem sector_area (θ : Real) (arc_length : Real) (area : Real) : 
  θ = π / 3 →  -- 60° in radians
  arc_length = 2 * π → 
  area = 6 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3187_318745


namespace NUMINAMATH_CALUDE_power_of_two_plus_two_eq_rational_square_l3187_318722

theorem power_of_two_plus_two_eq_rational_square (r : ℚ) :
  (∃ z : ℤ, 2^z + 2 = r^2) ↔ (r = 2 ∨ r = -2 ∨ r = 3/2 ∨ r = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_two_eq_rational_square_l3187_318722


namespace NUMINAMATH_CALUDE_equation_solution_range_l3187_318774

theorem equation_solution_range (x m : ℝ) : 
  ((2 * x + m) / (x - 1) = 1) → 
  (x > 0) → 
  (x ≠ 1) → 
  (m < -1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3187_318774


namespace NUMINAMATH_CALUDE_cat_cafe_theorem_l3187_318726

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := meow_cats + paw_cats

theorem cat_cafe_theorem : total_cats = 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_cafe_theorem_l3187_318726


namespace NUMINAMATH_CALUDE_unique_prime_base_l3187_318753

theorem unique_prime_base : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^4 + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_base_l3187_318753


namespace NUMINAMATH_CALUDE_ratio_proof_l3187_318754

/-- Given two positive integers with specific properties, prove their ratio -/
theorem ratio_proof (A B : ℕ+) (h1 : A = 48) (h2 : Nat.lcm A B = 432) :
  (A : ℚ) / B = 1 / (4.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l3187_318754


namespace NUMINAMATH_CALUDE_exists_special_configuration_l3187_318797

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set (ℝ × ℝ))
  intersection_points : Set (ℝ × ℝ)

/-- The property that any 9 lines intersect at all points of intersection -/
def any_nine_cover_all (config : LineConfiguration) : Prop :=
  ∀ (subset : Finset (Set (ℝ × ℝ))), subset ⊆ config.lines → subset.card = 9 →
    (⋂ l ∈ subset, l) = config.intersection_points

/-- The property that any 8 lines do not intersect at all points of intersection -/
def any_eight_miss_some (config : LineConfiguration) : Prop :=
  ∀ (subset : Finset (Set (ℝ × ℝ))), subset ⊆ config.lines → subset.card = 8 →
    (⋂ l ∈ subset, l) ≠ config.intersection_points

/-- The main theorem stating the existence of a configuration satisfying both properties -/
theorem exists_special_configuration :
  ∃ (config : LineConfiguration), config.lines.card = 10 ∧
    any_nine_cover_all config ∧ any_eight_miss_some config := by
  sorry

end NUMINAMATH_CALUDE_exists_special_configuration_l3187_318797


namespace NUMINAMATH_CALUDE_altons_weekly_profit_l3187_318711

/-- Calculates the weekly profit for a business owner given daily earnings and weekly rent. -/
def weekly_profit (daily_earnings : ℕ) (weekly_rent : ℕ) : ℕ :=
  daily_earnings * 7 - weekly_rent

/-- Theorem stating that given specific daily earnings and weekly rent, the weekly profit is 36. -/
theorem altons_weekly_profit :
  weekly_profit 8 20 = 36 := by
  sorry

end NUMINAMATH_CALUDE_altons_weekly_profit_l3187_318711


namespace NUMINAMATH_CALUDE_count_power_functions_l3187_318703

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = x^k

def f₁ (x : ℝ) : ℝ := x^3
def f₂ (x : ℝ) : ℝ := 4*x^2
def f₃ (x : ℝ) : ℝ := x^5 + 1
def f₄ (x : ℝ) : ℝ := (x-1)^2
def f₅ (x : ℝ) : ℝ := x

theorem count_power_functions : 
  (is_power_function f₁ ∧ ¬is_power_function f₂ ∧ ¬is_power_function f₃ ∧ 
   ¬is_power_function f₄ ∧ is_power_function f₅) :=
by sorry

end NUMINAMATH_CALUDE_count_power_functions_l3187_318703


namespace NUMINAMATH_CALUDE_right_angled_triangle_k_values_l3187_318704

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def is_right_angled (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem right_angled_triangle_k_values :
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ 
  (∀ k : ℝ, (is_right_angled AB (AC k) ∨ 
             is_right_angled AB (AC k - AB) ∨ 
             is_right_angled (AC k - AB) (AC k)) 
  ↔ (k = k₁ ∨ k = k₂)) :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_k_values_l3187_318704


namespace NUMINAMATH_CALUDE_m_range_l3187_318781

/-- Proposition p: m is a real number and m + 1 ≤ 0 -/
def p (m : ℝ) : Prop := m + 1 ≤ 0

/-- Proposition q: For all real x, x² + mx + 1 > 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

/-- The range of m satisfying the given conditions -/
theorem m_range (m : ℝ) : 
  (p m ∧ q m → False) →  -- p ∧ q is false
  (p m ∨ q m) →          -- p ∨ q is true
  (m ≤ -2 ∨ (-1 < m ∧ m < 2)) := by
sorry

end NUMINAMATH_CALUDE_m_range_l3187_318781


namespace NUMINAMATH_CALUDE_percentage_equality_l3187_318794

theorem percentage_equality (x : ℝ) (h : x > 0) :
  ∃ p : ℝ, p / 100 * (x + 20) = 0.3 * (0.6 * x) ∧ p = 1800 * x / (x + 20) := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3187_318794


namespace NUMINAMATH_CALUDE_student_mistake_difference_l3187_318747

theorem student_mistake_difference : (5/6 : ℚ) * 576 - (5/16 : ℚ) * 576 = 300 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l3187_318747


namespace NUMINAMATH_CALUDE_sams_cows_l3187_318760

theorem sams_cows (C : ℕ) : 
  (C / 2 + 5 = C - 4) → C = 18 := by
  sorry

end NUMINAMATH_CALUDE_sams_cows_l3187_318760


namespace NUMINAMATH_CALUDE_team_savings_is_36_dollars_l3187_318785

-- Define the prices and team size
def regular_shirt_price : ℝ := 7.50
def regular_pants_price : ℝ := 15.00
def regular_socks_price : ℝ := 4.50
def discounted_shirt_price : ℝ := 6.75
def discounted_pants_price : ℝ := 13.50
def discounted_socks_price : ℝ := 3.75
def team_size : ℕ := 12

-- Define the total savings function
def total_savings : ℝ :=
  let regular_uniform_price := regular_shirt_price + regular_pants_price + regular_socks_price
  let discounted_uniform_price := discounted_shirt_price + discounted_pants_price + discounted_socks_price
  let savings_per_uniform := regular_uniform_price - discounted_uniform_price
  savings_per_uniform * team_size

-- Theorem statement
theorem team_savings_is_36_dollars : total_savings = 36 := by
  sorry

end NUMINAMATH_CALUDE_team_savings_is_36_dollars_l3187_318785


namespace NUMINAMATH_CALUDE_roses_remaining_l3187_318730

/-- Given 3 dozen roses, prove that after giving half away and removing one-third of the remaining flowers, 12 flowers are left. -/
theorem roses_remaining (initial_roses : ℕ) (dozen : ℕ) (half : ℕ → ℕ) (third : ℕ → ℕ) : 
  initial_roses = 3 * dozen → 
  dozen = 12 →
  half n = n / 2 →
  third n = n / 3 →
  third (initial_roses - half initial_roses) = 12 := by
sorry

end NUMINAMATH_CALUDE_roses_remaining_l3187_318730


namespace NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_rate_l3187_318782

/-- Calculates the interest rate of an additional investment given initial investment conditions and total annual income. -/
theorem additional_investment_rate (initial_investment : ℝ) (initial_rate : ℝ) (total_rate : ℝ) (total_income : ℝ) : ℝ :=
  let additional_investment := (total_income - initial_investment * total_rate) / (total_rate - initial_rate)
  let additional_income := total_income - initial_investment * initial_rate
  additional_income / additional_investment

/-- Proves that the interest rate of the additional investment is approximately 6.13% given the specified conditions. -/
theorem barbata_investment_rate : 
  let initial_investment : ℝ := 2200
  let initial_rate : ℝ := 0.05
  let total_rate : ℝ := 0.06
  let total_income : ℝ := 1099.9999999999998
  abs (additional_investment_rate initial_investment initial_rate total_rate total_income - 0.0613) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_rate_l3187_318782


namespace NUMINAMATH_CALUDE_problem_statement_l3187_318784

theorem problem_statement : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3187_318784


namespace NUMINAMATH_CALUDE_volunteers_from_third_grade_l3187_318743

/-- Calculates the number of volunteers to be recruited from a specific grade --/
def volunteersFromGrade (totalStudents : ℕ) (gradeStudents : ℕ) (totalVolunteers : ℕ) : ℕ :=
  (gradeStudents * totalVolunteers) / totalStudents

/-- Represents the problem of calculating volunteers from the third grade --/
theorem volunteers_from_third_grade 
  (totalStudents : ℕ) 
  (firstGradeStudents : ℕ) 
  (secondGradeStudents : ℕ) 
  (thirdGradeStudents : ℕ) 
  (totalVolunteers : ℕ) :
  totalStudents = firstGradeStudents + secondGradeStudents + thirdGradeStudents →
  totalStudents = 2040 →
  firstGradeStudents = 680 →
  secondGradeStudents = 850 →
  thirdGradeStudents = 510 →
  totalVolunteers = 12 →
  volunteersFromGrade totalStudents thirdGradeStudents totalVolunteers = 3 :=
by sorry

end NUMINAMATH_CALUDE_volunteers_from_third_grade_l3187_318743


namespace NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l3187_318750

def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem two_ones_in_twelve_dice :
  probability_two_ones 12 2 (1/6) = (66 * 5^10 : ℚ) / (36 * 6^10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l3187_318750


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l3187_318728

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

/-- The equation of a parabola in general form -/
def parabola_equation (p : Parabola) (x y : ℝ) : Prop :=
  x^2 - 2*x*y + y^2 - 12*x - 16*y + 78 = 0

theorem parabola_equation_correct (p : Parabola) :
  p.focus = Point.mk 4 5 →
  p.directrix = Line.mk 1 1 (-2) →
  ∀ x y : ℝ, (x^2 - 2*x*y + y^2 - 12*x - 16*y + 78 = 0) ↔
    (((x - 4)^2 + (y - 5)^2) = ((x + y - 2)^2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l3187_318728


namespace NUMINAMATH_CALUDE_eugene_pencils_count_l3187_318759

/-- The total number of pencils Eugene has after receiving additional pencils -/
def total_pencils (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Eugene's total pencils equals the sum of his initial pencils and additional pencils -/
theorem eugene_pencils_count : 
  total_pencils 51 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_count_l3187_318759


namespace NUMINAMATH_CALUDE_shaniqua_haircuts_l3187_318786

/-- Represents the pricing and earnings of a hairstylist --/
structure HairstylistEarnings where
  haircut_price : ℕ
  style_price : ℕ
  total_earnings : ℕ
  num_styles : ℕ

/-- Calculates the number of haircuts given the hairstylist's earnings information --/
def calculate_haircuts (e : HairstylistEarnings) : ℕ :=
  (e.total_earnings - e.style_price * e.num_styles) / e.haircut_price

/-- Theorem stating that given Shaniqua's earnings information, she gave 8 haircuts --/
theorem shaniqua_haircuts :
  let e : HairstylistEarnings := {
    haircut_price := 12,
    style_price := 25,
    total_earnings := 221,
    num_styles := 5
  }
  calculate_haircuts e = 8 := by sorry

end NUMINAMATH_CALUDE_shaniqua_haircuts_l3187_318786


namespace NUMINAMATH_CALUDE_fish_tank_capacity_l3187_318727

/-- The capacity of a fish tank given pouring rate, duration, and remaining volume --/
theorem fish_tank_capacity
  (pour_rate : ℚ)  -- Pouring rate in gallons per second
  (pour_duration : ℕ)  -- Pouring duration in minutes
  (remaining_volume : ℕ)  -- Remaining volume to fill the tank in gallons
  (h1 : pour_rate = 1 / 20)  -- 1 gallon every 20 seconds
  (h2 : pour_duration = 6)  -- Poured for 6 minutes
  (h3 : remaining_volume = 32)  -- 32 more gallons needed
  : ℕ :=
by
  sorry

#check fish_tank_capacity

end NUMINAMATH_CALUDE_fish_tank_capacity_l3187_318727


namespace NUMINAMATH_CALUDE_complement_union_intersection_equivalence_l3187_318716

-- Define the sets U, M, and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_union_intersection_equivalence :
  ∀ x : ℝ, x ∈ (U \ M) ∪ (M ∩ N) ↔ x < 5 := by sorry

end NUMINAMATH_CALUDE_complement_union_intersection_equivalence_l3187_318716


namespace NUMINAMATH_CALUDE_polygon_with_special_angle_property_l3187_318715

theorem polygon_with_special_angle_property (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_special_angle_property_l3187_318715


namespace NUMINAMATH_CALUDE_horner_method_V₂_l3187_318746

-- Define the polynomial coefficients
def a₄ : ℤ := 3
def a₃ : ℤ := 5
def a₂ : ℤ := 6
def a₁ : ℤ := 79
def a₀ : ℤ := -8

-- Define the x value
def x : ℤ := -4

-- Define Horner's method steps
def V₀ : ℤ := a₄
def V₁ : ℤ := x * V₀ + a₃
def V₂ : ℤ := x * V₁ + a₂

-- Theorem statement
theorem horner_method_V₂ : V₂ = 34 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_V₂_l3187_318746


namespace NUMINAMATH_CALUDE_money_distribution_l3187_318707

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 310) :
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3187_318707


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3187_318769

theorem geometric_sequence_first_term
  (a : ℝ)  -- first term of the sequence
  (r : ℝ)  -- common ratio
  (h1 : a * r^2 = 27)  -- third term is 27
  (h2 : a * r^3 = 81)  -- fourth term is 81
  : a = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3187_318769


namespace NUMINAMATH_CALUDE_min_value_tangent_sum_l3187_318738

theorem min_value_tangent_sum (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π / 2) :
  3 * Real.tan B * Real.tan C + 2 * Real.tan A * Real.tan C + Real.tan A * Real.tan B ≥ 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_tangent_sum_l3187_318738


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l3187_318714

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of ties available. -/
def num_ties : ℕ := 7

/-- The number of specific shirt-tie pairs that cannot be worn together. -/
def num_restricted_pairs : ℕ := 3

/-- The total number of possible shirt-tie combinations. -/
def total_combinations : ℕ := num_shirts * num_ties

/-- The number of allowable shirt-tie combinations. -/
def allowable_combinations : ℕ := total_combinations - num_restricted_pairs

theorem shirt_tie_combinations : allowable_combinations = 53 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l3187_318714


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l3187_318770

theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (like_apple_pie : ℕ) 
  (like_chocolate_cake : ℕ) 
  (like_neither : ℕ) 
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 25)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 15) :
  like_apple_pie + like_chocolate_cake - (total_students - like_neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l3187_318770


namespace NUMINAMATH_CALUDE_circle_has_zero_radius_l3187_318702

/-- The equation of a circle with radius 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 10*x + y^2 - 4*y + 29 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-5, 2)

theorem circle_has_zero_radius :
  ∀ x y : ℝ, circle_equation x y ↔ (x, y) = circle_center :=
by sorry

end NUMINAMATH_CALUDE_circle_has_zero_radius_l3187_318702


namespace NUMINAMATH_CALUDE_systematic_sampling_l3187_318737

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_population : ℕ)
  (sample_size : ℕ)
  (first_drawn : ℕ)
  (interval_start : ℕ)
  (interval_end : ℕ)
  (h1 : total_population = 960)
  (h2 : sample_size = 32)
  (h3 : first_drawn = 29)
  (h4 : interval_start = 200)
  (h5 : interval_end = 480) :
  (Finset.filter (fun n => interval_start ≤ (first_drawn + (total_population / sample_size) * (n - 1)) ∧
                           (first_drawn + (total_population / sample_size) * (n - 1)) ≤ interval_end)
                 (Finset.range sample_size)).card = 10 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_l3187_318737


namespace NUMINAMATH_CALUDE_at_least_two_sums_divisible_by_p_l3187_318731

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem at_least_two_sums_divisible_by_p (p a b c d : ℕ) (hp : p > 2) (hprime : Nat.Prime p)
  (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) (hc : ¬ p ∣ c) (hd : ¬ p ∣ d)
  (h : ∀ r : ℕ, ¬ p ∣ r → 
    fractional_part (r * a / p) + fractional_part (r * b / p) + 
    fractional_part (r * c / p) + fractional_part (r * d / p) = 2) :
  (∃ (x y : ℕ × ℕ), x ≠ y ∧ 
    (x ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) ∧
    (y ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) ∧
    p ∣ (x.1 + x.2) ∧ p ∣ (y.1 + y.2)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_sums_divisible_by_p_l3187_318731


namespace NUMINAMATH_CALUDE_shaded_area_is_24_5_l3187_318749

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  bottomLeft : Point
  baseLength : ℝ

/-- Calculates the area of the shaded region -/
def shadedArea (square : Square) (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_24_5 (square : Square) (triangle : IsoscelesTriangle) :
  square.bottomLeft = Point.mk 0 0 →
  square.sideLength = 7 →
  triangle.bottomLeft = Point.mk 7 0 →
  triangle.baseLength = 7 →
  shadedArea square triangle = 24.5 :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_24_5_l3187_318749


namespace NUMINAMATH_CALUDE_point_coordinates_l3187_318762

/-- A point in the first quadrant with given distances to axes -/
structure FirstQuadrantPoint where
  m : ℝ
  n : ℝ
  first_quadrant : m > 0 ∧ n > 0
  x_axis_distance : n = 5
  y_axis_distance : m = 3

/-- Theorem: The coordinates of the point are (3,5) -/
theorem point_coordinates (P : FirstQuadrantPoint) : P.m = 3 ∧ P.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3187_318762


namespace NUMINAMATH_CALUDE_pizza_toppings_l3187_318725

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ)
  (h1 : total_slices = 16)
  (h2 : cheese_slices = 10)
  (h3 : pepperoni_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range cheese_slices ∨ slice ∈ Finset.range pepperoni_slices)) :
  cheese_slices + pepperoni_slices - total_slices = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3187_318725


namespace NUMINAMATH_CALUDE_greg_travel_distance_l3187_318773

/-- Greg's travel problem -/
theorem greg_travel_distance :
  let distance_to_market : ℝ := 30
  let time_from_market : ℝ := 30 / 60  -- 30 minutes converted to hours
  let speed_from_market : ℝ := 20
  let distance_from_market : ℝ := time_from_market * speed_from_market
  distance_to_market + distance_from_market = 40 := by
  sorry

end NUMINAMATH_CALUDE_greg_travel_distance_l3187_318773


namespace NUMINAMATH_CALUDE_evaluate_expression_l3187_318739

theorem evaluate_expression : 6 - 9 * (10 - 4^2) * 5 = 276 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3187_318739


namespace NUMINAMATH_CALUDE_baf_compound_composition_l3187_318779

/-- Represents the molecular structure of a compound containing Barium and Fluorine --/
structure BaFCompound where
  ba_count : ℕ
  f_count : ℕ
  molecular_weight : ℝ

/-- Atomic weights of elements --/
def atomic_weight : String → ℝ
  | "Ba" => 137.33
  | "F" => 18.998
  | _ => 0

/-- Calculates the molecular weight of a BaFCompound --/
def calculate_weight (c : BaFCompound) : ℝ :=
  c.ba_count * atomic_weight "Ba" + c.f_count * atomic_weight "F"

/-- Theorem stating that a compound with 2 Fluorine atoms and molecular weight 175 contains 1 Barium atom --/
theorem baf_compound_composition :
  ∃ (c : BaFCompound), c.f_count = 2 ∧ c.molecular_weight = 175 ∧ c.ba_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_baf_compound_composition_l3187_318779


namespace NUMINAMATH_CALUDE_total_insects_on_leaves_l3187_318788

/-- The total number of insects on leaves with given conditions -/
def total_insects (
  num_leaves : ℕ
  ) (ladybugs_per_leaf : ℕ
  ) (ants_per_leaf : ℕ
  ) (caterpillars_per_third_leaf : ℕ
  ) : ℕ :=
  (num_leaves * ladybugs_per_leaf) +
  (num_leaves * ants_per_leaf) +
  (num_leaves / 3 * caterpillars_per_third_leaf)

/-- Theorem stating the total number of insects under given conditions -/
theorem total_insects_on_leaves :
  total_insects 84 139 97 53 = 21308 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_on_leaves_l3187_318788


namespace NUMINAMATH_CALUDE_linear_equation_root_range_l3187_318789

theorem linear_equation_root_range (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) ↔ (k < 1 ∨ k > 3) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_root_range_l3187_318789


namespace NUMINAMATH_CALUDE_jacob_already_twice_as_old_l3187_318741

/-- Proves that Jacob is already twice as old as his brother -/
theorem jacob_already_twice_as_old (jacob_age : ℕ) (brother_age : ℕ) 
  (h1 : jacob_age = 18) 
  (h2 : jacob_age = 2 * brother_age) : 
  jacob_age = 2 * brother_age := by
  sorry

end NUMINAMATH_CALUDE_jacob_already_twice_as_old_l3187_318741


namespace NUMINAMATH_CALUDE_pentagon_area_half_decagon_area_l3187_318718

/-- The area of a pentagon formed by connecting every second vertex of a regular decagon
    is half the area of the decagon. -/
theorem pentagon_area_half_decagon_area (n : ℝ) (h : n > 0) :
  ∃ (m : ℝ), m > 0 ∧ m / n = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_half_decagon_area_l3187_318718


namespace NUMINAMATH_CALUDE_square_sum_power_of_two_l3187_318768

theorem square_sum_power_of_two (n : ℕ) : 
  (∃ m : ℕ, 2^6 + 2^9 + 2^n = m^2) → n = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_power_of_two_l3187_318768


namespace NUMINAMATH_CALUDE_train_passengers_l3187_318761

theorem train_passengers (initial : ℕ) 
  (h1 : initial + 17 - 29 - 27 + 35 = 116) : initial = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l3187_318761


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l3187_318748

theorem square_reciprocal_sum (p : ℝ) (h : p + 1/p = 10) :
  p^2 + 1/p^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l3187_318748


namespace NUMINAMATH_CALUDE_f_value_l3187_318756

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem f_value (α : Real) 
  (h1 : Real.cos (α + 3 * Real.pi / 2) = 1 / 5)
  (h2 : 0 < α - Real.pi / 2 ∧ α - Real.pi / 2 < Real.pi / 2) : 
  f α = 2 * Real.sqrt 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_f_value_l3187_318756


namespace NUMINAMATH_CALUDE_only_two_is_sum_of_squares_among_repeating_twos_l3187_318791

def is_repeating_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * (10^k - 1) / 9

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

theorem only_two_is_sum_of_squares_among_repeating_twos :
  ∀ n : ℕ, is_repeating_two n → (is_sum_of_two_squares n ↔ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_only_two_is_sum_of_squares_among_repeating_twos_l3187_318791


namespace NUMINAMATH_CALUDE_system_solution_l3187_318742

theorem system_solution :
  let f (x y : ℝ) := y^2 - (x^3 - 3*x^2 + 2*x)
  let g (x y : ℝ) := x^2 - (y^3 - 3*y^2 + 2*y)
  ∀ x y : ℝ, f x y = 0 ∧ g x y = 0 ↔
    (x = 0 ∧ y = 0) ∨
    (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
    (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3187_318742


namespace NUMINAMATH_CALUDE_randy_initial_biscuits_l3187_318700

/-- The number of biscuits Randy's father gave him -/
def father_gift : ℕ := 13

/-- The number of biscuits Randy's mother gave him -/
def mother_gift : ℕ := 15

/-- The number of biscuits Randy's brother ate -/
def brother_ate : ℕ := 20

/-- The number of biscuits Randy is left with -/
def remaining_biscuits : ℕ := 40

/-- Randy's initial number of biscuits -/
def initial_biscuits : ℕ := 32

theorem randy_initial_biscuits :
  initial_biscuits + father_gift + mother_gift - brother_ate = remaining_biscuits :=
by sorry

end NUMINAMATH_CALUDE_randy_initial_biscuits_l3187_318700


namespace NUMINAMATH_CALUDE_complex_subtraction_l3187_318757

theorem complex_subtraction (z₁ z₂ : ℂ) (h1 : z₁ = 7 - 6*I) (h2 : z₂ = 4 - 7*I) :
  z₁ - z₂ = 3 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3187_318757


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l3187_318717

/-- The actual distance traveled by a person, given two walking speeds and additional distance information. -/
theorem actual_distance_traveled (slow_speed fast_speed : ℝ) (additional_distance : ℝ) 
  (h1 : slow_speed = 5)
  (h2 : fast_speed = 10)
  (h3 : additional_distance = 20)
  (h4 : ∀ d : ℝ, d / slow_speed = (d + additional_distance) / fast_speed) :
  ∃ d : ℝ, d = 20 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l3187_318717


namespace NUMINAMATH_CALUDE_trapezoid_area_l3187_318778

/-- The area of a trapezoid given the areas of the triangles adjacent to its bases -/
theorem trapezoid_area (K₁ K₂ : ℝ) (h₁ : K₁ > 0) (h₂ : K₂ > 0) :
  ∃ (A : ℝ), A = K₁ + K₂ + 2 * Real.sqrt (K₁ * K₂) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3187_318778


namespace NUMINAMATH_CALUDE_carries_money_l3187_318783

/-- The amount of money Carrie spent on the sweater -/
def sweater_cost : ℕ := 24

/-- The amount of money Carrie spent on the T-shirt -/
def tshirt_cost : ℕ := 6

/-- The amount of money Carrie spent on the shoes -/
def shoes_cost : ℕ := 11

/-- The amount of money Carrie has left after shopping -/
def money_left : ℕ := 50

/-- The total amount of money Carrie's mom gave her -/
def total_money : ℕ := sweater_cost + tshirt_cost + shoes_cost + money_left

theorem carries_money : total_money = 91 := by sorry

end NUMINAMATH_CALUDE_carries_money_l3187_318783


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3187_318793

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    p > 30 ∧ q > 30 ∧ 
    p ≠ q ∧
    ∀ r s : ℕ, Prime r → Prime s → r > 30 → s > 30 → r ≠ s → p * q ≤ r * s :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3187_318793


namespace NUMINAMATH_CALUDE_larger_number_proof_l3187_318713

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 4 * x + 15) : 
  y = 1815 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3187_318713


namespace NUMINAMATH_CALUDE_expression_evaluation_l3187_318799

theorem expression_evaluation (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5*y) * (-x - 5*y) - (-x + 5*y)^2 = -5.5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3187_318799


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3187_318729

theorem arithmetic_geometric_sequence_problem :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    b - a = c - b ∧
    a + b + c = 15 ∧
    (a + 2) * (c + 13) = (b + 5)^2 ∧
    a = 3 ∧ b = 5 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3187_318729


namespace NUMINAMATH_CALUDE_ellipse_properties_l3187_318765

-- Define the ellipse G
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / 4 = 1

-- Define the foci
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (a : ℝ) (M : ℝ × ℝ) : Prop := ellipse a M.1 M.2

-- Define perpendicularity condition
def perpendicular (M F₁ F₂ : ℝ × ℝ) : Prop :=
  (M.1 - F₂.1) * (F₂.1 - F₁.1) + (M.2 - F₂.2) * (F₂.2 - F₁.2) = 0

-- Define the distance difference condition
def distance_diff (M F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) -
  Real.sqrt ((M.2 - F₂.1)^2 + (M.2 - F₂.2)^2) = 4*a/3

-- Define the theorem
theorem ellipse_properties (a c : ℝ) (M : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_on_ellipse : point_on_ellipse a M)
  (h_perp : perpendicular M (left_focus c) (right_focus c))
  (h_dist : distance_diff M (left_focus c) (right_focus c) a) :
  (∀ x y, ellipse a x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipse a A.1 A.2 ∧ 
    ellipse a B.1 B.2 ∧
    B.2 - A.2 = B.1 - A.1 ∧ 
    (let P : ℝ × ℝ := (-3, 2);
     let S := (B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1);
     S * S / 2 = 9/2)) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3187_318765


namespace NUMINAMATH_CALUDE_football_practice_hours_l3187_318740

/-- Given a football team's practice schedule and a week with one missed day,
    calculate the total practice hours for the week. -/
theorem football_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 5 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 30 := by
sorry

end NUMINAMATH_CALUDE_football_practice_hours_l3187_318740


namespace NUMINAMATH_CALUDE_computer_price_increase_l3187_318721

theorem computer_price_increase (c : ℝ) (h : 2 * c = 540) : 
  (351 - c) / c * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3187_318721


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3187_318796

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3187_318796


namespace NUMINAMATH_CALUDE_at_least_one_not_divisible_l3187_318734

theorem at_least_one_not_divisible (a b c d : ℕ) (h : a * d - b * c > 1) :
  ¬(a * d - b * c ∣ a) ∨ ¬(a * d - b * c ∣ b) ∨ ¬(a * d - b * c ∣ c) ∨ ¬(a * d - b * c ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_divisible_l3187_318734


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l3187_318732

theorem product_of_sum_and_sum_of_squares (m n : ℝ) 
  (h1 : m + n = 3) 
  (h2 : m^2 + n^2 = 3) : 
  m * n = 3 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l3187_318732


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l3187_318710

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + a - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l3187_318710


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3187_318733

theorem arithmetic_calculation : 12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3187_318733


namespace NUMINAMATH_CALUDE_oplus_neg_two_three_l3187_318701

def oplus (a b : ℝ) : ℝ := a * (a - b) + 1

theorem oplus_neg_two_three : oplus (-2) 3 = 11 := by sorry

end NUMINAMATH_CALUDE_oplus_neg_two_three_l3187_318701


namespace NUMINAMATH_CALUDE_equal_interest_rate_equal_interest_l3187_318780

/-- The rate at which a principal of 200 invested for 12 years produces the same
    interest as 400 invested for 5 years at 12% annual interest rate -/
theorem equal_interest_rate : ℝ :=
  let principal1 : ℝ := 200
  let time1 : ℝ := 12
  let principal2 : ℝ := 400
  let time2 : ℝ := 5
  let rate2 : ℝ := 12 / 100
  let interest2 : ℝ := principal2 * rate2 * time2
  10 / 100

/-- Proof that the calculated rate produces equal interest -/
theorem equal_interest (rate : ℝ) (h : rate = equal_interest_rate) :
  200 * rate * 12 = 400 * (12 / 100) * 5 := by
  sorry

#check equal_interest
#check equal_interest_rate

end NUMINAMATH_CALUDE_equal_interest_rate_equal_interest_l3187_318780


namespace NUMINAMATH_CALUDE_certain_number_problem_l3187_318776

theorem certain_number_problem : 
  ∃ x : ℝ, (3500 - (x / 20.50) = 3451.2195121951218) ∧ (x = 1000) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3187_318776


namespace NUMINAMATH_CALUDE_f_neg_one_value_l3187_318736

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x^2 + x

theorem f_neg_one_value (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f_nonneg f) : f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_value_l3187_318736


namespace NUMINAMATH_CALUDE_original_class_size_l3187_318777

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 →
  new_students = 10 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ x : ℕ, x * original_avg + new_students * new_avg = (x + new_students) * (original_avg - avg_decrease) ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_original_class_size_l3187_318777


namespace NUMINAMATH_CALUDE_proportional_function_slope_l3187_318798

/-- A proportional function passing through the point (3, -5) has a slope of -5/3 -/
theorem proportional_function_slope (k : ℝ) (h1 : k ≠ 0) 
  (h2 : -5 = k * 3) : k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_slope_l3187_318798


namespace NUMINAMATH_CALUDE_bucket_capacity_change_l3187_318790

theorem bucket_capacity_change (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 200 →
  capacity_ratio = 4/5 →
  (original_buckets : ℚ) / capacity_ratio = 250 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_change_l3187_318790


namespace NUMINAMATH_CALUDE_circle_s_radius_l3187_318744

/-- Triangle XYZ with given side lengths -/
structure Triangle :=
  (xy : ℝ)
  (xz : ℝ)
  (yz : ℝ)

/-- Circle with given radius -/
structure Circle :=
  (radius : ℝ)

/-- Theorem stating the radius of circle S in the given triangle configuration -/
theorem circle_s_radius (t : Triangle) (r : Circle) (s : Circle) :
  t.xy = 120 →
  t.xz = 120 →
  t.yz = 80 →
  r.radius = 20 →
  -- Circle R is tangent to XZ and YZ
  -- Circle S is externally tangent to R and tangent to XY and YZ
  -- No point of circle S lies outside of triangle XYZ
  s.radius = 56 - 8 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_circle_s_radius_l3187_318744


namespace NUMINAMATH_CALUDE_tan_double_angle_l3187_318708

theorem tan_double_angle (α β : Real) 
  (h1 : Real.tan (α + β) = 7)
  (h2 : Real.tan (α - β) = 1) :
  Real.tan (2 * α) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3187_318708


namespace NUMINAMATH_CALUDE_markus_family_ages_l3187_318755

theorem markus_family_ages :
  ∀ (grandson_age son_age markus_age : ℕ),
    son_age = 2 * grandson_age →
    markus_age = 2 * son_age →
    grandson_age + son_age + markus_age = 140 →
    grandson_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_markus_family_ages_l3187_318755


namespace NUMINAMATH_CALUDE_max_sum_is_four_l3187_318720

-- Define the system of inequalities and conditions
def system (x y : ℕ) : Prop :=
  5 * x + 10 * y ≤ 30 ∧ 2 * x - y ≤ 3

-- Theorem statement
theorem max_sum_is_four :
  ∃ (x y : ℕ), system x y ∧ x + y = 4 ∧
  ∀ (a b : ℕ), system a b → a + b ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_four_l3187_318720


namespace NUMINAMATH_CALUDE_positive_sum_geq_two_l3187_318751

theorem positive_sum_geq_two (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 3) : 
  a + b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_geq_two_l3187_318751


namespace NUMINAMATH_CALUDE_price_decrease_l3187_318758

/-- Given a 24% decrease in price resulting in a cost of Rs. 684, prove that the original price was Rs. 900. -/
theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.24) = 684) → original_price = 900 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l3187_318758


namespace NUMINAMATH_CALUDE_square_perimeter_l3187_318719

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 144 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3187_318719


namespace NUMINAMATH_CALUDE_balloon_distribution_l3187_318772

theorem balloon_distribution (yellow_balloons : ℕ) (black_balloon_difference : ℕ) (num_schools : ℕ) : 
  yellow_balloons = 3414 →
  black_balloon_difference = 1762 →
  num_schools = 10 →
  (yellow_balloons + (yellow_balloons + black_balloon_difference)) / num_schools = 859 :=
by sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3187_318772


namespace NUMINAMATH_CALUDE_product_remainder_mod_nine_l3187_318764

theorem product_remainder_mod_nine : (2156 * 4427 * 9313) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_nine_l3187_318764


namespace NUMINAMATH_CALUDE_percentage_equality_l3187_318705

theorem percentage_equality : (0.375 / 100) * 41000 = 153.75 := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l3187_318705


namespace NUMINAMATH_CALUDE_fraction_equality_l3187_318792

theorem fraction_equality : (2 + 4 - 8 + 16 + 32 - 64 + 128) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3187_318792


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3187_318752

/-- Given a line L1 with equation 3x + 2y - 7 = 0, prove that the line L2 passing through
    the point (-1, 2) and perpendicular to L1 has the equation 2x - 3y + 8 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  (3 * x + 2 * y - 7 = 0) →  -- equation of L1
  (2 * (-1) - 3 * 2 + 8 = 0) ∧  -- L2 passes through (-1, 2)
  (3 * 2 + 2 * 3 = 0) →  -- perpendicularity condition
  (2 * x - 3 * y + 8 = 0)  -- equation of L2
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3187_318752


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l3187_318775

/-- Weather forecast for a day --/
structure DailyForecast where
  sunny_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculate expected rainfall for a single day --/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.light_rain_prob * f.light_rain_amount + f.heavy_rain_prob * f.heavy_rain_amount

/-- Calculate expected rainfall for a week --/
def expected_weekly_rainfall (f : DailyForecast) (days : ℕ) : ℝ :=
  (expected_daily_rainfall f) * days

/-- The weather forecast for the week --/
def weekly_forecast : DailyForecast :=
  { sunny_prob := 0.30
  , light_rain_prob := 0.35
  , heavy_rain_prob := 0.35
  , light_rain_amount := 3
  , heavy_rain_amount := 8 }

/-- The number of days in the forecast --/
def forecast_days : ℕ := 7

/-- Theorem: The expected rainfall for the week is approximately 26.9 inches --/
theorem expected_rainfall_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |expected_weekly_rainfall weekly_forecast forecast_days - 26.9| < ε := by
  sorry

end NUMINAMATH_CALUDE_expected_rainfall_theorem_l3187_318775


namespace NUMINAMATH_CALUDE_symmetric_point_of_M_l3187_318771

/-- The symmetric point of (x, y) with respect to the x-axis is (x, -y) -/
def symmetricPointXAxis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Given M(-2, -3), its symmetric point with respect to the x-axis is (-2, 3) -/
theorem symmetric_point_of_M : 
  let M : ℝ × ℝ := (-2, -3)
  symmetricPointXAxis M = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_M_l3187_318771


namespace NUMINAMATH_CALUDE_min_wednesday_birthdays_l3187_318767

/-- Given a company with 61 employees, where the number of employees with birthdays on Wednesday
    is greater than the number on any other day, and all other days have an equal number of birthdays,
    the minimum number of employees with birthdays on Wednesday is 13. -/
theorem min_wednesday_birthdays (total_employees : ℕ) (wednesday_birthdays : ℕ) 
  (other_day_birthdays : ℕ) : 
  total_employees = 61 →
  wednesday_birthdays > other_day_birthdays →
  total_employees = wednesday_birthdays + 6 * other_day_birthdays →
  wednesday_birthdays ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_min_wednesday_birthdays_l3187_318767


namespace NUMINAMATH_CALUDE_evaluate_expression_l3187_318723

theorem evaluate_expression : 
  Real.sqrt (9 / 4) - Real.sqrt (8 / 9) + Real.sqrt 1 = (15 - 4 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3187_318723


namespace NUMINAMATH_CALUDE_jogging_average_l3187_318712

theorem jogging_average (days_short : ℕ) (days_long : ℕ) (minutes_short : ℕ) (minutes_long : ℕ) 
  (target_average : ℕ) (total_days : ℕ) :
  days_short = 6 →
  days_long = 4 →
  minutes_short = 80 →
  minutes_long = 105 →
  target_average = 100 →
  total_days = 11 →
  (days_short * minutes_short + days_long * minutes_long + 
   (target_average * total_days - (days_short * minutes_short + days_long * minutes_long))) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_jogging_average_l3187_318712


namespace NUMINAMATH_CALUDE_max_gcd_of_consecutive_bn_l3187_318766

theorem max_gcd_of_consecutive_bn (n : ℕ) : Nat.gcd (2^n - 1) (2^(n+1) - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_consecutive_bn_l3187_318766


namespace NUMINAMATH_CALUDE_black_fraction_after_changes_l3187_318709

/-- Represents the fraction of the triangle that remains black after each change. -/
def black_fraction_after_change : ℚ := 8/9

/-- Represents the fraction of the triangle that is always black (the central triangle). -/
def always_black_fraction : ℚ := 1/9

/-- Represents the number of changes applied to the triangle. -/
def num_changes : ℕ := 4

/-- Theorem stating the fractional part of the original area that remains black after the changes. -/
theorem black_fraction_after_changes :
  (black_fraction_after_change ^ num_changes) * (1 - always_black_fraction) + always_black_fraction = 39329/59049 := by
  sorry

end NUMINAMATH_CALUDE_black_fraction_after_changes_l3187_318709


namespace NUMINAMATH_CALUDE_floor_paving_cost_l3187_318787

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 10) 
  (h2 : width = 4.75) 
  (h3 : rate = 900) : 
  length * width * rate = 42750 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l3187_318787


namespace NUMINAMATH_CALUDE_problem_solution_l3187_318763

theorem problem_solution : 
  (9^100 : ℕ) % 8 = 1 ∧ (2012^2012 : ℕ) % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3187_318763


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3187_318735

theorem water_tank_capacity : 
  ∀ (tank_capacity : ℝ),
  (0.75 * tank_capacity - 0.4 * tank_capacity = 36) →
  ⌈tank_capacity⌉ = 103 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3187_318735
