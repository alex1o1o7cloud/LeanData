import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_complex_distance_l3418_341869

theorem max_value_of_complex_distance (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (max_val : ℝ), max_val = 5 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_of_complex_distance_l3418_341869


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3418_341891

theorem circle_intersection_range (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y - 1)^2 = r^2 ∧ r > 0 ∧
   ∃ (x' y' : ℝ), (x' - 2)^2 + (y' - 1)^2 = 1 ∧
   x' = y ∧ y' = x) →
  r ∈ Set.Icc (Real.sqrt 2 - 1) (Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3418_341891


namespace NUMINAMATH_CALUDE_cricket_game_remaining_overs_l3418_341892

def cricket_game (total_overs : ℕ) (target_runs : ℕ) (initial_overs : ℕ) (initial_run_rate : ℚ) : Prop :=
  let runs_scored := initial_run_rate * initial_overs
  let remaining_runs := target_runs - runs_scored
  let remaining_overs := total_overs - initial_overs
  remaining_overs = 40

theorem cricket_game_remaining_overs :
  cricket_game 50 282 10 (32/10) :=
sorry

end NUMINAMATH_CALUDE_cricket_game_remaining_overs_l3418_341892


namespace NUMINAMATH_CALUDE_three_dice_probability_l3418_341896

theorem three_dice_probability : 
  let dice := 6
  let prob_first := (3 : ℚ) / dice  -- Probability of rolling less than 4 on first die
  let prob_second := (3 : ℚ) / dice -- Probability of rolling an even number on second die
  let prob_third := (2 : ℚ) / dice  -- Probability of rolling greater than 4 on third die
  prob_first * prob_second * prob_third = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_three_dice_probability_l3418_341896


namespace NUMINAMATH_CALUDE_A_union_B_eq_real_l3418_341851

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem A_union_B_eq_real : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_real_l3418_341851


namespace NUMINAMATH_CALUDE_value_of_a_l3418_341822

theorem value_of_a (a b c : ℚ) 
  (eq1 : a + b = c)
  (eq2 : b + c + 2 * b = 11)
  (eq3 : c = 7) :
  a = 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3418_341822


namespace NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l3418_341863

theorem largest_coefficient_in_expansion (a : ℝ) : 
  (a - 1)^5 = 32 → 
  ∃ (r : ℕ), r ≤ 5 ∧ 
    ∀ (s : ℕ), s ≤ 5 → 
      |(-1)^r * a^(5-r) * (Nat.choose 5 r)| ≥ |(-1)^s * a^(5-s) * (Nat.choose 5 s)| ∧
      (-1)^r * a^(5-r) * (Nat.choose 5 r) = 270 :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l3418_341863


namespace NUMINAMATH_CALUDE_find_second_number_l3418_341860

theorem find_second_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 28 + x) / 3) + 4 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_find_second_number_l3418_341860


namespace NUMINAMATH_CALUDE_wendy_facebook_pictures_l3418_341830

def total_pictures (one_album_pictures : ℕ) (num_other_albums : ℕ) (pictures_per_other_album : ℕ) : ℕ :=
  one_album_pictures + num_other_albums * pictures_per_other_album

theorem wendy_facebook_pictures :
  total_pictures 27 9 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendy_facebook_pictures_l3418_341830


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l3418_341873

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- The total number of apples picked by Lexie and Tom -/
def total_apples : ℕ := 36

/-- The number of apples Tom picked -/
def tom_apples : ℕ := total_apples - lexie_apples

/-- The ratio of Tom's apples to Lexie's apples -/
def apple_ratio : ℚ := tom_apples / lexie_apples

theorem apple_picking_ratio :
  apple_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l3418_341873


namespace NUMINAMATH_CALUDE_gummy_vitamins_cost_l3418_341848

/-- Calculates the total cost of gummy vitamin bottles after discounts and coupons -/
def calculate_total_cost (regular_price : ℚ) (individual_discount : ℚ) (coupon_value : ℚ) (num_bottles : ℕ) (bulk_discount : ℚ) : ℚ :=
  let discounted_price := regular_price * (1 - individual_discount)
  let price_after_coupon := discounted_price - coupon_value
  let total_before_bulk := price_after_coupon * num_bottles
  let bulk_discount_amount := total_before_bulk * bulk_discount
  total_before_bulk - bulk_discount_amount

/-- Theorem stating that the total cost for 3 bottles of gummy vitamins is $29.78 -/
theorem gummy_vitamins_cost :
  calculate_total_cost 15 (17/100) 2 3 (5/100) = 2978/100 :=
by sorry

end NUMINAMATH_CALUDE_gummy_vitamins_cost_l3418_341848


namespace NUMINAMATH_CALUDE_inequality_proof_l3418_341881

theorem inequality_proof (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : x + y = 2 * a) :
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 ∧
  (x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 ↔ x = a ∧ y = a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3418_341881


namespace NUMINAMATH_CALUDE_time_after_duration_sum_l3418_341802

/-- Represents time on a 12-hour digital clock -/
structure Time12 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds a duration to a given time and returns the resulting time on a 12-hour clock -/
def addDuration (start : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Converts a Time12 to the sum of its components -/
def timeSum (t : Time12) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_after_duration_sum :
  let start := Time12.mk 3 0 0
  let result := addDuration start 307 58 59
  timeSum result = 127 := by
  sorry

end NUMINAMATH_CALUDE_time_after_duration_sum_l3418_341802


namespace NUMINAMATH_CALUDE_parametric_to_standard_hyperbola_l3418_341815

theorem parametric_to_standard_hyperbola 
  (a b t x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (ht : t ≠ 0) :
  x = (a / 2) * (t + 1 / t) ∧ y = (b / 2) * (t - 1 / t) → 
  x^2 / a^2 - y^2 / b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_standard_hyperbola_l3418_341815


namespace NUMINAMATH_CALUDE_optimal_class_is_most_appropriate_l3418_341839

/-- Represents the number of students in a class with specific friendship conditions -/
structure ClassComposition where
  boys : ℕ
  girls : ℕ
  total : ℕ
  desks : ℕ
  boy_girl_friendships : boys * 2 = girls * 3
  total_students : total = boys + girls

/-- The optimal class composition satisfying the given conditions -/
def optimal_class : ClassComposition :=
  { boys := 21,
    girls := 14,
    total := 35,
    desks := 19,
    boy_girl_friendships := rfl,
    total_students := rfl }

/-- Theorem stating that the optimal class composition is the most appropriate solution -/
theorem optimal_class_is_most_appropriate :
  optimal_class.total = 35 ∧
  optimal_class.total > 31 ∧
  optimal_class.total = optimal_class.boys + optimal_class.girls ∧
  optimal_class.boys * 2 = optimal_class.girls * 3 :=
by sorry


end NUMINAMATH_CALUDE_optimal_class_is_most_appropriate_l3418_341839


namespace NUMINAMATH_CALUDE_parameterization_validity_l3418_341874

def is_valid_parameterization (x₀ y₀ dx dy : ℝ) : Prop :=
  y₀ = -3 * x₀ + 5 ∧ dy / dx = -3

theorem parameterization_validity 
  (x₀ y₀ dx dy : ℝ) (dx_nonzero : dx ≠ 0) :
  is_valid_parameterization x₀ y₀ dx dy ↔
  (∀ t : ℝ, -3 * (x₀ + t * dx) + 5 = y₀ + t * dy) :=
sorry

end NUMINAMATH_CALUDE_parameterization_validity_l3418_341874


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3418_341897

theorem election_votes_theorem (total_votes : ℕ) : 
  (total_votes : ℚ) * (60 / 100) - (total_votes : ℚ) * (40 / 100) = 280 → 
  total_votes = 1400 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3418_341897


namespace NUMINAMATH_CALUDE_some_number_value_l3418_341828

theorem some_number_value (x : ℝ) : 40 + 5 * 12 / (180 / x) = 41 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3418_341828


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l3418_341847

theorem pure_imaginary_product (x : ℝ) : 
  (x^4 + 6*x^3 + 7*x^2 - 14*x - 12 = 0) ↔ (x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l3418_341847


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3418_341818

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords with n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of four randomly selected chords from n points on a circle forming a convex quadrilateral -/
theorem convex_quadrilateral_probability :
  (n.choose k : ℚ) / (total_chords.choose k : ℚ) = 1 / 171 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3418_341818


namespace NUMINAMATH_CALUDE_min_colors_for_grid_colors_506_sufficient_min_colors_is_506_l3418_341846

def grid_size : ℕ := 2021

theorem min_colors_for_grid (n : ℕ) : 
  (∀ (col : Fin grid_size) (color : Fin n) (i j k l : Fin grid_size),
    i < j → j < k → k < l →
    (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
      (∀ (col' : Fin grid_size), col' > col → 
        ∃ (color' : Fin n), color' ≠ color))) →
  n ≥ 506 :=
sorry

theorem colors_506_sufficient : 
  ∃ (coloring : Fin grid_size → Fin grid_size → Fin 506),
    ∀ (col : Fin grid_size) (color : Fin 506) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin 506), color' ≠ color)) :=
sorry

theorem min_colors_is_506 : 
  (∃ (n : ℕ), 
    (∀ (col : Fin grid_size) (color : Fin n) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin n), color' ≠ color))) ∧
    (∀ m < n, ¬(∀ (col : Fin grid_size) (color : Fin m) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin m), color' ≠ color))))) →
  n = 506 :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_grid_colors_506_sufficient_min_colors_is_506_l3418_341846


namespace NUMINAMATH_CALUDE_car_speed_adjustment_l3418_341801

theorem car_speed_adjustment (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) :
  distance = 324 →
  original_time = 6 →
  time_factor = 3 / 2 →
  (distance / (original_time * time_factor)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_adjustment_l3418_341801


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_l3418_341809

/-- The quadratic equation x^2 - 4x + 2a = 0 has real roots when a = 1 -/
theorem quadratic_has_real_roots : ∃ (x : ℝ), x^2 - 4*x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_l3418_341809


namespace NUMINAMATH_CALUDE_circle_symmetry_l3418_341894

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 1 = 0

-- Define symmetry with respect to origin
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-2)^2 + y^2 = 5

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ, original_circle x y ↔ symmetric_circle (symmetric_point x y).1 (symmetric_point x y).2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3418_341894


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l3418_341878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 20 → ¬(n % p = 0)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, n > 1 ∧ ¬(is_prime n) ∧ has_no_prime_factor_less_than_20 n ∧
  (∀ m : ℕ, m > 1 → ¬(is_prime m) → has_no_prime_factor_less_than_20 m → m ≥ n) ∧
  n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l3418_341878


namespace NUMINAMATH_CALUDE_selling_price_ratio_l3418_341868

/-- Given an item with cost price c, prove that the ratio of selling prices y:x is 25:16,
    where x results in a 20% loss and y results in a 25% profit. -/
theorem selling_price_ratio (c x y : ℝ) 
  (loss : x = 0.8 * c)   -- 20% loss condition
  (profit : y = 1.25 * c) -- 25% profit condition
  : y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l3418_341868


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l3418_341806

/-- The probability of drawing two chips of different colors with replacement -/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific :
  prob_different_colors 6 5 4 = 148 / 225 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_specific_l3418_341806


namespace NUMINAMATH_CALUDE_longest_lifetime_l3418_341817

/-- A binary string is a list of booleans, where true represents 1 and false represents 0. -/
def BinaryString := List Bool

/-- The transformation function f as described in the problem. -/
def f (s : BinaryString) : BinaryString :=
  sorry

/-- The lifetime of a binary string is the number of times f can be applied until no falses remain. -/
def lifetime (s : BinaryString) : Nat :=
  sorry

/-- Generate a binary string of length n with repeated 110 pattern. -/
def repeated110 (n : Nat) : BinaryString :=
  sorry

/-- Theorem: For any n ≥ 2, the binary string with repeated 110 pattern has the longest lifetime. -/
theorem longest_lifetime (n : Nat) (h : n ≥ 2) :
  ∀ s : BinaryString, s.length = n → lifetime (repeated110 n) ≥ lifetime s :=
  sorry

end NUMINAMATH_CALUDE_longest_lifetime_l3418_341817


namespace NUMINAMATH_CALUDE_star_operation_two_neg_three_l3418_341870

def star_operation (a b : ℤ) : ℤ := a * b - (b - 1) * b

theorem star_operation_two_neg_three :
  star_operation 2 (-3) = -18 := by sorry

end NUMINAMATH_CALUDE_star_operation_two_neg_three_l3418_341870


namespace NUMINAMATH_CALUDE_incorrect_proposition_l3418_341829

theorem incorrect_proposition :
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l3418_341829


namespace NUMINAMATH_CALUDE_number_exceeding_80_percent_l3418_341833

theorem number_exceeding_80_percent : ∃ x : ℝ, x = 0.8 * x + 120 ∧ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_80_percent_l3418_341833


namespace NUMINAMATH_CALUDE_three_digit_four_digit_count_l3418_341832

theorem three_digit_four_digit_count : 
  (Finset.filter (fun x : ℕ => 
    100 ≤ 3 * x ∧ 3 * x ≤ 999 ∧ 
    1000 ≤ 4 * x ∧ 4 * x ≤ 9999) (Finset.range 10000)).card = 84 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_four_digit_count_l3418_341832


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l3418_341856

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 25 * (x + 3)^2 + 4 * y^2 = 100

-- Define the center of the ellipse
def center : ℝ × ℝ := (-3, 0)

-- Define the semi-major and semi-minor axis lengths
def semi_major_axis : ℝ := 5
def semi_minor_axis : ℝ := 2

-- Define the endpoints of the major and minor axes
def major_axis_endpoint : ℝ × ℝ := (-3, 5)
def minor_axis_endpoint : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem ellipse_axis_endpoints_distance :
  let C := major_axis_endpoint
  let D := minor_axis_endpoint
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l3418_341856


namespace NUMINAMATH_CALUDE_parabola_properties_l3418_341841

/-- Definition of the parabola function -/
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, -3)

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- Theorem stating that the given vertex and axis of symmetry are correct for the parabola -/
theorem parabola_properties :
  (∀ x, f x ≥ f (vertex.1)) ∧
  (∀ x, f x = f (2 * axis_of_symmetry - x)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3418_341841


namespace NUMINAMATH_CALUDE_plato_city_schools_l3418_341876

/-- The number of high schools in Plato City -/
def num_schools : ℕ := 21

/-- The total number of participants in the competition -/
def total_participants : ℕ := 3 * num_schools

/-- Charlie's rank in the competition -/
def charlie_rank : ℕ := (total_participants + 1) / 2

/-- Alice's rank in the competition -/
def alice_rank : ℕ := 45

/-- Bob's rank in the competition -/
def bob_rank : ℕ := 58

/-- Theorem stating that the number of schools satisfies all conditions -/
theorem plato_city_schools :
  num_schools = 21 ∧
  charlie_rank < alice_rank ∧
  charlie_rank < bob_rank ∧
  charlie_rank ≤ 45 ∧
  3 * num_schools ≥ bob_rank :=
sorry

end NUMINAMATH_CALUDE_plato_city_schools_l3418_341876


namespace NUMINAMATH_CALUDE_frog_eats_two_flies_per_day_l3418_341871

/-- The number of flies Betty's frog eats per day -/
def frog_daily_flies : ℕ :=
  let current_flies : ℕ := 5 + 6 - 1
  let total_flies : ℕ := current_flies + 4
  total_flies / 7

theorem frog_eats_two_flies_per_day : frog_daily_flies = 2 := by
  sorry

end NUMINAMATH_CALUDE_frog_eats_two_flies_per_day_l3418_341871


namespace NUMINAMATH_CALUDE_eleven_days_sufficiency_l3418_341811

/-- Represents the amount of cat food in a package -/
structure CatFood where
  days : ℝ
  nonneg : days ≥ 0

/-- The amount of food in a large package -/
def large_package : CatFood := sorry

/-- The amount of food in a small package -/
def small_package : CatFood := sorry

/-- One large package and four small packages last for 14 days -/
axiom package_combination : large_package.days + 4 * small_package.days = 14

theorem eleven_days_sufficiency :
  large_package.days + 3 * small_package.days ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_days_sufficiency_l3418_341811


namespace NUMINAMATH_CALUDE_min_value_expression_l3418_341812

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha' : a < 2) (hb' : b < 2) (hc' : c < 2) :
  (1 / ((2 - a) * (2 - b) * (2 - c))) + (1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3418_341812


namespace NUMINAMATH_CALUDE_counterexample_exists_l3418_341803

theorem counterexample_exists : ∃ (a b c : ℝ), a > b ∧ a * c ≤ b * c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3418_341803


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l3418_341825

/-- The initial number of bottle caps Joshua had -/
def initial_caps : ℕ := 40

/-- The number of bottle caps Joshua bought -/
def bought_caps : ℕ := 7

/-- The total number of bottle caps Joshua has after buying more -/
def total_caps : ℕ := 47

theorem joshua_bottle_caps : initial_caps + bought_caps = total_caps := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l3418_341825


namespace NUMINAMATH_CALUDE_correct_annual_take_home_pay_l3418_341826

def annual_take_home_pay (hourly_rate : ℝ) (regular_hours_per_week : ℝ) (weeks_per_year : ℝ)
  (overtime_hours_per_quarter : ℝ) (overtime_rate_multiplier : ℝ)
  (federal_tax_rate_1 : ℝ) (federal_tax_threshold_1 : ℝ)
  (federal_tax_rate_2 : ℝ) (federal_tax_threshold_2 : ℝ)
  (state_tax_rate : ℝ) (unemployment_insurance_rate : ℝ)
  (unemployment_insurance_threshold : ℝ) (social_security_rate : ℝ)
  (social_security_threshold : ℝ) : ℝ :=
  sorry

theorem correct_annual_take_home_pay :
  annual_take_home_pay 15 40 52 20 1.5 0.1 10000 0.12 30000 0.05 0.01 7000 0.062 142800 = 25474 :=
by sorry

end NUMINAMATH_CALUDE_correct_annual_take_home_pay_l3418_341826


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3418_341866

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3418_341866


namespace NUMINAMATH_CALUDE_k_range_l3418_341883

theorem k_range (x y k : ℝ) : 
  (3 * x + y = k + 1) → 
  (x + 3 * y = 3) → 
  (0 < x + y) → 
  (x + y < 1) → 
  (-4 < k ∧ k < 0) := by
sorry

end NUMINAMATH_CALUDE_k_range_l3418_341883


namespace NUMINAMATH_CALUDE_product_correction_l3418_341819

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem product_correction (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverse_digits a * b = 182) →  -- reversed a multiplied by b is 182
  (a * b = 533) :=  -- the correct product is 533
by sorry

end NUMINAMATH_CALUDE_product_correction_l3418_341819


namespace NUMINAMATH_CALUDE_melissa_games_l3418_341877

theorem melissa_games (total_points : ℕ) (points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) :
  total_points / points_per_game = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_melissa_games_l3418_341877


namespace NUMINAMATH_CALUDE_total_orders_filled_l3418_341864

/-- Represents the price of a catfish dinner in dollars -/
def catfish_price : ℚ := 6

/-- Represents the price of a popcorn shrimp dinner in dollars -/
def popcorn_shrimp_price : ℚ := 7/2

/-- Represents the total amount collected in dollars -/
def total_collected : ℚ := 267/2

/-- Represents the number of popcorn shrimp dinners sold -/
def popcorn_shrimp_orders : ℕ := 9

/-- Theorem stating that the total number of orders filled is 26 -/
theorem total_orders_filled : ∃ (catfish_orders : ℕ), 
  catfish_price * catfish_orders + popcorn_shrimp_price * popcorn_shrimp_orders = total_collected ∧ 
  catfish_orders + popcorn_shrimp_orders = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_orders_filled_l3418_341864


namespace NUMINAMATH_CALUDE_arccos_one_half_l3418_341882

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l3418_341882


namespace NUMINAMATH_CALUDE_negative_fraction_greater_than_negative_decimal_l3418_341845

theorem negative_fraction_greater_than_negative_decimal : -3/4 > -0.8 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_greater_than_negative_decimal_l3418_341845


namespace NUMINAMATH_CALUDE_melanie_remaining_plums_l3418_341889

/-- The number of plums Melanie picked initially -/
def initial_plums : ℕ := 7

/-- The number of plums Melanie gave away -/
def plums_given_away : ℕ := 3

/-- Theorem: Melanie has 4 plums after giving some away -/
theorem melanie_remaining_plums : 
  initial_plums - plums_given_away = 4 := by sorry

end NUMINAMATH_CALUDE_melanie_remaining_plums_l3418_341889


namespace NUMINAMATH_CALUDE_henry_collection_cost_l3418_341859

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Henry needs $144 to finish his collection -/
theorem henry_collection_cost :
  money_needed 3 15 12 = 144 := by
  sorry

end NUMINAMATH_CALUDE_henry_collection_cost_l3418_341859


namespace NUMINAMATH_CALUDE_Jose_age_is_14_l3418_341849

-- Define the ages as natural numbers
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 3
def Jose_age : ℕ := Zack_age - 4

-- Theorem statement
theorem Jose_age_is_14 : Jose_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_Jose_age_is_14_l3418_341849


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3418_341824

theorem cube_root_equation_solution : 
  {x : ℝ | ∃ y : ℝ, y^3 = 4*x - 1 ∧ ∃ z : ℝ, z^3 = 4*x + 1 ∧ ∃ w : ℝ, w^3 = 8*x ∧ y + z = w} = 
  {0, 1/4, -1/4} := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3418_341824


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3418_341804

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3418_341804


namespace NUMINAMATH_CALUDE_min_value_of_f_l3418_341867

/-- The quadratic function f(x) = 2x^2 - 16x + 22 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

/-- Theorem: The minimum value of f(x) = 2x^2 - 16x + 22 is -10 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -10 ∧ ∃ x₀ : ℝ, f x₀ = -10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3418_341867


namespace NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_two_l3418_341852

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ -2) :
  ((3 * x + 4) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 - 2*x + 1)) = (x - 1) / (x + 1) :=
by sorry

theorem fraction_value_at_two :
  ((3 * 2 + 4) / (2^2 - 1) - 2 / (2 - 1)) / ((2 + 2) / (2^2 - 2*2 + 1)) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_two_l3418_341852


namespace NUMINAMATH_CALUDE_hours_minutes_conversion_tons_kilograms_conversion_seconds_conversion_square_meters_conversion_l3418_341821

-- Define conversion rates
def minutes_per_hour : ℕ := 60
def kilograms_per_ton : ℕ := 1000
def seconds_per_minute : ℕ := 60
def square_meters_per_hectare : ℕ := 10000

-- Define the conversion functions
def hours_minutes_to_minutes (hours minutes : ℕ) : ℕ :=
  hours * minutes_per_hour + minutes

def tons_kilograms_to_kilograms (tons kilograms : ℕ) : ℕ :=
  tons * kilograms_per_ton + kilograms

def seconds_to_minutes_seconds (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / seconds_per_minute, total_seconds % seconds_per_minute)

def square_meters_to_hectares (square_meters : ℕ) : ℕ :=
  square_meters / square_meters_per_hectare

-- State the theorems
theorem hours_minutes_conversion :
  hours_minutes_to_minutes 4 35 = 275 := by sorry

theorem tons_kilograms_conversion :
  tons_kilograms_to_kilograms 4 35 = 4035 := by sorry

theorem seconds_conversion :
  seconds_to_minutes_seconds 678 = (11, 18) := by sorry

theorem square_meters_conversion :
  square_meters_to_hectares 120000 = 12 := by sorry

end NUMINAMATH_CALUDE_hours_minutes_conversion_tons_kilograms_conversion_seconds_conversion_square_meters_conversion_l3418_341821


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_four_point_five_l3418_341884

theorem reciprocal_of_negative_four_point_five :
  ((-4.5)⁻¹ : ℝ) = -2/9 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_four_point_five_l3418_341884


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt5_l3418_341857

theorem complex_modulus_sqrt5 (a : ℝ) : 
  Complex.abs (1 + a * Complex.I) = Real.sqrt 5 ↔ a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt5_l3418_341857


namespace NUMINAMATH_CALUDE_remainder_sum_squares_mod_11_l3418_341805

theorem remainder_sum_squares_mod_11 :
  (2 * (88134^2 + 88135^2 + 88136^2 + 88137^2 + 88138^2 + 88139^2)) % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_squares_mod_11_l3418_341805


namespace NUMINAMATH_CALUDE_no_141_cents_combination_l3418_341898

/-- Represents the different types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.HalfDollar => 50

/-- Represents a selection of three coins --/
structure CoinSelection :=
  (coin1 : Coin)
  (coin2 : Coin)
  (coin3 : Coin)

/-- Calculates the total value of a coin selection in cents --/
def totalValue (selection : CoinSelection) : Nat :=
  coinValue selection.coin1 + coinValue selection.coin2 + coinValue selection.coin3

/-- Theorem stating that no combination of three coins can sum to 141 cents --/
theorem no_141_cents_combination :
  ∀ (selection : CoinSelection), totalValue selection ≠ 141 := by
  sorry

end NUMINAMATH_CALUDE_no_141_cents_combination_l3418_341898


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3418_341834

theorem tangent_line_to_circle (r : ℝ) (hr : r > 0) : 
  (∀ x y : ℝ, 2*x + y = r → x^2 + y^2 = 2*r → 
   ∃ x₀ y₀ : ℝ, 2*x₀ + y₀ = r ∧ x₀^2 + y₀^2 = 2*r ∧ 
   ∀ x₁ y₁ : ℝ, 2*x₁ + y₁ = r → x₁^2 + y₁^2 ≥ 2*r) →
  r = 10 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3418_341834


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_extreme_angles_l3418_341820

/-- A cyclic quadrilateral with a specific angle ratio -/
structure CyclicQuadrilateral where
  -- Three consecutive angles with ratio 5:6:4
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  angle_ratio : a = 5 * (b / 6) ∧ c = 4 * (b / 6)
  -- Sum of opposite angles is 180°
  opposite_sum : a + d = 180 ∧ b + c = 180

/-- The largest and smallest angles in the cyclic quadrilateral -/
def extreme_angles (q : CyclicQuadrilateral) : ℝ × ℝ :=
  (108, 72)

theorem cyclic_quadrilateral_extreme_angles (q : CyclicQuadrilateral) :
  extreme_angles q = (108, 72) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_extreme_angles_l3418_341820


namespace NUMINAMATH_CALUDE_cookie_count_l3418_341861

theorem cookie_count (cookies_per_bag : ℕ) (num_bags : ℕ) : 
  cookies_per_bag = 41 → num_bags = 53 → cookies_per_bag * num_bags = 2173 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l3418_341861


namespace NUMINAMATH_CALUDE_min_value_of_f_l3418_341888

open Real

noncomputable def f (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / 
  (4 - x^2 - 10 * x * y - 25 * y^2)^(7/2)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 5/32 ∧ ∀ (x y : ℝ), f x y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3418_341888


namespace NUMINAMATH_CALUDE_integer_product_equivalence_l3418_341843

theorem integer_product_equivalence (a : ℝ) :
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = m) ↔
  (∃ k : ℤ, a = k / 6) :=
by sorry

end NUMINAMATH_CALUDE_integer_product_equivalence_l3418_341843


namespace NUMINAMATH_CALUDE_toothpick_pattern_15th_stage_l3418_341816

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem toothpick_pattern_15th_stage :
  arithmetic_sequence 5 3 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_pattern_15th_stage_l3418_341816


namespace NUMINAMATH_CALUDE_find_a_range_of_t_l3418_341875

-- Define the function f
def f (x a : ℝ) := |2 * x - a| + a

-- Theorem 1
theorem find_a : 
  (∀ x, f x 1 ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) → 
  (∃! a, ∀ x, f x a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ 
  (∀ x, f x 1 ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) :=
sorry

-- Theorem 2
theorem range_of_t :
  (∀ t : ℝ, (∃ n : ℝ, |2 * n - 1| + 1 ≤ t - (|2 * (-n) - 1| + 1)) ↔ t ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_find_a_range_of_t_l3418_341875


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_100_l3418_341887

theorem closest_integer_to_cube_root_100 :
  ∃ n : ℤ, ∀ m : ℤ, |n ^ 3 - 100| ≤ |m ^ 3 - 100| ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_100_l3418_341887


namespace NUMINAMATH_CALUDE_min_groups_for_club_l3418_341858

/-- Given a club with 30 members and a maximum group size of 12,
    the minimum number of groups required is 3. -/
theorem min_groups_for_club (total_members : ℕ) (max_group_size : ℕ) :
  total_members = 30 →
  max_group_size = 12 →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_members ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_members ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) ∧
  (∀ (num_groups : ℕ),
    num_groups * max_group_size ≥ total_members ∧
    (∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) →
    num_groups = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_min_groups_for_club_l3418_341858


namespace NUMINAMATH_CALUDE_inequalities_solution_l3418_341853

theorem inequalities_solution :
  (∀ x : ℝ, (x - 2) * (1 - 3 * x) > 2 ↔ 1 < x ∧ x < 4/3) ∧
  (∀ x : ℝ, |((x + 1) / (x - 1))| > 2 ↔ (1/3 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l3418_341853


namespace NUMINAMATH_CALUDE_machine_production_difference_l3418_341836

/-- Proves that Machine B makes 20 more products than Machine A under given conditions -/
theorem machine_production_difference :
  ∀ (rate_A rate_B total_B : ℕ) (time : ℚ),
    rate_A = 8 →
    rate_B = 10 →
    total_B = 100 →
    time = total_B / rate_B →
    total_B - (rate_A * time) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_machine_production_difference_l3418_341836


namespace NUMINAMATH_CALUDE_lines_equal_angles_with_plane_l3418_341886

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define the angle between a line and a plane
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define intersecting lines
def intersecting (l1 l2 : Line) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_equal_angles_with_plane (l1 l2 : Line) (p : Plane) 
  (h_distinct : l1 ≠ l2) 
  (h_equal_angles : angle_line_plane l1 p = angle_line_plane l2 p) :
  parallel l1 l2 ∨ intersecting l1 l2 ∨ skew l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_equal_angles_with_plane_l3418_341886


namespace NUMINAMATH_CALUDE_enemies_left_undefeated_video_game_enemies_l3418_341885

theorem enemies_left_undefeated 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) : ℕ :=
  let enemies_defeated := points_earned / points_per_enemy
  total_enemies - enemies_defeated

theorem video_game_enemies 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) 
  (h1 : points_per_enemy = 8) 
  (h2 : total_enemies = 7) 
  (h3 : points_earned = 40) : 
  enemies_left_undefeated points_per_enemy total_enemies points_earned = 2 := by
  sorry

end NUMINAMATH_CALUDE_enemies_left_undefeated_video_game_enemies_l3418_341885


namespace NUMINAMATH_CALUDE_area_ratio_of_inscribed_squares_l3418_341800

/-- A square inscribed in a circle -/
structure InscribedSquare where
  side : ℝ
  radius : ℝ
  inscribed : radius = side * Real.sqrt 2 / 2

/-- A square with two vertices on a line segment and two on a circle -/
structure PartiallyInscribedSquare where
  side : ℝ
  outer_square : InscribedSquare
  vertices_on_side : side ≤ outer_square.side
  vertices_on_circle : side = outer_square.side * Real.sqrt 2 / 5

/-- The theorem to be proved -/
theorem area_ratio_of_inscribed_squares (outer : InscribedSquare) 
    (inner : PartiallyInscribedSquare) (h : inner.outer_square = outer) :
    (inner.side ^ 2) / (outer.side ^ 2) = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_inscribed_squares_l3418_341800


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3418_341862

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3418_341862


namespace NUMINAMATH_CALUDE_reflection_matrix_correct_l3418_341879

/-- Reflection matrix over the line y = x -/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1],
    ![1, 0]]

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection over the line y = x -/
def reflect (p : Point2D) : Point2D :=
  ⟨p.y, p.x⟩

theorem reflection_matrix_correct :
  ∀ (p : Point2D),
  let reflected := reflect p
  let matrix_result := reflection_matrix.mulVec ![p.x, p.y]
  matrix_result = ![reflected.x, reflected.y] := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_correct_l3418_341879


namespace NUMINAMATH_CALUDE_integer_An_l3418_341808

theorem integer_An (a b : ℕ+) (h1 : a > b) (θ : Real) 
  (h2 : 0 < θ) (h3 : θ < Real.pi / 2) 
  (h4 : Real.sin θ = (2 * a * b : ℝ) / ((a * a + b * b) : ℝ)) :
  ∀ n : ℕ, ∃ k : ℤ, (((a * a + b * b) : ℝ) ^ n * Real.sin (n * θ)) = k := by
  sorry

end NUMINAMATH_CALUDE_integer_An_l3418_341808


namespace NUMINAMATH_CALUDE_no_y_term_in_polynomial_l3418_341890

theorem no_y_term_in_polynomial (x y k : ℝ) : 
  (2*x - 3*y + 4 + 3*k*x + 2*k*y - k = (2 + 3*k)*x + (-k + 4)) → k = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_no_y_term_in_polynomial_l3418_341890


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3418_341838

theorem complex_magnitude_equation (t : ℝ) : 
  t > 2 ∧ Complex.abs (t + 4 * Complex.I * Real.sqrt 3) * Complex.abs (7 - 2 * Complex.I) = 17 * Real.sqrt 13 ↔ 
  t = Real.sqrt (1213/53) :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3418_341838


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l3418_341865

/-- Given a cubical block of metal weighing 8 pounds, proves that another cube of the same metal with sides twice as long will weigh 64 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (h : s > 0) : 
  let original_weight : ℝ := 8
  let original_volume : ℝ := s^3
  let density : ℝ := original_weight / original_volume
  let new_side_length : ℝ := 2 * s
  let new_volume : ℝ := new_side_length^3
  let new_weight : ℝ := density * new_volume
  new_weight = 64 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l3418_341865


namespace NUMINAMATH_CALUDE_dvd_rack_sequence_l3418_341895

theorem dvd_rack_sequence (rack : Fin 6 → ℕ) 
  (h1 : rack 0 = 2)
  (h2 : rack 1 = 4)
  (h4 : rack 3 = 16)
  (h5 : rack 4 = 32)
  (h6 : rack 5 = 64)
  (h_double : ∀ i : Fin 5, rack (i.succ) = 2 * rack i) :
  rack 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rack_sequence_l3418_341895


namespace NUMINAMATH_CALUDE_min_socks_for_pair_l3418_341807

/-- Represents the color of a sock -/
inductive SockColor
| White
| Blue
| Grey

/-- Represents a sock with its color and whether it has a hole -/
structure Sock :=
  (color : SockColor)
  (hasHole : Bool)

/-- The contents of the sock box -/
def sockBox : List Sock := sorry

/-- The number of socks in the box -/
def totalSocks : Nat := sockBox.length

/-- The number of socks with holes -/
def socksWithHoles : Nat := 3

/-- The number of white socks -/
def whiteSocks : Nat := 2

/-- The number of blue socks -/
def blueSocks : Nat := 3

/-- The number of grey socks -/
def greySocks : Nat := 4

/-- Theorem stating that 7 is the minimum number of socks needed to guarantee a pair without holes -/
theorem min_socks_for_pair (draw : Nat → Sock) :
  ∃ (n : Nat), n ≤ 7 ∧
  ∃ (i j : Nat), i < j ∧ j < n ∧
  (draw i).color = (draw j).color ∧
  ¬(draw i).hasHole ∧ ¬(draw j).hasHole :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_pair_l3418_341807


namespace NUMINAMATH_CALUDE_divisor_proof_l3418_341810

theorem divisor_proof (original : Nat) (added : Nat) (sum : Nat) (h1 : original = 859622) (h2 : added = 859560) (h3 : sum = original + added) :
  sum % added = 0 ∧ added ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_proof_l3418_341810


namespace NUMINAMATH_CALUDE_notebook_distribution_l3418_341855

theorem notebook_distribution (total_notebooks : ℕ) 
  (h1 : total_notebooks = 512) : 
  ∃ (num_children : ℕ), 
    (num_children > 0) ∧ 
    (total_notebooks = num_children * (num_children / 8)) ∧
    (total_notebooks = (num_children / 2) * 16) := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3418_341855


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l3418_341831

/-- Given a circular arrangement of students, if the 7th and 27th positions
    are opposite each other, then the total number of students is 40. -/
theorem circular_arrangement_students (n : ℕ) : 
  (7 + n / 2 = 27 ∨ 27 + n / 2 = n + 7) → n = 40 :=
by sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l3418_341831


namespace NUMINAMATH_CALUDE_tenth_root_of_unity_l3418_341814

theorem tenth_root_of_unity (n : ℕ) (h : n = 3) :
  (Complex.tan (π / 4) + Complex.I) / (Complex.tan (π / 4) - Complex.I) =
  Complex.exp (Complex.I * (2 * ↑n * π / 10)) :=
by sorry

end NUMINAMATH_CALUDE_tenth_root_of_unity_l3418_341814


namespace NUMINAMATH_CALUDE_school_meeting_attendance_l3418_341835

/-- The number of parents at a school meeting -/
def num_parents : ℕ := 23

/-- The number of teachers at a school meeting -/
def num_teachers : ℕ := 8

/-- The total number of people at the school meeting -/
def total_people : ℕ := 31

/-- The number of parents who asked questions to the Latin teacher -/
def latin_teacher_parents : ℕ := 16

theorem school_meeting_attendance :
  (num_parents + num_teachers = total_people) ∧
  (num_parents = latin_teacher_parents + num_teachers - 1) ∧
  (∀ i : ℕ, i < num_teachers → latin_teacher_parents + i ≤ num_parents) ∧
  (latin_teacher_parents + num_teachers - 1 = num_parents) := by
  sorry

end NUMINAMATH_CALUDE_school_meeting_attendance_l3418_341835


namespace NUMINAMATH_CALUDE_f_properties_l3418_341899

/-- Represents a natural number in base 3 notation -/
structure Base3 where
  digits : List Nat
  first_nonzero : digits.head? ≠ some 0
  all_less_than_3 : ∀ d ∈ digits, d < 3

/-- Converts a natural number to its Base3 representation -/
noncomputable def toBase3 (n : ℕ) : Base3 := sorry

/-- Converts a Base3 representation back to a natural number -/
noncomputable def fromBase3 (b : Base3) : ℕ := sorry

/-- The function f as described in the problem -/
noncomputable def f (n : ℕ) : ℕ :=
  let b := toBase3 n
  match b.digits with
  | 1 :: rest => fromBase3 ⟨2 :: rest, sorry, sorry⟩
  | 2 :: rest => fromBase3 ⟨1 :: (rest ++ [0]), sorry, sorry⟩
  | _ => n  -- This case should not occur for valid Base3 numbers

/-- The main theorem to be proved -/
theorem f_properties :
  (∀ m n, m < n → f m < f n) ∧  -- Strictly monotone
  (∀ n, f (f n) = 3 * n) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l3418_341899


namespace NUMINAMATH_CALUDE_seedlings_per_packet_l3418_341893

theorem seedlings_per_packet (total_seedlings : ℕ) (num_packets : ℕ) 
  (h1 : total_seedlings = 420) (h2 : num_packets = 60) :
  total_seedlings / num_packets = 7 := by
  sorry

end NUMINAMATH_CALUDE_seedlings_per_packet_l3418_341893


namespace NUMINAMATH_CALUDE_third_day_distance_is_15_l3418_341827

/-- Represents a three-day hike with given distances --/
structure ThreeDayHike where
  total_distance : ℝ
  first_day_distance : ℝ
  second_day_distance : ℝ

/-- Calculates the distance hiked on the third day --/
def third_day_distance (hike : ThreeDayHike) : ℝ :=
  hike.total_distance - hike.first_day_distance - hike.second_day_distance

/-- Theorem: The distance hiked on the third day is 15 kilometers --/
theorem third_day_distance_is_15 (hike : ThreeDayHike)
    (h1 : hike.total_distance = 50)
    (h2 : hike.first_day_distance = 10)
    (h3 : hike.second_day_distance = hike.total_distance / 2) :
    third_day_distance hike = 15 := by
  sorry

end NUMINAMATH_CALUDE_third_day_distance_is_15_l3418_341827


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l3418_341850

theorem divisible_by_eleven (n : ℤ) : (18888 - n) % 11 = 0 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l3418_341850


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3418_341854

theorem product_sum_theorem : ∃ (a b c : ℕ), 
  a ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  b ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  c ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 80 ∧
  a + b + c = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3418_341854


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3418_341844

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (p : ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_roots : a 1 + a 5 = p ∧ a 1 * a 5 = 4)
  (h_p_neg : p < 0) :
  a 3 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3418_341844


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3418_341842

/-- Given a quadratic equation x^2 - 2(k-1)x + 2k^2 - 12k + 17 = 0 with roots x₁ and x₂,
    this theorem proves properties about the maximum and minimum values of x₁² + x₂²
    and the roots at these values. -/
theorem quadratic_roots_properties :
  ∀ k x₁ x₂ : ℝ,
  (x₁^2 - 2*(k-1)*x₁ + 2*k^2 - 12*k + 17 = 0) →
  (x₂^2 - 2*(k-1)*x₂ + 2*k^2 - 12*k + 17 = 0) →
  (∃ kmax : ℝ, (x₁^2 + x₂^2 ≤ 98) ∧ (k = kmax → x₁^2 + x₂^2 = 98) ∧ (k = kmax → x₁ = 7 ∧ x₂ = 7)) ∧
  (∃ kmin : ℝ, (x₁^2 + x₂^2 ≥ 2) ∧ (k = kmin → x₁^2 + x₂^2 = 2) ∧ (k = kmin → x₁ = 1 ∧ x₂ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3418_341842


namespace NUMINAMATH_CALUDE_function_properties_l3418_341813

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : f (Real.pi / 12) φ = f (Real.pi / 4) φ) :
  (φ = Real.pi / 6) ∧ 
  (∀ x, f x φ = f (-x - Real.pi / 6) φ) ∧
  (∀ x ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 6), 
    ∀ y ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 6), 
    x < y → f x φ < f y φ) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3418_341813


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l3418_341823

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_function_sum (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g) 
  (h_sum : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := by
sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l3418_341823


namespace NUMINAMATH_CALUDE_nine_books_arrangement_l3418_341880

/-- Represents a collection of books with specific adjacency requirements -/
structure BookArrangement where
  total_books : Nat
  adjacent_pairs : Nat
  single_books : Nat

/-- Calculates the number of ways to arrange books with adjacency requirements -/
def arrange_books (ba : BookArrangement) : Nat :=
  (2 ^ ba.adjacent_pairs) * Nat.factorial (ba.single_books + ba.adjacent_pairs)

/-- Theorem stating the number of ways to arrange 9 books with 2 adjacent pairs -/
theorem nine_books_arrangement :
  arrange_books ⟨9, 2, 5⟩ = 4 * Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_nine_books_arrangement_l3418_341880


namespace NUMINAMATH_CALUDE_function_solution_l3418_341840

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x

/-- The theorem stating that the function satisfying the equation has a specific form -/
theorem function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := by
  sorry

end NUMINAMATH_CALUDE_function_solution_l3418_341840


namespace NUMINAMATH_CALUDE_retail_price_calculation_l3418_341872

/-- The retail price of a machine, given wholesale price, discount, and profit margin. -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2) :
  ∃ w : ℝ, w = 120 ∧ 
    (1 - discount_rate) * w = wholesale_price + profit_rate * wholesale_price :=
by sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l3418_341872


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3418_341837

def f (a : ℝ) (x : ℝ) : ℝ := 2 - |x + a|

theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3418_341837
