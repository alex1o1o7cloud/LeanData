import Mathlib

namespace NUMINAMATH_CALUDE_julia_tag_game_l300_30024

theorem julia_tag_game (monday_kids tuesday_kids : ℕ) 
  (h1 : monday_kids = 4) 
  (h2 : tuesday_kids = 14) : 
  monday_kids + tuesday_kids = 18 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l300_30024


namespace NUMINAMATH_CALUDE_triangle_longest_side_l300_30020

/-- Given a triangle with side lengths 10, y+5, and 3y-2, and a perimeter of 50,
    prove that the longest side length is 25.75. -/
theorem triangle_longest_side (y : ℝ) : 
  10 + (y + 5) + (3 * y - 2) = 50 →
  max 10 (max (y + 5) (3 * y - 2)) = 25.75 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l300_30020


namespace NUMINAMATH_CALUDE_current_speed_l300_30041

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 9.4) :
  ∃ (man_speed current_speed : ℝ),
    speed_with_current = man_speed + current_speed ∧
    speed_against_current = man_speed - current_speed ∧
    current_speed = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l300_30041


namespace NUMINAMATH_CALUDE_train_journey_theorem_l300_30081

/-- Represents the train's journey with two potential accident scenarios -/
theorem train_journey_theorem (D v : ℝ) : 
  (D > 0) →  -- Distance is positive
  (v > 0) →  -- Speed is positive
  -- First accident scenario
  (2 + 1 + (3 * (D - 2*v)) / (2*v) = D/v + 4) → 
  -- Second accident scenario
  (2.5 + 120/v + (6 * (D - 2*v - 120)) / (5*v) = D/v + 3.5) → 
  -- The distance D is one of the given choices
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by sorry


end NUMINAMATH_CALUDE_train_journey_theorem_l300_30081


namespace NUMINAMATH_CALUDE_age_difference_l300_30007

theorem age_difference (older_age younger_age : ℕ) : 
  older_age + younger_age = 74 → older_age = 38 → older_age - younger_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l300_30007


namespace NUMINAMATH_CALUDE_power_multiplication_l300_30056

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l300_30056


namespace NUMINAMATH_CALUDE_ellipse_properties_l300_30060

-- Define the ellipse (E)
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

-- Define the focus of a parabola
def parabola_focus (x y : ℝ) : Prop :=
  parabola x y ∧ y = 0

-- Define a point on the ellipse
def point_on_ellipse (a b x y : ℝ) : Prop :=
  ellipse a b x y

-- Define a point on the major axis
def point_on_major_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, ellipse a b m 0

-- State the theorem
theorem ellipse_properties :
  ∃ a b : ℝ,
    (∀ x y : ℝ, ellipse a b x y →
      (∃ xf yf : ℝ, parabola_focus xf yf ∧ ellipse a b xf yf) ∧
      (∀ xh yh : ℝ, hyperbola xh yh → 
        ∃ c : ℝ, c^2 = a^2 - b^2 ∧ c^2 = xh^2 - yh^2 / 2)) →
    (a^2 = 16 ∧ b^2 = 12) ∧
    (∀ m : ℝ, point_on_major_axis m →
      (∀ x y : ℝ, point_on_ellipse a b x y →
        (x = 4 → (∀ x' y' : ℝ, point_on_ellipse a b x' y' →
          (x' - m)^2 + y'^2 ≥ (x - m)^2 + y^2))) →
      1 ≤ m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l300_30060


namespace NUMINAMATH_CALUDE_farm_chickens_l300_30029

/-- Represents the number of chickens on a farm -/
def num_chickens (total_legs total_animals : ℕ) : ℕ :=
  total_animals - (total_legs - 2 * total_animals) / 2

/-- Theorem stating that given the conditions of the farm, there are 5 chickens -/
theorem farm_chickens : num_chickens 38 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_farm_chickens_l300_30029


namespace NUMINAMATH_CALUDE_mario_poster_count_l300_30051

/-- The number of posters Mario made -/
def mario_posters : ℕ := 18

/-- The number of posters Samantha made -/
def samantha_posters : ℕ := mario_posters + 15

/-- The total number of posters made -/
def total_posters : ℕ := 51

theorem mario_poster_count : 
  mario_posters = 18 ∧ 
  samantha_posters = mario_posters + 15 ∧ 
  mario_posters + samantha_posters = total_posters :=
by sorry

end NUMINAMATH_CALUDE_mario_poster_count_l300_30051


namespace NUMINAMATH_CALUDE_ravish_exam_marks_l300_30058

theorem ravish_exam_marks (pass_percentage : ℚ) (max_marks : ℕ) (fail_margin : ℕ) : 
  pass_percentage = 40 / 100 →
  max_marks = 200 →
  fail_margin = 40 →
  (pass_percentage * max_marks : ℚ) - fail_margin = 40 := by
  sorry

end NUMINAMATH_CALUDE_ravish_exam_marks_l300_30058


namespace NUMINAMATH_CALUDE_joe_height_difference_l300_30027

/-- Proves that Joe is 6 inches taller than double Sara's height -/
theorem joe_height_difference (sara : ℝ) (joe : ℝ) : 
  sara + joe = 120 →
  joe = 82 →
  joe - 2 * sara = 6 := by
sorry

end NUMINAMATH_CALUDE_joe_height_difference_l300_30027


namespace NUMINAMATH_CALUDE_share_difference_l300_30017

/-- Represents the share of money for each person -/
structure Share where
  amount : ℕ

/-- Represents the distribution of money -/
structure Distribution where
  a : Share
  b : Share
  c : Share
  d : Share

/-- The proposition that a distribution follows the given proportion -/
def follows_proportion (dist : Distribution) : Prop :=
  6 * dist.b.amount = 3 * dist.a.amount ∧
  5 * dist.b.amount = 3 * dist.c.amount ∧
  4 * dist.b.amount = 3 * dist.d.amount

/-- The theorem to be proved -/
theorem share_difference (dist : Distribution) 
  (h1 : follows_proportion dist) 
  (h2 : dist.b.amount = 3000) : 
  dist.c.amount - dist.d.amount = 1000 := by
  sorry


end NUMINAMATH_CALUDE_share_difference_l300_30017


namespace NUMINAMATH_CALUDE_rational_expressions_theorem_l300_30098

theorem rational_expressions_theorem 
  (a b c : ℚ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) :
  (a < 0 → a / |a| = -1) ∧ 
  (∃ m : ℚ, m = -2 ∧ 
    ∀ x y z : ℚ, x ≠ 0 → y ≠ 0 → z ≠ 0 → 
      m ≤ (x*y/|x*y| + |y*z|/(y*z) + z*x/|z*x| + |x*y*z|/(x*y*z))) :=
by sorry

end NUMINAMATH_CALUDE_rational_expressions_theorem_l300_30098


namespace NUMINAMATH_CALUDE_binomial_square_equivalence_l300_30082

theorem binomial_square_equivalence (x y : ℝ) : 
  (-x - y) * (-x + y) = (-x - y)^2 := by sorry

end NUMINAMATH_CALUDE_binomial_square_equivalence_l300_30082


namespace NUMINAMATH_CALUDE_special_decimal_value_l300_30089

/-- A two-digit decimal number with specific digit placements -/
def special_decimal (n : ℚ) : Prop :=
  ∃ (w : ℕ), w < 100 ∧ n = w + 0.55

/-- The theorem stating that the special decimal number is equal to 50.05 -/
theorem special_decimal_value :
  ∀ n : ℚ, special_decimal n → n = 50.05 := by
sorry

end NUMINAMATH_CALUDE_special_decimal_value_l300_30089


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l300_30043

theorem lcm_hcf_relation (x : ℕ) :
  Nat.lcm 4 x = 36 ∧ Nat.gcd 4 x = 2 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l300_30043


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l300_30037

def f (a b x : ℝ) : ℝ := 2 * x^2 + a * x + b

theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) → -- f is an even function
  f a b 1 = -3 →                -- f(1) = -3
  (∀ x, f a b x = 2 * x^2 - 5) ∧ -- f(x) = 2x² - 5
  {x : ℝ | 2 * x^2 - 5 ≥ 3 * x + 4} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3} := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_theorem_l300_30037


namespace NUMINAMATH_CALUDE_common_difference_is_neg_four_general_term_l300_30036

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  a_1 : a 1 = 23
  d : ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  a_6_positive : a 6 > 0
  a_7_negative : a 7 < 0

/-- The common difference of the arithmetic sequence is -4 -/
theorem common_difference_is_neg_four (seq : ArithmeticSequence) : seq.d = -4 := by
  sorry

/-- The general term of the arithmetic sequence is -4n + 27 -/
theorem general_term (seq : ArithmeticSequence) (n : ℕ) : seq.a n = -4 * n + 27 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_neg_four_general_term_l300_30036


namespace NUMINAMATH_CALUDE_magnified_diameter_is_0_3_l300_30025

/-- The magnification factor of the electron microscope -/
def magnification : ℝ := 1000

/-- The actual diameter of the tissue in centimeters -/
def actual_diameter : ℝ := 0.0003

/-- The diameter of the magnified image in centimeters -/
def magnified_diameter : ℝ := actual_diameter * magnification

/-- Theorem stating that the magnified diameter is 0.3 centimeters -/
theorem magnified_diameter_is_0_3 : magnified_diameter = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_magnified_diameter_is_0_3_l300_30025


namespace NUMINAMATH_CALUDE_chicken_eggs_today_l300_30040

theorem chicken_eggs_today (eggs_yesterday : ℕ) (eggs_difference : ℕ) : 
  eggs_yesterday = 10 → eggs_difference = 59 → eggs_yesterday + eggs_difference = 69 :=
by sorry

end NUMINAMATH_CALUDE_chicken_eggs_today_l300_30040


namespace NUMINAMATH_CALUDE_tablet_diagonal_length_l300_30032

theorem tablet_diagonal_length (d : ℝ) : 
  d > 0 →  -- d is positive (diagonal length)
  (d^2 / 2) = (5^2 / 2) + 5.5 → -- area difference condition
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_tablet_diagonal_length_l300_30032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l300_30079

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) 
  (h1 : a₁ = -33)
  (h2 : aₙ = 72)
  (h3 : d = 7)
  (h4 : aₙ = a₁ + (n - 1) * d) :
  n = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l300_30079


namespace NUMINAMATH_CALUDE_complementary_angle_adjustment_l300_30044

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- The ratio of two real numbers is 4:5 -/
def ratio_4_to_5 (a b : ℝ) : Prop := 5 * a = 4 * b

theorem complementary_angle_adjustment (a b : ℝ) 
  (h1 : complementary a b) 
  (h2 : ratio_4_to_5 a b) :
  complementary (1.1 * a) (0.92 * b) := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_adjustment_l300_30044


namespace NUMINAMATH_CALUDE_concyclicity_equivalence_l300_30093

-- Define the types for points and complex numbers
variable (P A B C D E F G H O₁ O₂ O₃ O₄ : ℂ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (A B C D : ℂ) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D P : ℂ) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : ℂ) : Prop := M = (A + B) / 2

-- Define circumcenter
def is_circumcenter (O P Q R : ℂ) : Prop := sorry

-- Define concyclicity
def are_concyclic (A B C D : ℂ) : Prop := sorry

-- State the theorem
theorem concyclicity_equivalence :
  is_convex_quadrilateral A B C D →
  diagonals_intersect A B C D P →
  is_midpoint E A B →
  is_midpoint F B C →
  is_midpoint G C D →
  is_midpoint H D A →
  is_circumcenter O₁ P H E →
  is_circumcenter O₂ P E F →
  is_circumcenter O₃ P F G →
  is_circumcenter O₄ P G H →
  (are_concyclic O₁ O₂ O₃ O₄ ↔ are_concyclic A B C D) :=
by sorry

end NUMINAMATH_CALUDE_concyclicity_equivalence_l300_30093


namespace NUMINAMATH_CALUDE_ball_color_probability_l300_30023

def num_balls : ℕ := 8
def num_colors : ℕ := 2

theorem ball_color_probability :
  let p : ℚ := 1 / 2  -- probability of each color
  let n : ℕ := num_balls
  let k : ℕ := n / 2  -- number of balls of each color
  (n.choose k) * p^n = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_ball_color_probability_l300_30023


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l300_30084

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {2,3,5,6}
def B : Set ℕ := {1,3,4,6,7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l300_30084


namespace NUMINAMATH_CALUDE_mixed_fraction_calculation_l300_30061

theorem mixed_fraction_calculation : 
  (((5:ℚ)/2 - 10/3)^2) / ((17:ℚ)/4 + 7/6) = 5/39 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_calculation_l300_30061


namespace NUMINAMATH_CALUDE_complex_equation_solution_l300_30048

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = Complex.I + z) :
  z = 1/2 - (1/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l300_30048


namespace NUMINAMATH_CALUDE_reading_time_difference_l300_30096

/-- Given Xanthia's and Molly's reading speeds and a book length, 
    calculate the difference in reading time in minutes. -/
theorem reading_time_difference 
  (xanthia_speed molly_speed book_length : ℕ) 
  (hx : xanthia_speed = 120)
  (hm : molly_speed = 40)
  (hb : book_length = 360) : 
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 360 := by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_reading_time_difference_l300_30096


namespace NUMINAMATH_CALUDE_g_of_2_eq_3_l300_30086

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem g_of_2_eq_3 : g 2 = 3 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_3_l300_30086


namespace NUMINAMATH_CALUDE_probability_is_three_tenths_l300_30026

/-- A bag containing 5 balls numbered from 1 to 5 -/
def Bag : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of all possible pairs of balls -/
def AllPairs : Finset (ℕ × ℕ) := (Bag.product Bag).filter (fun p => p.1 < p.2)

/-- The set of pairs whose sum is either 3 or 6 -/
def FavorablePairs : Finset (ℕ × ℕ) := AllPairs.filter (fun p => p.1 + p.2 = 3 ∨ p.1 + p.2 = 6)

/-- The probability of drawing a pair with sum 3 or 6 -/
def ProbabilityOfSum3Or6 : ℚ := (FavorablePairs.card : ℚ) / (AllPairs.card : ℚ)

theorem probability_is_three_tenths : ProbabilityOfSum3Or6 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_tenths_l300_30026


namespace NUMINAMATH_CALUDE_teacher_worked_six_months_l300_30076

/-- Calculates the number of months a teacher has worked based on given conditions -/
def teacher_months_worked (periods_per_day : ℕ) (days_per_month : ℕ) (pay_per_period : ℕ) (total_earned : ℕ) : ℕ :=
  let daily_earnings := periods_per_day * pay_per_period
  let monthly_earnings := daily_earnings * days_per_month
  total_earned / monthly_earnings

/-- Theorem stating that the teacher has worked for 6 months given the specified conditions -/
theorem teacher_worked_six_months :
  teacher_months_worked 5 24 5 3600 = 6 := by
  sorry

end NUMINAMATH_CALUDE_teacher_worked_six_months_l300_30076


namespace NUMINAMATH_CALUDE_sum_of_rationals_is_rational_l300_30094

theorem sum_of_rationals_is_rational (r₁ r₂ : ℚ) : ∃ (q : ℚ), r₁ + r₂ = q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rationals_is_rational_l300_30094


namespace NUMINAMATH_CALUDE_allison_wins_probability_l300_30045

def allison_cube : Fin 6 → ℕ := λ _ => 6

def brian_cube : Fin 6 → ℕ := λ i => i.val + 1

def noah_cube : Fin 6 → ℕ := λ i => if i.val < 3 then 3 else 5

def prob_brian_less_than_6 : ℚ := 5 / 6

def prob_noah_less_than_6 : ℚ := 1

theorem allison_wins_probability :
  (prob_brian_less_than_6 * prob_noah_less_than_6 : ℚ) = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_allison_wins_probability_l300_30045


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequality_l300_30018

/-- A quadrilateral with sides a, b, c, d and area S -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  S : ℝ

/-- Predicate for a cyclic quadrilateral with perpendicular diagonals -/
def is_cyclic_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_area_inequality (q : Quadrilateral) :
  q.S ≤ (q.a * q.c + q.b * q.d) / 2 ∧
  (q.S = (q.a * q.c + q.b * q.d) / 2 ↔ is_cyclic_perpendicular_diagonals q) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequality_l300_30018


namespace NUMINAMATH_CALUDE_ten_to_ninety_mod_seven_day_after_ten_to_ninety_friday_after_ten_to_ninety_is_saturday_l300_30070

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after_n_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => next_day (day_after_n_days start n)

theorem ten_to_ninety_mod_seven : 10^90 % 7 = 1 := by sorry

theorem day_after_ten_to_ninety (start : DayOfWeek) :
  day_after_n_days start (10^90) = next_day start := by sorry

theorem friday_after_ten_to_ninety_is_saturday :
  day_after_n_days DayOfWeek.Friday (10^90) = DayOfWeek.Saturday := by sorry

end NUMINAMATH_CALUDE_ten_to_ninety_mod_seven_day_after_ten_to_ninety_friday_after_ten_to_ninety_is_saturday_l300_30070


namespace NUMINAMATH_CALUDE_max_distance_line_circle_l300_30001

/-- Given a line ax + 2by = 1 intersecting a circle x^2 + y^2 = 1 at points A and B,
    where triangle AOB is right-angled (O is the origin), prove that the maximum
    distance between P(a,b) and Q(0,0) is √2. -/
theorem max_distance_line_circle (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, (a * A.1 + 2 * b * A.2 = 1 ∧ A.1^2 + A.2^2 = 1) ∧
                   (a * B.1 + 2 * b * B.2 = 1 ∧ B.1^2 + B.2^2 = 1) ∧
                   ((A.1 - B.1) * (A.1 + B.1) + (A.2 - B.2) * (A.2 + B.2) = 0)) →
  (∃ P : ℝ × ℝ, P.1 = a ∧ P.2 = b) →
  (∃ d : ℝ, d = Real.sqrt (a^2 + b^2) ∧ d ≤ Real.sqrt 2 ∧
            (∀ a' b' : ℝ, Real.sqrt (a'^2 + b'^2) ≤ d)) :=
by sorry


end NUMINAMATH_CALUDE_max_distance_line_circle_l300_30001


namespace NUMINAMATH_CALUDE_y_derivative_l300_30008

noncomputable def y (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l300_30008


namespace NUMINAMATH_CALUDE_pool_count_l300_30064

/-- The number of pools in two stores -/
def total_pools (store_a : ℕ) (store_b : ℕ) : ℕ :=
  store_a + store_b

/-- Theorem stating the total number of pools given the conditions -/
theorem pool_count : 
  let store_a := 200
  let store_b := 3 * store_a
  total_pools store_a store_b = 800 := by
sorry

end NUMINAMATH_CALUDE_pool_count_l300_30064


namespace NUMINAMATH_CALUDE_rectangle_max_area_l300_30028

/-- Given a rectangle with perimeter 40 units and length twice its width, 
    the maximum area of the rectangle is 800/9 square units. -/
theorem rectangle_max_area : 
  ∀ w l : ℝ, 
  w > 0 → 
  l > 0 → 
  2 * (w + l) = 40 → 
  l = 2 * w → 
  ∀ a : ℝ, a = w * l → a ≤ 800 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l300_30028


namespace NUMINAMATH_CALUDE_certain_number_proof_l300_30030

theorem certain_number_proof : 
  ∃ x : ℝ, (15 / 100) * x = (2.5 / 100) * 450 ∧ x = 75 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l300_30030


namespace NUMINAMATH_CALUDE_soccer_team_win_percentage_l300_30010

/-- Given a soccer team that played 140 games and won 70 games, 
    prove that the percentage of games won is 50%. -/
theorem soccer_team_win_percentage 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (h1 : total_games = 140) 
  (h2 : games_won = 70) : 
  (games_won : ℚ) / total_games * 100 = 50 := by
  sorry

#check soccer_team_win_percentage

end NUMINAMATH_CALUDE_soccer_team_win_percentage_l300_30010


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l300_30059

theorem fixed_point_on_line (k : ℝ) : 
  (k + 1) * 4 + (-6) + 2 - 4 * k = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l300_30059


namespace NUMINAMATH_CALUDE_negation_equivalence_l300_30042

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l300_30042


namespace NUMINAMATH_CALUDE_cubes_after_removing_layer_l300_30097

/-- The number of smaller cubes in one dimension of the large cube -/
def cube_dimension : ℕ := 10

/-- The total number of smaller cubes in the large cube -/
def total_cubes : ℕ := cube_dimension ^ 3

/-- The number of smaller cubes in one layer -/
def layer_cubes : ℕ := cube_dimension ^ 2

/-- Theorem: Removing one layer from a cube of 10x10x10 smaller cubes leaves 900 cubes -/
theorem cubes_after_removing_layer :
  total_cubes - layer_cubes = 900 := by
  sorry


end NUMINAMATH_CALUDE_cubes_after_removing_layer_l300_30097


namespace NUMINAMATH_CALUDE_half_animals_are_goats_l300_30069

/-- The number of cows the farmer has initially -/
def initial_cows : ℕ := 7

/-- The number of sheep the farmer has initially -/
def initial_sheep : ℕ := 8

/-- The number of goats the farmer has initially -/
def initial_goats : ℕ := 6

/-- The total number of animals initially -/
def initial_total : ℕ := initial_cows + initial_sheep + initial_goats

/-- The number of goats to be bought -/
def goats_to_buy : ℕ := 9

/-- Theorem stating that buying 9 goats will make half of the animals goats -/
theorem half_animals_are_goats : 
  2 * (initial_goats + goats_to_buy) = initial_total + goats_to_buy := by
  sorry

#check half_animals_are_goats

end NUMINAMATH_CALUDE_half_animals_are_goats_l300_30069


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l300_30054

theorem shirt_cost_problem (total_cost : ℕ) (num_shirts : ℕ) (known_shirt_cost : ℕ) (num_known_shirts : ℕ) :
  total_cost = 85 →
  num_shirts = 5 →
  known_shirt_cost = 15 →
  num_known_shirts = 3 →
  ∃ (remaining_shirt_cost : ℕ),
    remaining_shirt_cost * (num_shirts - num_known_shirts) + known_shirt_cost * num_known_shirts = total_cost ∧
    remaining_shirt_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l300_30054


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l300_30049

theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : total_students = 416) (h2 : girls = 160) :
  (total_students - girls) / girls = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l300_30049


namespace NUMINAMATH_CALUDE_horizontal_asymptote_rational_function_l300_30014

/-- The function f(x) = (7x^2 - 4) / (4x^2 + 8x - 3) has a horizontal asymptote at y = 7/4 -/
theorem horizontal_asymptote_rational_function :
  let f : ℝ → ℝ := λ x => (7 * x^2 - 4) / (4 * x^2 + 8 * x - 3)
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → |f x - 7/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_rational_function_l300_30014


namespace NUMINAMATH_CALUDE_marbles_selection_count_l300_30065

def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5

theorem marbles_selection_count :
  (special_marbles * (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - 1))) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_count_l300_30065


namespace NUMINAMATH_CALUDE_egg_transfer_proof_l300_30015

/-- Proves that transferring 24 eggs from basket B to basket A will make the number of eggs in basket A twice the number of eggs in basket B -/
theorem egg_transfer_proof (initial_A initial_B transferred : ℕ) 
  (h1 : initial_A = 54)
  (h2 : initial_B = 63)
  (h3 : transferred = 24) :
  initial_A + transferred = 2 * (initial_B - transferred) := by
  sorry

end NUMINAMATH_CALUDE_egg_transfer_proof_l300_30015


namespace NUMINAMATH_CALUDE_platform_length_l300_30012

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (cross_platform_time : ℝ) (cross_pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : cross_platform_time = 39)
  (h3 : cross_pole_time = 8) :
  let train_speed := train_length / cross_pole_time
  let platform_length := train_speed * cross_platform_time - train_length
  platform_length = 1162.5 := by sorry

end NUMINAMATH_CALUDE_platform_length_l300_30012


namespace NUMINAMATH_CALUDE_red_highest_probability_l300_30002

/-- Represents the colors of the balls in the box -/
inductive Color
  | Red
  | Yellow
  | Black

/-- Represents the box of balls -/
structure Box where
  total : Nat
  red : Nat
  yellow : Nat
  black : Nat

/-- Calculates the probability of drawing a ball of a given color -/
def probability (box : Box) (color : Color) : Rat :=
  match color with
  | Color.Red => box.red / box.total
  | Color.Yellow => box.yellow / box.total
  | Color.Black => box.black / box.total

/-- The box with the given conditions -/
def givenBox : Box :=
  { total := 10
    red := 7
    yellow := 2
    black := 1 }

theorem red_highest_probability :
  probability givenBox Color.Red > probability givenBox Color.Yellow ∧
  probability givenBox Color.Red > probability givenBox Color.Black :=
by sorry

end NUMINAMATH_CALUDE_red_highest_probability_l300_30002


namespace NUMINAMATH_CALUDE_candy_cookies_per_tray_l300_30067

/-- Represents the cookie distribution problem --/
structure CookieDistribution where
  num_trays : ℕ
  num_packs : ℕ
  cookies_per_pack : ℕ
  has_equal_trays : Bool

/-- The number of cookies in each tray given the distribution --/
def cookies_per_tray (d : CookieDistribution) : ℕ :=
  (d.num_packs * d.cookies_per_pack) / d.num_trays

/-- Theorem stating the number of cookies per tray in Candy's distribution --/
theorem candy_cookies_per_tray :
  let d : CookieDistribution := {
    num_trays := 4,
    num_packs := 8,
    cookies_per_pack := 12,
    has_equal_trays := true
  }
  cookies_per_tray d = 24 := by
  sorry


end NUMINAMATH_CALUDE_candy_cookies_per_tray_l300_30067


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l300_30080

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l300_30080


namespace NUMINAMATH_CALUDE_ratio_to_percentage_l300_30087

theorem ratio_to_percentage (x : ℝ) (h : x ≠ 0) :
  (x / 2) / (3 * x / 5) = 3 / 5 → (x / 2) / (3 * x / 5) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_l300_30087


namespace NUMINAMATH_CALUDE_game_c_higher_prob_l300_30019

/-- A biased coin with probability of heads 3/5 and tails 2/5 -/
structure BiasedCoin where
  p_heads : ℚ
  p_tails : ℚ
  head_prob : p_heads = 3/5
  tail_prob : p_tails = 2/5
  total_prob : p_heads + p_tails = 1

/-- Game C: Win if all three outcomes are the same -/
def prob_win_game_c (coin : BiasedCoin) : ℚ :=
  coin.p_heads^3 + coin.p_tails^3

/-- Game D: Win if first two outcomes are the same and third is different -/
def prob_win_game_d (coin : BiasedCoin) : ℚ :=
  2 * (coin.p_heads^2 * coin.p_tails + coin.p_tails^2 * coin.p_heads)

/-- The main theorem stating that Game C has a 1/25 higher probability of winning -/
theorem game_c_higher_prob (coin : BiasedCoin) :
  prob_win_game_c coin - prob_win_game_d coin = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_game_c_higher_prob_l300_30019


namespace NUMINAMATH_CALUDE_combination_equation_solution_permutation_equation_solution_l300_30083

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Permutation (Arrangement) -/
def permutation (n k : ℕ) : ℕ := sorry

theorem combination_equation_solution :
  ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9 →
  (binomial 9 x = binomial 9 (2*x - 3)) ↔ (x = 3 ∨ x = 4) := by sorry

theorem permutation_equation_solution :
  ∀ x : ℕ, 0 < x ∧ x ≤ 8 →
  (permutation 8 x = 6 * permutation 8 (x - 2)) ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_combination_equation_solution_permutation_equation_solution_l300_30083


namespace NUMINAMATH_CALUDE_soda_price_before_increase_l300_30099

/-- The original price of a can of soda -/
def original_price : ℝ := 6

/-- The percentage increase in the price of a can of soda -/
def price_increase_percentage : ℝ := 50

/-- The new price of a can of soda after the price increase -/
def new_price : ℝ := 9

/-- Theorem stating that the original price of a can of soda was 6 pounds -/
theorem soda_price_before_increase :
  original_price * (1 + price_increase_percentage / 100) = new_price :=
by sorry

end NUMINAMATH_CALUDE_soda_price_before_increase_l300_30099


namespace NUMINAMATH_CALUDE_geometric_sum_property_l300_30074

-- Define a geometric sequence with positive terms and common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sum_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum : a 1 + a 2 + a 3 = 21) : 
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_property_l300_30074


namespace NUMINAMATH_CALUDE_solve_equation_l300_30055

theorem solve_equation (x : ℝ) : 2*x + 5 - 3*x + 7 = 8 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l300_30055


namespace NUMINAMATH_CALUDE_median_moons_theorem_l300_30085

/-- Represents the two categories of planets -/
inductive PlanetCategory
| Rocky
| GasGiant

/-- Represents a planet with its category and number of moons -/
structure Planet where
  name : String
  category : PlanetCategory
  moons : ℕ

/-- The list of all planets with their data -/
def planets : List Planet := [
  ⟨"Mercury", PlanetCategory.Rocky, 0⟩,
  ⟨"Venus", PlanetCategory.Rocky, 0⟩,
  ⟨"Earth", PlanetCategory.Rocky, 1⟩,
  ⟨"Mars", PlanetCategory.Rocky, 3⟩,
  ⟨"Jupiter", PlanetCategory.GasGiant, 20⟩,
  ⟨"Saturn", PlanetCategory.GasGiant, 25⟩,
  ⟨"Uranus", PlanetCategory.GasGiant, 17⟩,
  ⟨"Neptune", PlanetCategory.GasGiant, 3⟩,
  ⟨"Pluto", PlanetCategory.GasGiant, 8⟩
]

/-- Calculate the median number of moons for a given category -/
def medianMoons (category : PlanetCategory) : ℚ := sorry

/-- The theorem stating the median number of moons for each category -/
theorem median_moons_theorem :
  medianMoons PlanetCategory.Rocky = 1/2 ∧
  medianMoons PlanetCategory.GasGiant = 17 := by sorry

end NUMINAMATH_CALUDE_median_moons_theorem_l300_30085


namespace NUMINAMATH_CALUDE_dans_marbles_l300_30004

/-- Represents the number of marbles Dan has -/
structure Marbles where
  violet : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (m : Marbles) : ℕ := m.violet + m.red + m.blue

/-- Theorem stating the total number of marbles Dan has -/
theorem dans_marbles (x : ℕ) : 
  let initial := Marbles.mk 64 0 0
  let fromMary := Marbles.mk 0 14 0
  let fromJohn := Marbles.mk 0 0 x
  let final := Marbles.mk (initial.violet + fromMary.violet + fromJohn.violet)
                          (initial.red + fromMary.red + fromJohn.red)
                          (initial.blue + fromMary.blue + fromJohn.blue)
  totalMarbles final = 78 + x := by
  sorry

end NUMINAMATH_CALUDE_dans_marbles_l300_30004


namespace NUMINAMATH_CALUDE_gym_class_distance_l300_30077

/-- The total distance students have to run in gym class -/
def total_distance (track_length : ℕ) (completed_laps remaining_laps : ℕ) : ℕ :=
  track_length * (completed_laps + remaining_laps)

/-- Proof that the total distance to run is 1500 meters -/
theorem gym_class_distance : total_distance 150 6 4 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_gym_class_distance_l300_30077


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_l300_30011

/-- Proof of the first simplification -/
theorem simplify_expression_1 (a b : ℝ) : 2 * a^2 + 9 * b - 5 * a^2 - 4 * b = -3 * a^2 + 5 * b := by
  sorry

/-- Proof of the second simplification and evaluation -/
theorem simplify_and_evaluate_expression_2 : 3 * 1 * (-2)^2 + 1^2 * (-2) - 2 * (2 * 1 * (-2)^2 - 1^2 * (-2)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_l300_30011


namespace NUMINAMATH_CALUDE_roots_of_equation_l300_30072

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6) * x * (x - 5)
  ∀ x : ℝ, f x = 0 ↔ x ∈ ({0, 2, 3, 5} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l300_30072


namespace NUMINAMATH_CALUDE_emily_trivia_score_l300_30066

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round second_round last_round : ℤ) 
  (h1 : first_round = 16)
  (h2 : second_round = 33)
  (h3 : last_round = -48) :
  first_round + second_round + last_round = 1 := by
  sorry


end NUMINAMATH_CALUDE_emily_trivia_score_l300_30066


namespace NUMINAMATH_CALUDE_product_mod_seven_l300_30050

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l300_30050


namespace NUMINAMATH_CALUDE_number_of_observations_l300_30047

theorem number_of_observations (initial_mean new_mean : ℝ) 
  (wrong_value correct_value : ℝ) (n : ℕ) : 
  initial_mean = 36 →
  wrong_value = 23 →
  correct_value = 34 →
  new_mean = 36.5 →
  (n : ℝ) * initial_mean + (correct_value - wrong_value) = (n : ℝ) * new_mean →
  n = 22 := by
  sorry

#check number_of_observations

end NUMINAMATH_CALUDE_number_of_observations_l300_30047


namespace NUMINAMATH_CALUDE_floor_sum_example_l300_30073

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l300_30073


namespace NUMINAMATH_CALUDE_bicycle_profit_percentage_l300_30075

/-- Profit percentage calculation for bicycle sale --/
theorem bicycle_profit_percentage
  (cost_price_A : ℝ)
  (profit_percentage_A : ℝ)
  (final_price : ℝ)
  (h1 : cost_price_A = 144)
  (h2 : profit_percentage_A = 25)
  (h3 : final_price = 225) :
  let selling_price_A := cost_price_A * (1 + profit_percentage_A / 100)
  let profit_B := final_price - selling_price_A
  let profit_percentage_B := (profit_B / selling_price_A) * 100
  profit_percentage_B = 25 := by sorry

end NUMINAMATH_CALUDE_bicycle_profit_percentage_l300_30075


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l300_30021

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l300_30021


namespace NUMINAMATH_CALUDE_root_of_f_given_inverse_intersection_l300_30033

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem root_of_f_given_inverse_intersection (h1 : Function.Bijective f) 
  (h2 : f_inv f 0 = -2) : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_f_given_inverse_intersection_l300_30033


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l300_30022

/-- The area of the region outside a circle of radius 2 and inside two circles of radius 4 
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem shaded_area_calculation : 
  ∃ (r₁ r₂ : ℝ) (A B : ℝ × ℝ),
    r₁ = 2 ∧ 
    r₂ = 4 ∧
    A.1^2 + A.2^2 = r₁^2 ∧
    B.1^2 + B.2^2 = r₁^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * r₁)^2 →
    let shaded_area := 
      2 * (π * r₂^2 / 6 - r₁ * (r₂^2 - r₁^2).sqrt / 2 - π * r₁^2 / 4)
    shaded_area = (20 / 3) * π - 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l300_30022


namespace NUMINAMATH_CALUDE_expected_allergies_in_sample_l300_30095

/-- The probability that an American suffers from allergies -/
def allergy_probability : ℚ := 1 / 5

/-- The size of the random sample -/
def sample_size : ℕ := 250

/-- The expected number of Americans with allergies in the sample -/
def expected_allergies : ℚ := allergy_probability * sample_size

theorem expected_allergies_in_sample :
  expected_allergies = 50 := by sorry

end NUMINAMATH_CALUDE_expected_allergies_in_sample_l300_30095


namespace NUMINAMATH_CALUDE_train_speed_l300_30090

/-- Proves that a train with given length crossing a bridge with given length in a given time has a specific speed -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 160 →
  bridge_length = 215 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l300_30090


namespace NUMINAMATH_CALUDE_fifth_term_is_seven_l300_30063

/-- An arithmetic sequence with first term a, common difference d, and n-th term given by a + (n-1)d -/
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

/-- The second term of the sequence -/
def x : ℝ := 1

/-- Given an arithmetic sequence where the first three terms are -1, x, and 3, 
    the fifth term of this sequence is 7 -/
theorem fifth_term_is_seven :
  let a := -1
  let d := x - a
  arithmetic_sequence a d 5 = 7 := by sorry

end NUMINAMATH_CALUDE_fifth_term_is_seven_l300_30063


namespace NUMINAMATH_CALUDE_cookies_leftover_l300_30006

theorem cookies_leftover (naomi oliver penelope : ℕ) 
  (h_naomi : naomi = 53)
  (h_oliver : oliver = 67)
  (h_penelope : penelope = 29)
  (package_size : ℕ) 
  (h_package : package_size = 15) : 
  (naomi + oliver + penelope) % package_size = 14 := by
  sorry

end NUMINAMATH_CALUDE_cookies_leftover_l300_30006


namespace NUMINAMATH_CALUDE_sum_and_operations_l300_30046

/-- Given three numbers a, b, and c, and a value M, such that:
    1. a + b + c = 100
    2. a - 10 = M
    3. b + 10 = M
    4. 10 * c = M
    Prove that M = 1000/21 -/
theorem sum_and_operations (a b c M : ℚ) 
  (sum_eq : a + b + c = 100)
  (a_dec : a - 10 = M)
  (b_inc : b + 10 = M)
  (c_mul : 10 * c = M) :
  M = 1000 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_operations_l300_30046


namespace NUMINAMATH_CALUDE_cylinder_uniqueness_l300_30071

/-- A cylinder with given volume and surface area -/
structure Cylinder where
  volume : ℝ
  surfaceArea : ℝ
  radius : ℝ
  height : ℝ

/-- Two cylinders are equal if they have the same radius and height -/
def Cylinder.equal (c1 c2 : Cylinder) : Prop :=
  c1.radius = c2.radius ∧ c1.height = c2.height

theorem cylinder_uniqueness (c1 c2 : Cylinder) (h_vol : c1.volume = c2.volume) 
    (h_surf : c1.surfaceArea = c2.surfaceArea) :
    c1.radius = (c1.volume / (2 * Real.pi)) ^ (1/3) ∧
    c1.height = (2 * c1.volume / Real.pi) ^ (1/3) ∧
    c2.radius = (c2.volume / (2 * Real.pi)) ^ (1/3) ∧
    c2.height = (2 * c2.volume / Real.pi) ^ (1/3) →
    Cylinder.equal c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_uniqueness_l300_30071


namespace NUMINAMATH_CALUDE_cube_diff_multiple_implies_sum_squares_multiple_of_sum_l300_30005

theorem cube_diff_multiple_implies_sum_squares_multiple_of_sum (a b c : ℕ) : 
  a < 2017 → b < 2017 → c < 2017 →
  a ≠ b → b ≠ c → a ≠ c →
  (∃ k₁ k₂ k₃ : ℤ, a^3 - b^3 = k₁ * 2017 ∧ b^3 - c^3 = k₂ * 2017 ∧ c^3 - a^3 = k₃ * 2017) →
  ∃ m : ℤ, a^2 + b^2 + c^2 = m * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_cube_diff_multiple_implies_sum_squares_multiple_of_sum_l300_30005


namespace NUMINAMATH_CALUDE_least_possible_QGK_l300_30039

theorem least_possible_QGK : ∃ (G Q K : ℕ),
  (G ≥ 1 ∧ G ≤ 9) ∧
  (Q ≥ 0 ∧ Q ≤ 9) ∧
  (K ≥ 0 ∧ K ≤ 9) ∧
  (G ≠ K) ∧
  (10 * G + G) * G = 100 * Q + 10 * G + K ∧
  ∀ (G' Q' K' : ℕ),
    (G' ≥ 1 ∧ G' ≤ 9) →
    (Q' ≥ 0 ∧ Q' ≤ 9) →
    (K' ≥ 0 ∧ K' ≤ 9) →
    (G' ≠ K') →
    (10 * G' + G') * G' = 100 * Q' + 10 * G' + K' →
    100 * Q + 10 * G + K ≤ 100 * Q' + 10 * G' + K' ∧
  100 * Q + 10 * G + K = 044 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_QGK_l300_30039


namespace NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l300_30092

/-- A circle in a 2D plane. -/
structure Circle where
  -- We don't need to define the specifics of a circle for this problem

/-- A quadrilateral in a 2D plane. -/
structure Quadrilateral where
  -- We don't need to define the specifics of a quadrilateral for this problem

/-- The maximum number of intersection points between a line segment and a circle. -/
def max_intersections_line_circle : ℕ := 2

/-- The number of sides in a quadrilateral. -/
def quadrilateral_sides : ℕ := 4

/-- Theorem: The maximum number of intersection points between a circle and a quadrilateral is 8. -/
theorem max_intersections_circle_quadrilateral (c : Circle) (q : Quadrilateral) :
  (max_intersections_line_circle * quadrilateral_sides : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l300_30092


namespace NUMINAMATH_CALUDE_tank_capacity_is_640_verify_capacity_l300_30091

/-- Represents the capacity of a tank in litres. -/
def tank_capacity : ℝ := 640

/-- Represents the time in hours it takes to empty the tank with only the outlet pipe open. -/
def outlet_time : ℝ := 10

/-- Represents the rate at which the inlet pipe adds water, in litres per minute. -/
def inlet_rate : ℝ := 4

/-- Represents the time in hours it takes to empty the tank with both inlet and outlet pipes open. -/
def both_pipes_time : ℝ := 16

/-- Theorem stating that the tank capacity is 640 litres given the conditions. -/
theorem tank_capacity_is_640 :
  tank_capacity = outlet_time * (inlet_rate * 60) * both_pipes_time / (both_pipes_time - outlet_time) :=
by
  sorry

/-- Verifies that the calculated capacity matches the given value of 640 litres. -/
theorem verify_capacity :
  tank_capacity = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_640_verify_capacity_l300_30091


namespace NUMINAMATH_CALUDE_displacement_increment_from_2_to_2_plus_d_l300_30016

/-- Represents the displacement of an object at time t -/
def displacement (t : ℝ) : ℝ := 2 * t^2

/-- Represents the increment in displacement between two time points -/
def displacementIncrement (t₁ t₂ : ℝ) : ℝ := displacement t₂ - displacement t₁

theorem displacement_increment_from_2_to_2_plus_d (d : ℝ) :
  displacementIncrement 2 (2 + d) = 8 * d + 2 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_displacement_increment_from_2_to_2_plus_d_l300_30016


namespace NUMINAMATH_CALUDE_brodys_calculator_battery_life_l300_30031

theorem brodys_calculator_battery_life :
  ∀ (total_battery : ℝ) 
    (used_battery : ℝ) 
    (exam_duration : ℝ) 
    (remaining_battery : ℝ),
  used_battery = (3/4) * total_battery →
  exam_duration = 2 →
  remaining_battery = 13 →
  total_battery = 60 := by
sorry

end NUMINAMATH_CALUDE_brodys_calculator_battery_life_l300_30031


namespace NUMINAMATH_CALUDE_stock_value_change_l300_30034

theorem stock_value_change (x : ℝ) (h : x > 0) : 
  let day1_value := x * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  (day2_value - x) / x * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_stock_value_change_l300_30034


namespace NUMINAMATH_CALUDE_amy_required_school_hours_per_week_l300_30053

/-- Amy's work schedule and earnings --/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_weeks : ℕ
  school_target_earnings : ℕ

/-- Calculate the required hours per week during school --/
def required_school_hours_per_week (schedule : WorkSchedule) : ℚ :=
  let hourly_rate : ℚ := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let total_school_hours : ℚ := schedule.school_target_earnings / hourly_rate
  total_school_hours / schedule.school_weeks

/-- Amy's specific work schedule --/
def amy_schedule : WorkSchedule := {
  summer_weeks := 8
  summer_hours_per_week := 40
  summer_earnings := 3200
  school_weeks := 32
  school_target_earnings := 4000
}

theorem amy_required_school_hours_per_week :
  required_school_hours_per_week amy_schedule = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_amy_required_school_hours_per_week_l300_30053


namespace NUMINAMATH_CALUDE_roberto_chicken_price_l300_30035

/-- Represents the scenario of Roberto's chicken and egg expenses --/
structure ChickenEggScenario where
  num_chickens : ℕ
  weekly_feed_cost : ℚ
  eggs_per_chicken_per_week : ℕ
  previous_weekly_egg_cost : ℚ
  break_even_weeks : ℕ

/-- Calculates the price per chicken that makes raising chickens cheaper than buying eggs after a given number of weeks --/
def price_per_chicken (scenario : ChickenEggScenario) : ℚ :=
  (scenario.previous_weekly_egg_cost * scenario.break_even_weeks - scenario.weekly_feed_cost * scenario.break_even_weeks) / scenario.num_chickens

/-- The theorem states that given Roberto's specific scenario, the price per chicken is $20.25 --/
theorem roberto_chicken_price : 
  let scenario : ChickenEggScenario := {
    num_chickens := 4,
    weekly_feed_cost := 1,
    eggs_per_chicken_per_week := 3,
    previous_weekly_egg_cost := 2,
    break_even_weeks := 81
  }
  price_per_chicken scenario = 81/4 := by sorry

end NUMINAMATH_CALUDE_roberto_chicken_price_l300_30035


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l300_30078

theorem sum_of_coefficients (n : ℕ) : 
  (∀ x : ℝ, x ≠ 0 → (3 * x^2 + 1/x)^n = 256 → n = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l300_30078


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l300_30068

theorem max_imaginary_part_of_roots (z : ℂ) : 
  z^6 - z^4 + z^2 - z + 1 = 0 →
  ∃ (θ : ℝ), -π/2 ≤ θ ∧ θ ≤ π/2 ∧
  (∀ (w : ℂ), w^6 - w^4 + w^2 - w + 1 = 0 → Complex.im w ≤ Real.sin θ) ∧
  θ = 900 * π / (7 * 180) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l300_30068


namespace NUMINAMATH_CALUDE_field_length_calculation_l300_30052

theorem field_length_calculation (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →
  pond_side = 8 →
  pond_side^2 = (1 / 50) * (length * width) →
  length = 80 := by
sorry

end NUMINAMATH_CALUDE_field_length_calculation_l300_30052


namespace NUMINAMATH_CALUDE_complex_equation_solution_l300_30057

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 - Complex.I → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l300_30057


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l300_30088

theorem arithmetic_mean_after_removal (arr : Finset ℤ) (sum : ℤ) : 
  Finset.card arr = 40 →
  sum = Finset.sum arr id →
  sum / 40 = 45 →
  60 ∈ arr →
  70 ∈ arr →
  ((sum - 60 - 70) : ℚ) / 38 = 43.95 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l300_30088


namespace NUMINAMATH_CALUDE_ellipse_intersection_dot_product_range_l300_30003

def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem ellipse_intersection_dot_product_range :
  ∀ k : ℝ, k ≠ 0 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    -4 ≤ dot_product x₁ y₁ x₂ y₂ ∧ dot_product x₁ y₁ x₂ y₂ < 13/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_dot_product_range_l300_30003


namespace NUMINAMATH_CALUDE_prob_one_male_one_female_proof_l300_30062

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting one male and one female student -/
def prob_one_male_one_female : ℚ := 3 / 5

theorem prob_one_male_one_female_proof :
  (num_male.choose 1 * num_female.choose 1) / total_students.choose num_selected = prob_one_male_one_female :=
sorry

end NUMINAMATH_CALUDE_prob_one_male_one_female_proof_l300_30062


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l300_30009

/-- The ratio of the volume of a cone to the volume of a cylinder with shared base radius -/
theorem cone_cylinder_volume_ratio 
  (r : ℝ) 
  (h_cyl h_cone : ℝ) 
  (h_r : r = 5)
  (h_h_cyl : h_cyl = 18)
  (h_h_cone : h_cone = 9) :
  (1 / 3 * π * r^2 * h_cone) / (π * r^2 * h_cyl) = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l300_30009


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l300_30038

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_iff (a : ℕ → ℝ) 
  (h : is_geometric_sequence a) :
  (a 1 < a 2 ∧ a 2 < a 3) ↔ is_increasing_sequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l300_30038


namespace NUMINAMATH_CALUDE_arcsin_one_over_sqrt_two_l300_30000

theorem arcsin_one_over_sqrt_two (π : ℝ) : Real.arcsin (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_over_sqrt_two_l300_30000


namespace NUMINAMATH_CALUDE_inequality_proof_l300_30013

theorem inequality_proof (x y a : ℝ) 
  (h1 : x + a < y + a) 
  (h2 : a * x > a * y) : 
  x < y ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l300_30013
