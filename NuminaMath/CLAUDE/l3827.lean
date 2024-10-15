import Mathlib

namespace NUMINAMATH_CALUDE_marble_probability_l3827_382770

theorem marble_probability (b : ℕ) : 
  2 * (2 / (2 + b)) * (1 / (1 + b)) = 1/3 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3827_382770


namespace NUMINAMATH_CALUDE_roxy_garden_plants_l3827_382742

def calculate_remaining_plants (initial_flowering : ℕ) (saturday_flowering : ℕ) (saturday_fruiting : ℕ) (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  let initial_fruiting := 2 * initial_flowering
  let saturday_total := initial_flowering + initial_fruiting + saturday_flowering + saturday_fruiting
  let sunday_total := saturday_total - sunday_flowering - sunday_fruiting
  sunday_total

theorem roxy_garden_plants : 
  calculate_remaining_plants 7 3 2 1 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_garden_plants_l3827_382742


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3827_382751

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} =
  {(0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3827_382751


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l3827_382790

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The total number of people -/
def total_people : ℕ := num_teachers + num_students

/-- The number of arrangements of n elements taken r at a time -/
def arrangements (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where teachers are not adjacent -/
def arrangements_teachers_not_adjacent : ℕ := 
  arrangements num_students num_students * arrangements (num_students + 1) num_teachers

theorem teachers_not_adjacent_arrangements :
  arrangements_teachers_not_adjacent = 480 :=
by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l3827_382790


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3827_382797

/-- Proves that given an original price of $199.99999999999997, a final sale price of $144
    after two successive discounts, where the second discount is 20%,
    the first discount percentage is 10%. -/
theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 199.99999999999997)
  (h2 : final_price = 144)
  (h3 : second_discount = 0.2)
  : (original_price - final_price / (1 - second_discount)) / original_price = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3827_382797


namespace NUMINAMATH_CALUDE_power_sum_and_division_l3827_382786

theorem power_sum_and_division (a b c : ℕ) :
  2^345 + 9^5 / 9^3 = 2^345 + 81 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_l3827_382786


namespace NUMINAMATH_CALUDE_distinct_digit_sum_l3827_382739

theorem distinct_digit_sum (A B C D : Nat) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A + B + 1 = D →
  C + D = D + 1 →
  (∃ (count : Nat), count = 6 ∧ 
    (∀ (x : Nat), x < 10 → 
      (∃ (a b c : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ b ≠ c ∧ b ≠ x ∧ c ≠ x ∧
        a + b + 1 = x ∧ c + x = x + 1) ↔ 
      x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_digit_sum_l3827_382739


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3827_382719

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3827_382719


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3827_382759

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3827_382759


namespace NUMINAMATH_CALUDE_georgie_prank_ways_l3827_382749

/-- The number of windows in the mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can accomplish the prank -/
def prank_ways : ℕ := num_windows * (num_windows - 1) * (num_windows - 2)

/-- Theorem stating that the number of ways Georgie can accomplish the prank is 336 -/
theorem georgie_prank_ways : prank_ways = 336 := by
  sorry

end NUMINAMATH_CALUDE_georgie_prank_ways_l3827_382749


namespace NUMINAMATH_CALUDE_inequality_proof_l3827_382762

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hne : a ≠ b) :
  Real.sqrt a + Real.sqrt b < Real.sqrt 2 ∧ Real.sqrt 2 < 1 / (2^a) + 1 / (2^b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3827_382762


namespace NUMINAMATH_CALUDE_matching_color_probability_l3827_382792

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, red := 2, blue := 1, yellow := 0 }

/-- Clara's jelly bean distribution -/
def clara : JellyBeans :=
  { green := 3, red := 2, blue := 1, yellow := 2 }

/-- Calculates the probability of picking a specific color -/
def prob_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Theorem: The probability of Abe and Clara showing the same color is 11/40 -/
theorem matching_color_probability :
  (prob_color abe abe.green * prob_color clara clara.green) +
  (prob_color abe abe.red * prob_color clara clara.red) +
  (prob_color abe abe.blue * prob_color clara clara.blue) = 11 / 40 := by
  sorry

end NUMINAMATH_CALUDE_matching_color_probability_l3827_382792


namespace NUMINAMATH_CALUDE_absent_children_l3827_382794

theorem absent_children (total_children : ℕ) (bananas : ℕ) (absent : ℕ) : 
  total_children = 840 →
  bananas = 840 * 2 →
  bananas = (840 - absent) * 4 →
  absent = 420 := by
sorry

end NUMINAMATH_CALUDE_absent_children_l3827_382794


namespace NUMINAMATH_CALUDE_zero_integer_not_positive_negative_l3827_382725

theorem zero_integer_not_positive_negative :
  (0 : ℤ) ∈ Set.univ ∧ (0 : ℤ) ∉ {x : ℤ | x > 0} ∧ (0 : ℤ) ∉ {x : ℤ | x < 0} := by
  sorry

end NUMINAMATH_CALUDE_zero_integer_not_positive_negative_l3827_382725


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l3827_382782

theorem power_tower_mod_1000 : 3^(3^(3^3)) ≡ 387 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l3827_382782


namespace NUMINAMATH_CALUDE_range_of_s_l3827_382761

-- Define the type for composite positive integers
def CompositePositiveInteger := {n : ℕ | n > 1 ∧ ¬ Prime n}

-- Define the function s
def s : CompositePositiveInteger → ℕ :=
  sorry -- Definition of s as sum of distinct prime factors

-- State the theorem about the range of s
theorem range_of_s :
  ∀ m : ℕ, m ≥ 2 ↔ ∃ n : CompositePositiveInteger, s n = m :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l3827_382761


namespace NUMINAMATH_CALUDE_michael_fish_count_l3827_382701

def total_pets : ℕ := 160
def dog_percentage : ℚ := 225 / 1000
def cat_percentage : ℚ := 375 / 1000
def bunny_percentage : ℚ := 15 / 100
def bird_percentage : ℚ := 1 / 10

theorem michael_fish_count :
  let dogs := (dog_percentage * total_pets).floor
  let cats := (cat_percentage * total_pets).floor
  let bunnies := (bunny_percentage * total_pets).floor
  let birds := (bird_percentage * total_pets).floor
  let fish := total_pets - (dogs + cats + bunnies + birds)
  fish = 24 := by
sorry

end NUMINAMATH_CALUDE_michael_fish_count_l3827_382701


namespace NUMINAMATH_CALUDE_sallys_nickels_from_dad_l3827_382791

/-- The number of nickels Sally's dad gave her -/
def dads_nickels (initial_nickels mother_nickels total_nickels : ℕ) : ℕ :=
  total_nickels - (initial_nickels + mother_nickels)

/-- Proof that Sally's dad gave her 9 nickels -/
theorem sallys_nickels_from_dad :
  dads_nickels 7 2 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_from_dad_l3827_382791


namespace NUMINAMATH_CALUDE_sum_of_integers_l3827_382784

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 180) : 
  x.val + y.val = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3827_382784


namespace NUMINAMATH_CALUDE_no_solution_implies_positive_b_l3827_382703

theorem no_solution_implies_positive_b (a b : ℝ) :
  (∀ x y : ℝ, y ≠ x^2 + a*x + b ∨ x ≠ y^2 + a*y + b) →
  b > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_positive_b_l3827_382703


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l3827_382798

/-- Given two lines intersecting at point P(2,5) with slopes -1 and -2 respectively,
    and points Q and R on the x-axis, prove that the area of triangle PQR is 6.25 -/
theorem area_of_triangle_PQR : ∃ (Q R : ℝ × ℝ),
  let P : ℝ × ℝ := (2, 5)
  let slope_PQ : ℝ := -1
  let slope_PR : ℝ := -2
  Q.2 = 0 ∧ R.2 = 0 ∧
  (Q.1 - P.1) / (Q.2 - P.2) = slope_PQ ∧
  (R.1 - P.1) / (R.2 - P.2) = slope_PR ∧
  (1/2 : ℝ) * |Q.1 - R.1| * P.2 = 6.25 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_l3827_382798


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l3827_382711

/-- Quadratic function f(x) = x^2 + 3x + n -/
def f (n : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + n

/-- Predicate for exactly one positive real root -/
def has_exactly_one_positive_root (n : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ f n x = 0

theorem quadratic_intersection_theorem :
  has_exactly_one_positive_root (-2) ∧
  ∀ n : ℝ, has_exactly_one_positive_root n → n = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l3827_382711


namespace NUMINAMATH_CALUDE_quadratic_point_between_roots_l3827_382779

/-- Given a quadratic function y = x^2 + 2x + c with roots x₁ and x₂ (where x₁ < x₂),
    and a point P(m, n) on the graph, if n < 0, then x₁ < m < x₂. -/
theorem quadratic_point_between_roots
  (c : ℝ) (x₁ x₂ m n : ℝ)
  (h_roots : x₁ < x₂)
  (h_on_graph : n = m^2 + 2*m + c)
  (h_roots_def : x₁^2 + 2*x₁ + c = 0 ∧ x₂^2 + 2*x₂ + c = 0)
  (h_n_neg : n < 0) :
  x₁ < m ∧ m < x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_point_between_roots_l3827_382779


namespace NUMINAMATH_CALUDE_theatre_fraction_l3827_382714

/-- Represents the fraction of students in a school -/
structure SchoolFractions where
  pe : ℚ  -- Fraction of students who took P.E.
  theatre : ℚ  -- Fraction of students who took theatre
  music : ℚ  -- Fraction of students who took music

/-- Represents the fraction of students who left the school -/
structure LeavingFractions where
  pe : ℚ  -- Fraction of P.E. students who left
  theatre : ℚ  -- Fraction of theatre students who left

theorem theatre_fraction (s : SchoolFractions) (l : LeavingFractions) : 
  s.pe = 1/2 ∧ 
  s.pe + s.theatre + s.music = 1 ∧
  l.pe = 1/3 ∧
  l.theatre = 1/4 ∧
  (s.pe * (1 - l.pe) + s.music) / (1 - s.pe * l.pe - s.theatre * l.theatre) = 2/3 →
  s.theatre = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_theatre_fraction_l3827_382714


namespace NUMINAMATH_CALUDE_cooper_remaining_pies_l3827_382767

/-- The number of apple pies Cooper makes per day -/
def pies_per_day : ℕ := 7

/-- The number of days Cooper makes pies -/
def days_making_pies : ℕ := 12

/-- The number of pies Ashley eats -/
def pies_eaten : ℕ := 50

/-- The number of pies remaining with Cooper -/
def remaining_pies : ℕ := pies_per_day * days_making_pies - pies_eaten

theorem cooper_remaining_pies : remaining_pies = 34 := by sorry

end NUMINAMATH_CALUDE_cooper_remaining_pies_l3827_382767


namespace NUMINAMATH_CALUDE_sibling_difference_l3827_382737

/-- Given the number of siblings for Masud, calculate the number of siblings for Janet -/
def janet_siblings (masud_siblings : ℕ) : ℕ :=
  4 * masud_siblings - 60

/-- Given the number of siblings for Masud, calculate the number of siblings for Carlos -/
def carlos_siblings (masud_siblings : ℕ) : ℕ :=
  (3 * masud_siblings) / 4

/-- Theorem stating the difference in siblings between Janet and Carlos -/
theorem sibling_difference (masud_siblings : ℕ) (h : masud_siblings = 60) :
  janet_siblings masud_siblings - carlos_siblings masud_siblings = 135 := by
  sorry


end NUMINAMATH_CALUDE_sibling_difference_l3827_382737


namespace NUMINAMATH_CALUDE_crackers_eaten_equals_180_l3827_382783

/-- Calculates the total number of animal crackers eaten by Mrs. Gable's students -/
def total_crackers_eaten (total_students : ℕ) (students_not_eating : ℕ) (crackers_per_pack : ℕ) : ℕ :=
  (total_students - students_not_eating) * crackers_per_pack

/-- Proves that the total number of animal crackers eaten is 180 -/
theorem crackers_eaten_equals_180 :
  total_crackers_eaten 20 2 10 = 180 := by
  sorry

#eval total_crackers_eaten 20 2 10

end NUMINAMATH_CALUDE_crackers_eaten_equals_180_l3827_382783


namespace NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l3827_382727

theorem decimal_sum_equals_fraction : 
  0.2 + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l3827_382727


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l3827_382717

/-- A parabola with x-axis as its axis of symmetry, vertex at the origin, and latus rectum length of 8 -/
structure Parabola where
  symmetry_axis : ℝ → ℝ
  vertex : ℝ × ℝ
  latus_rectum : ℝ
  h_symmetry : symmetry_axis = λ y => 0
  h_vertex : vertex = (0, 0)
  h_latus_rectum : latus_rectum = 8

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 8*x ∨ y^2 = -8*x}

theorem parabola_equation_correct (p : Parabola) :
  ∀ (x y : ℝ), (x, y) ∈ parabola_equation p ↔
    (∃ t : ℝ, x = t^2 / 2 ∧ y = t) ∨ (∃ t : ℝ, x = -t^2 / 2 ∧ y = t) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l3827_382717


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_88_l3827_382710

theorem twenty_percent_greater_than_88 (x : ℝ) : x = 88 * 1.2 → x = 105.6 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_88_l3827_382710


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l3827_382789

def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def initial_dimes : ℕ := 19
def initial_quarters : ℕ := 6

def candy_bars_bought : ℕ := 4
def dimes_per_candy : ℕ := 3

def lollipops_bought : ℕ := 1

theorem money_left_after_purchase : 
  (initial_dimes * dime_value + initial_quarters * quarter_value) - 
  (candy_bars_bought * dimes_per_candy * dime_value + lollipops_bought * quarter_value) = 195 := by
sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l3827_382789


namespace NUMINAMATH_CALUDE_truth_and_lie_l3827_382723

/-- Represents a person who either always tells the truth or always lies -/
inductive Person
| Truthful
| Liar

/-- The setup of three people sitting side by side -/
structure Setup :=
  (left : Person)
  (middle : Person)
  (right : Person)

/-- The statement made by the left person about the middle person's response -/
def leftStatement (s : Setup) : Prop :=
  s.middle = Person.Truthful

/-- The statement made by the right person about the middle person's response -/
def rightStatement (s : Setup) : Prop :=
  s.middle = Person.Liar

theorem truth_and_lie (s : Setup) :
  (leftStatement s = true ↔ s.left = Person.Truthful) ∧
  (rightStatement s = false ↔ s.right = Person.Liar) :=
sorry

end NUMINAMATH_CALUDE_truth_and_lie_l3827_382723


namespace NUMINAMATH_CALUDE_sqrt_neg_three_squared_l3827_382747

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_squared_l3827_382747


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l3827_382700

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define angle congruence
def are_congruent (α β : Angle) : Prop := α = β

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) :
  are_vertical_angles α β → are_congruent α β := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l3827_382700


namespace NUMINAMATH_CALUDE_unique_divisibility_pair_l3827_382733

/-- A predicate that checks if there are infinitely many positive integers k 
    for which (k^n + k^2 - 1) divides (k^m + k - 1) -/
def InfinitelyManyDivisors (m n : ℕ) : Prop :=
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ (k^n + k^2 - 1) ∣ (k^m + k - 1)

/-- The theorem stating that (5,3) is the only pair of integers (m,n) 
    satisfying the given conditions -/
theorem unique_divisibility_pair :
  ∀ m n : ℕ, m > 2 → n > 2 → InfinitelyManyDivisors m n → m = 5 ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_divisibility_pair_l3827_382733


namespace NUMINAMATH_CALUDE_combine_like_terms_l3827_382707

theorem combine_like_terms (a : ℝ) : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l3827_382707


namespace NUMINAMATH_CALUDE_f_inequality_l3827_382755

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + a

-- State the theorem
theorem f_inequality (a b : ℝ) :
  (f a b 0 = 3) →
  (∀ x, f a b (2 - x) = f a b x) →
  (∀ x, f a b (b^x) ≤ f a b (a^x)) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l3827_382755


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3827_382776

theorem min_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := by
  sorry

theorem lower_bound_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3827_382776


namespace NUMINAMATH_CALUDE_family_reunion_attendance_l3827_382772

/-- The number of people at a family reunion --/
def family_reunion (male_adults female_adults children : ℕ) : ℕ :=
  male_adults + female_adults + children

/-- Theorem: Given the conditions, the family reunion has 750 people --/
theorem family_reunion_attendance :
  ∀ (male_adults female_adults children : ℕ),
  male_adults = 100 →
  female_adults = male_adults + 50 →
  children = 2 * (male_adults + female_adults) →
  family_reunion male_adults female_adults children = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_family_reunion_attendance_l3827_382772


namespace NUMINAMATH_CALUDE_ball_volume_ratio_l3827_382775

theorem ball_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) (h₃ : r₃ = 3 * r₁) :
  (4 / 3 * π * r₃^3) = 3 * ((4 / 3 * π * r₁^3) + (4 / 3 * π * r₂^3)) :=
by sorry

end NUMINAMATH_CALUDE_ball_volume_ratio_l3827_382775


namespace NUMINAMATH_CALUDE_darcie_age_ratio_l3827_382763

theorem darcie_age_ratio (darcie_age mother_age father_age : ℕ) :
  darcie_age = 4 →
  mother_age = (4 * father_age) / 5 →
  father_age = 30 →
  darcie_age * 6 = mother_age :=
by
  sorry

end NUMINAMATH_CALUDE_darcie_age_ratio_l3827_382763


namespace NUMINAMATH_CALUDE_litter_size_l3827_382713

/-- Represents the number of puppies in the litter -/
def puppies : ℕ := sorry

/-- The profit John makes from selling the puppies -/
def profit : ℕ := 1500

/-- The amount John pays to the stud owner -/
def stud_fee : ℕ := 300

/-- The price for which John sells each puppy -/
def price_per_puppy : ℕ := 600

theorem litter_size : 
  puppies = 8 ∧ 
  (puppies / 2 - 1) * price_per_puppy - stud_fee = profit :=
sorry

end NUMINAMATH_CALUDE_litter_size_l3827_382713


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l3827_382738

/-- Given a line y = a intersecting the parabola y = x^2 at points A and B, 
    and a point C on the parabola such that angle ACB is a right angle, 
    the range of possible values for a is [1, +∞) -/
theorem parabola_intersection_range (a : ℝ) : 
  (∃ A B C : ℝ × ℝ, 
    (A.2 = a ∧ A.2 = A.1^2) ∧ 
    (B.2 = a ∧ B.2 = B.1^2) ∧ 
    (C.2 = C.1^2) ∧ 
    ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0)) 
  ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l3827_382738


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3827_382726

theorem complex_magnitude_squared (z : ℂ) (h : 2 * z + Complex.abs z = -3 + 12 * Complex.I) : Complex.normSq z = 61 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3827_382726


namespace NUMINAMATH_CALUDE_vovochka_max_candies_l3827_382736

/-- Represents the problem of distributing candies among classmates. -/
structure CandyDistribution where
  totalCandies : Nat
  totalClassmates : Nat
  minGroupSize : Nat
  minGroupCandies : Nat

/-- Calculates the maximum number of candies Vovochka can keep. -/
def maxCandiesForVovochka (dist : CandyDistribution) : Nat :=
  sorry

/-- Theorem stating the maximum number of candies Vovochka can keep. -/
theorem vovochka_max_candies :
  let dist : CandyDistribution := {
    totalCandies := 200,
    totalClassmates := 25,
    minGroupSize := 16,
    minGroupCandies := 100
  }
  maxCandiesForVovochka dist = 37 := by sorry

end NUMINAMATH_CALUDE_vovochka_max_candies_l3827_382736


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_l3827_382728

theorem quadrilateral_angle_sum (a b c d : ℕ) : 
  50 ≤ a ∧ a ≤ 200 ∧
  50 ≤ b ∧ b ≤ 200 ∧
  50 ≤ c ∧ c ≤ 200 ∧
  50 ≤ d ∧ d ≤ 200 ∧
  b = 75 ∧ c = 80 ∧ d = 120 ∧
  a + b + c + d = 360 →
  a = 85 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_l3827_382728


namespace NUMINAMATH_CALUDE_c_nonzero_necessary_not_sufficient_l3827_382702

/-- Represents a conic section defined by the equation ax^2 + y^2 = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Determines if a conic section is an ellipse or hyperbola -/
def isEllipseOrHyperbola (conic : ConicSection) : Prop :=
  (conic.a > 0 ∧ conic.c > 0) ∨ (conic.a < 0 ∧ conic.c ≠ 0)

/-- The main theorem stating that c ≠ 0 is necessary but not sufficient -/
theorem c_nonzero_necessary_not_sufficient :
  (∀ conic : ConicSection, isEllipseOrHyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬isEllipseOrHyperbola conic) :=
sorry

end NUMINAMATH_CALUDE_c_nonzero_necessary_not_sufficient_l3827_382702


namespace NUMINAMATH_CALUDE_square_division_perimeters_l3827_382771

theorem square_division_perimeters (p : ℚ) : 
  (∃ a b c d e f : ℚ, 
    a + b + c = 1 ∧ 
    d + e + f = 1 ∧ 
    2 * (a + d) = p ∧ 
    2 * (b + e) = p ∧ 
    2 * (c + f) = p) → 
  (p = 8/3 ∨ p = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_square_division_perimeters_l3827_382771


namespace NUMINAMATH_CALUDE_exists_valid_strategy_l3827_382793

/-- Represents a strategy for distributing balls in boxes -/
structure Strategy where
  distribute : Fin 2018 → ℕ

/-- Represents the game setup and rules -/
structure Game where
  boxes : Fin 2018
  pairs : Fin 4032
  pairAssignment : Fin 4032 → Fin 2018 × Fin 2018

/-- Predicate to check if a strategy results in distinct ball counts -/
def isValidStrategy (g : Game) (s : Strategy) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → s.distribute i ≠ s.distribute j

/-- Theorem stating the existence of a valid strategy -/
theorem exists_valid_strategy (g : Game) : ∃ s : Strategy, isValidStrategy g s := by
  sorry


end NUMINAMATH_CALUDE_exists_valid_strategy_l3827_382793


namespace NUMINAMATH_CALUDE_right_triangle_max_area_l3827_382729

theorem right_triangle_max_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1/2) * a * b ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_max_area_l3827_382729


namespace NUMINAMATH_CALUDE_B_join_time_l3827_382752

/-- Represents the time (in months) when B joined the business -/
def time_B_joined : ℝ := 7.5

/-- A's initial investment -/
def A_investment : ℝ := 27000

/-- B's investment when joining -/
def B_investment : ℝ := 36000

/-- Total duration of the business in months -/
def total_duration : ℝ := 12

/-- Ratio of A's profit share to B's profit share -/
def profit_ratio : ℝ := 2

theorem B_join_time :
  (A_investment * total_duration) / (B_investment * (total_duration - time_B_joined)) = profit_ratio := by
  sorry

end NUMINAMATH_CALUDE_B_join_time_l3827_382752


namespace NUMINAMATH_CALUDE_lana_extra_tickets_l3827_382766

/-- Calculates the number of extra tickets bought given the ticket price, number of tickets for friends, and total amount spent. -/
def extra_tickets (ticket_price : ℕ) (friends_tickets : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - friends_tickets * ticket_price) / ticket_price

/-- Proves that Lana bought 2 extra tickets given the problem conditions. -/
theorem lana_extra_tickets :
  let ticket_price : ℕ := 6
  let friends_tickets : ℕ := 8
  let total_spent : ℕ := 60
  extra_tickets ticket_price friends_tickets total_spent = 2 := by
  sorry

end NUMINAMATH_CALUDE_lana_extra_tickets_l3827_382766


namespace NUMINAMATH_CALUDE_division_problem_l3827_382764

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3) 
  (h2 : y / z = 2 / 5) : 
  z / x = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3827_382764


namespace NUMINAMATH_CALUDE_no_real_solution_l3827_382777

theorem no_real_solution : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + 1/y = 5 ∧ y + 1/x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l3827_382777


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3827_382731

/-- The minimum squared distance from the origin to the line 4x + 3y - 10 = 0 is 4 -/
theorem min_distance_to_line : 
  (∀ m n : ℝ, 4*m + 3*n - 10 = 0 → m^2 + n^2 ≥ 4) ∧ 
  (∃ m n : ℝ, 4*m + 3*n - 10 = 0 ∧ m^2 + n^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3827_382731


namespace NUMINAMATH_CALUDE_beckys_necklace_count_l3827_382758

/-- Calculates the final number of necklaces in Becky's collection -/
def final_necklace_count (initial : ℕ) (broken : ℕ) (new : ℕ) (gifted : ℕ) : ℕ :=
  initial - broken + new - gifted

/-- Theorem stating that Becky's final necklace count is 37 -/
theorem beckys_necklace_count :
  final_necklace_count 50 3 5 15 = 37 := by
  sorry

end NUMINAMATH_CALUDE_beckys_necklace_count_l3827_382758


namespace NUMINAMATH_CALUDE_new_building_windows_l3827_382734

/-- The number of windows needed for a new building -/
def total_windows (installed : ℕ) (install_time : ℕ) (remaining_time : ℕ) : ℕ :=
  installed + remaining_time / install_time

/-- Theorem: The new building needs 14 windows in total -/
theorem new_building_windows :
  total_windows 8 8 48 = 14 := by
  sorry

end NUMINAMATH_CALUDE_new_building_windows_l3827_382734


namespace NUMINAMATH_CALUDE_find_number_l3827_382756

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3827_382756


namespace NUMINAMATH_CALUDE_snow_probability_l3827_382750

theorem snow_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^5 = 1023/1024 := by sorry

end NUMINAMATH_CALUDE_snow_probability_l3827_382750


namespace NUMINAMATH_CALUDE_regular_hexagon_properties_l3827_382781

/-- A regular hexagon inscribed in a circle -/
structure RegularHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The circumference of the circumscribed circle -/
  circumference : ℝ
  /-- The arc length corresponding to one side of the hexagon -/
  arc_length : ℝ
  /-- The area of the hexagon -/
  area : ℝ

/-- Properties of a regular hexagon with side length 6 -/
theorem regular_hexagon_properties :
  ∃ (h : RegularHexagon),
    h.side_length = 6 ∧
    h.radius = 6 ∧
    h.circumference = 12 * Real.pi ∧
    h.arc_length = 2 * Real.pi ∧
    h.area = 54 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_properties_l3827_382781


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l3827_382778

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def polygon_interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = 
    (polygon_interior_angle_sum octagon_sides) / octagon_sides :=
by sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l3827_382778


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3827_382721

theorem toms_age_ratio (T N : ℝ) (h1 : T > 0) (h2 : N > 0) 
  (h3 : T - N = 3 * (T - 3 * N)) : T / N = 4 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l3827_382721


namespace NUMINAMATH_CALUDE_rachels_homework_l3827_382774

theorem rachels_homework (math_pages reading_pages total_pages : ℕ) : 
  reading_pages = math_pages + 3 →
  total_pages = math_pages + reading_pages →
  total_pages = 23 →
  math_pages = 10 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l3827_382774


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3827_382745

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 + 10*x - 3

-- Define the roots
axiom p : ℝ
axiom q : ℝ
axiom r : ℝ

-- Axioms stating that p, q, and r are roots of f
axiom p_root : f p = 0
axiom q_root : f q = 0
axiom r_root : f r = 0

-- The theorem to prove
theorem root_sum_theorem :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 38 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3827_382745


namespace NUMINAMATH_CALUDE_taxi_occupancy_l3827_382787

theorem taxi_occupancy (cars : Nat) (car_capacity : Nat) (vans : Nat) (van_capacity : Nat) 
  (taxis : Nat) (total_people : Nat) :
  cars = 3 → car_capacity = 4 → vans = 2 → van_capacity = 5 → taxis = 6 → total_people = 58 →
  ∃ (taxi_capacity : Nat), taxi_capacity = 6 ∧ 
    cars * car_capacity + vans * van_capacity + taxis * taxi_capacity = total_people :=
by sorry

end NUMINAMATH_CALUDE_taxi_occupancy_l3827_382787


namespace NUMINAMATH_CALUDE_normal_distribution_equality_l3827_382796

-- Define the random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the normal distribution parameters
variable (μ σ : ℝ)

-- Define the probability measure
variable (P : Set ℝ → ℝ)

-- State the theorem
theorem normal_distribution_equality (h1 : μ = 2) 
  (h2 : P {x | ξ x ≤ 4 - a} = P {x | ξ x ≥ 2 + 3 * a}) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_equality_l3827_382796


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l3827_382748

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The equation x² - x = 0 -/
def equation_D : QuadraticEquation where
  a := 1
  b := -1
  c := 0
  a_nonzero := by sorry

theorem equation_D_is_quadratic : equation_D.a ≠ 0 ∧ 
  equation_D.a * X^2 + equation_D.b * X + equation_D.c = X^2 - X := by sorry


end NUMINAMATH_CALUDE_equation_D_is_quadratic_l3827_382748


namespace NUMINAMATH_CALUDE_range_of_m_l3827_382785

def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem range_of_m : ∀ m : ℝ, (¬(p m ∨ ¬(q m))) ↔ m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3827_382785


namespace NUMINAMATH_CALUDE_diamond_two_three_l3827_382744

def diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_three_l3827_382744


namespace NUMINAMATH_CALUDE_complex_point_to_number_l3827_382708

theorem complex_point_to_number (z : ℂ) : (z / Complex.I).re = 3 ∧ (z / Complex.I).im = -1 → z = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_point_to_number_l3827_382708


namespace NUMINAMATH_CALUDE_range_of_a_l3827_382765

def A (a : ℝ) : Set ℝ := {x | |x - 1| ≤ a ∧ a > 0}

def B : Set ℝ := {x | x^2 - 6*x - 7 > 0}

theorem range_of_a (a : ℝ) :
  (A a ∩ B = ∅) → (0 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3827_382765


namespace NUMINAMATH_CALUDE_line_equation_through_points_l3827_382757

/-- The equation of a line passing through two points (5, 0) and (2, -5) -/
theorem line_equation_through_points :
  ∃ (A B C : ℝ),
    (A * 5 + B * 0 + C = 0) ∧
    (A * 2 + B * (-5) + C = 0) ∧
    (A = 5 ∧ B = -3 ∧ C = -25) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l3827_382757


namespace NUMINAMATH_CALUDE_stock_price_change_l3827_382706

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_day1 := initial_price * (1 - 0.15)
  let price_after_day2 := price_after_day1 * (1 + 0.25)
  let percent_change := (price_after_day2 - initial_price) / initial_price * 100
  percent_change = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l3827_382706


namespace NUMINAMATH_CALUDE_swim_club_scenario_l3827_382704

/-- Represents a swim club with members, some of whom have passed a lifesaving test
    and some of whom have taken a preparatory course. -/
structure SwimClub where
  total_members : ℕ
  passed_test : ℕ
  not_taken_course : ℕ

/-- The number of members who have taken the preparatory course but not passed the test -/
def members_taken_course_not_passed (club : SwimClub) : ℕ :=
  club.total_members - club.passed_test - club.not_taken_course

/-- Theorem stating the number of members who have taken the preparatory course
    but not passed the test in the given scenario -/
theorem swim_club_scenario :
  let club : SwimClub := {
    total_members := 60,
    passed_test := 18,  -- 30% of 60
    not_taken_course := 30
  }
  members_taken_course_not_passed club = 12 := by
  sorry

end NUMINAMATH_CALUDE_swim_club_scenario_l3827_382704


namespace NUMINAMATH_CALUDE_triangle_max_area_l3827_382788

theorem triangle_max_area (a b c : Real) (A B C : Real) :
  C = π / 6 →
  a + b = 12 →
  0 < a ∧ 0 < b ∧ 0 < c →
  (∃ (S : Real), S = (1 / 2) * a * b * Real.sin C ∧
    ∀ (S' : Real), S' = (1 / 2) * a * b * Real.sin C → S' ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3827_382788


namespace NUMINAMATH_CALUDE_map_distance_calculation_l3827_382709

/-- Given a map scale and actual distances, calculate the map distance --/
theorem map_distance_calculation 
  (map_distance_mountains : ℝ) 
  (actual_distance_mountains : ℝ) 
  (actual_distance_ram : ℝ) :
  let scale := actual_distance_mountains / map_distance_mountains
  actual_distance_ram / scale = map_distance_mountains * (actual_distance_ram / actual_distance_mountains) :=
by sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l3827_382709


namespace NUMINAMATH_CALUDE_inverse_in_S_l3827_382712

-- Define the set S
variable (S : Set ℝ)

-- Define the properties of S
variable (h1 : Set.Subset (Set.range (Int.cast : ℤ → ℝ)) S)
variable (h2 : (Real.sqrt 2 + Real.sqrt 3) ∈ S)
variable (h3 : ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S)
variable (h4 : ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S)

-- Theorem statement
theorem inverse_in_S : (Real.sqrt 2 + Real.sqrt 3)⁻¹ ∈ S := by
  sorry

end NUMINAMATH_CALUDE_inverse_in_S_l3827_382712


namespace NUMINAMATH_CALUDE_certain_number_multiplication_l3827_382754

theorem certain_number_multiplication (x : ℚ) : x / 11 = 2 → 6 * x = 132 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplication_l3827_382754


namespace NUMINAMATH_CALUDE_distinct_ages_count_l3827_382799

def average_age : ℕ := 31
def standard_deviation : ℕ := 5

def lower_bound : ℕ := average_age - standard_deviation
def upper_bound : ℕ := average_age + standard_deviation

theorem distinct_ages_count : 
  (Finset.range (upper_bound - lower_bound + 1)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_ages_count_l3827_382799


namespace NUMINAMATH_CALUDE_complex_number_location_l3827_382730

theorem complex_number_location (z : ℂ) :
  (z * (1 + Complex.I) = 3 - Complex.I) →
  (0 < z.re ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3827_382730


namespace NUMINAMATH_CALUDE_max_revenue_price_l3827_382773

/-- The revenue function for the toy shop -/
def revenue (p : ℝ) : ℝ := p * (100 - 4 * p)

/-- The theorem stating the price that maximizes revenue -/
theorem max_revenue_price :
  ∃ (p : ℝ), p ≤ 20 ∧ ∀ (q : ℝ), q ≤ 20 → revenue p ≥ revenue q ∧ p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_max_revenue_price_l3827_382773


namespace NUMINAMATH_CALUDE_product_5832_sum_62_l3827_382732

theorem product_5832_sum_62 : ∃ (a b c : ℕ+),
  (a.val > 1) ∧ (b.val > 1) ∧ (c.val > 1) ∧
  (a * b * c = 5832) ∧
  (Nat.gcd a.val b.val = 1) ∧ (Nat.gcd b.val c.val = 1) ∧ (Nat.gcd c.val a.val = 1) ∧
  (a + b + c = 62) := by
sorry

end NUMINAMATH_CALUDE_product_5832_sum_62_l3827_382732


namespace NUMINAMATH_CALUDE_sequence_monotonicity_l3827_382740

theorem sequence_monotonicity (b : ℝ) :
  (∀ n : ℕ, n^2 + b*n < (n+1)^2 + b*(n+1)) ↔ b > -3 :=
sorry

end NUMINAMATH_CALUDE_sequence_monotonicity_l3827_382740


namespace NUMINAMATH_CALUDE_three_intersections_l3827_382780

/-- The number of intersection points between a circle and a parabola -/
def intersection_count (b : ℝ) : ℕ :=
  -- Define the count based on the intersection points
  -- This is a placeholder; the actual implementation would involve solving the system of equations
  sorry

/-- Theorem stating the condition for exactly three intersection points -/
theorem three_intersections (b : ℝ) :
  intersection_count b = 3 ↔ b > (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_three_intersections_l3827_382780


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l3827_382715

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_values (m n : ℝ) :
  symmetric_wrt_y_axis (-m, 3) (-5, n) → m = -5 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l3827_382715


namespace NUMINAMATH_CALUDE_gcd_360_504_l3827_382743

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l3827_382743


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3827_382716

theorem arithmetic_calculations :
  ((-7 + 13 - 6 + 20 = 20) ∧
   (-2^3 + (2 - 3) - 2 * (-1)^2023 = -7)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3827_382716


namespace NUMINAMATH_CALUDE_namjoon_walk_proof_l3827_382724

/-- The additional distance Namjoon walked compared to his usual route -/
def additional_distance (usual_distance initial_walk : ℝ) : ℝ :=
  2 * initial_walk + usual_distance - usual_distance

theorem namjoon_walk_proof (usual_distance initial_walk : ℝ) 
  (h1 : usual_distance = 1.2)
  (h2 : initial_walk = 0.3) :
  additional_distance usual_distance initial_walk = 0.6 := by
  sorry

#eval additional_distance 1.2 0.3

end NUMINAMATH_CALUDE_namjoon_walk_proof_l3827_382724


namespace NUMINAMATH_CALUDE_polynomial_product_existence_l3827_382722

/-- Polynomial with integer coefficients and bounded absolute values -/
def BoundedPolynomial (n : ℕ) (bound : ℕ) := {p : Polynomial ℤ // ∀ i, i ≤ n → |p.coeff i| ≤ bound}

/-- The main theorem -/
theorem polynomial_product_existence 
  (f : BoundedPolynomial 5 4)
  (g : BoundedPolynomial 3 1)
  (h : BoundedPolynomial 2 1)
  (h1 : (f.val).eval 10 = (g.val).eval 10 * (h.val).eval 10) :
  ∃ (f' : Polynomial ℤ), ∀ x, f'.eval x = (g.val).eval x * (h.val).eval x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_existence_l3827_382722


namespace NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_product_nonzero_iff_both_nonzero_l3827_382746

-- Theorem for Option A
theorem abs_gt_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) :=
sorry

-- Theorem for Option C
theorem product_nonzero_iff_both_nonzero (a b : ℝ) :
  a * b ≠ 0 ↔ a ≠ 0 ∧ b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_product_nonzero_iff_both_nonzero_l3827_382746


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3827_382741

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 20)
  (h2 : added_water = 2)
  (h3 : final_percentage = 20)
  : ∃ initial_percentage : ℝ,
    initial_percentage * initial_volume / 100 + added_water =
    final_percentage * (initial_volume + added_water) / 100 ∧
    initial_percentage = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l3827_382741


namespace NUMINAMATH_CALUDE_range_of_m_m_value_for_diameter_l3827_382769

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x y, circle_eq x y m) → m < 37/4 :=
sorry

-- Define the condition for PQ being diameter of circle passing through origin
def pq_diameter_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁*x₂ + y₁*y₂ = 0

-- Theorem for the value of m when PQ is diameter of circle passing through origin
theorem m_value_for_diameter (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, 
    circle_eq x₁ y₁ m ∧ circle_eq x₂ y₂ m ∧
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    pq_diameter_through_origin x₁ y₁ x₂ y₂) →
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_m_value_for_diameter_l3827_382769


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3827_382735

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3827_382735


namespace NUMINAMATH_CALUDE_set_union_problem_l3827_382795

def M (a : ℕ) : Set ℕ := {3, 4^a}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem set_union_problem (a b : ℕ) :
  M a ∩ N a b = {1} → M a ∪ N a b = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l3827_382795


namespace NUMINAMATH_CALUDE_g_max_value_l3827_382753

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The maximum value of g(x) on the interval [0, 2] is 3 -/
theorem g_max_value : ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧ g c = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_max_value_l3827_382753


namespace NUMINAMATH_CALUDE_third_term_is_35_l3827_382705

/-- An arithmetic sequence with 6 terms -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (n : ℕ)
  (h_arithmetic : ∀ i j, i < n → j < n → a (i + 1) - a i = a (j + 1) - a j)
  (h_length : n = 6)
  (h_first : a 0 = 23)
  (h_last : a 5 = 47)

/-- The third term of the arithmetic sequence is 35 -/
theorem third_term_is_35 (seq : ArithmeticSequence) : seq.a 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_35_l3827_382705


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l3827_382768

/-- The number of books Mrs. Hilt read -/
def num_books : ℕ := 4

/-- The number of chapters in each book -/
def chapters_per_book : ℕ := 17

/-- The total number of chapters Mrs. Hilt read -/
def total_chapters : ℕ := num_books * chapters_per_book

theorem mrs_hilt_reading : total_chapters = 68 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l3827_382768


namespace NUMINAMATH_CALUDE_S_equals_T_l3827_382760

def S : Set ℤ := {x | ∃ n : ℤ, x = 2*n + 1}
def T : Set ℤ := {x | ∃ n : ℤ, x = 4*n + 1 ∨ x = 4*n - 1}

theorem S_equals_T : S = T := by sorry

end NUMINAMATH_CALUDE_S_equals_T_l3827_382760


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3827_382720

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x + 3 / (x^3) ≥ 12 :=
by sorry

theorem equality_condition : 9 * 1 + 3 / (1^3) = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3827_382720


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l3827_382718

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h_pos : x > 0) (h_eq : x + 1/x = 100) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 102 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l3827_382718
