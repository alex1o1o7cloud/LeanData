import Mathlib

namespace NUMINAMATH_CALUDE_caterpillar_insane_bill_sane_l3675_367505

-- Define the mental state of a character
inductive MentalState
| Sane
| Insane

-- Define the characters
structure Character where
  name : String
  state : MentalState

-- Define the Caterpillar's belief
def caterpillarBelief (caterpillar : Character) (bill : Character) : Prop :=
  caterpillar.state = MentalState.Insane ∧ bill.state = MentalState.Insane

-- Theorem statement
theorem caterpillar_insane_bill_sane 
  (caterpillar : Character) 
  (bill : Character) 
  (h : caterpillarBelief caterpillar bill) : 
  caterpillar.state = MentalState.Insane ∧ bill.state = MentalState.Sane :=
sorry

end NUMINAMATH_CALUDE_caterpillar_insane_bill_sane_l3675_367505


namespace NUMINAMATH_CALUDE_reginas_earnings_l3675_367524

/-- Represents Regina's farm and calculates her earnings -/
def ReginasFarm : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun num_cows num_pigs num_goats num_chickens num_rabbits
      cow_price pig_price goat_price chicken_price rabbit_price
      milk_income_per_cow rabbit_income_per_year maintenance_cost =>
    let total_animal_sale := num_cows * cow_price + num_pigs * pig_price +
                             num_goats * goat_price + num_chickens * chicken_price +
                             num_rabbits * rabbit_price
    let total_product_income := num_cows * milk_income_per_cow +
                                num_rabbits * rabbit_income_per_year
    total_animal_sale + total_product_income - maintenance_cost

/-- Theorem stating Regina's final earnings -/
theorem reginas_earnings :
  ReginasFarm 20 (4 * 20) ((4 * 20) / 2) (2 * 20) 30
               800 400 600 50 25
               500 10 10000 = 75050 := by
  sorry

end NUMINAMATH_CALUDE_reginas_earnings_l3675_367524


namespace NUMINAMATH_CALUDE_calculation_proof_l3675_367521

theorem calculation_proof :
  ((-20) - (-18) + 5 + (-9) = -6) ∧
  ((-3) * ((-1)^2003) - ((-4)^2) / (-2) = 11) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3675_367521


namespace NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l3675_367596

/-- The volume of a solid obtained by rotating an equilateral triangle around a line parallel to its altitude -/
theorem equilateral_triangle_rotation_volume (a : ℝ) (ha : a > 0) :
  let h := a * Real.sqrt 3 / 2
  let r := a / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.pi * a^3 * Real.sqrt 3) / 24 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l3675_367596


namespace NUMINAMATH_CALUDE_five_double_prime_value_l3675_367579

-- Define the prime operation
noncomputable def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem five_double_prime_value : prime (prime 5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_five_double_prime_value_l3675_367579


namespace NUMINAMATH_CALUDE_shaded_area_of_partitioned_triangle_l3675_367557

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_pos : leg_length > 0

/-- Represents a partition of a triangle -/
structure TrianglePartition where
  num_parts : ℕ
  num_parts_pos : num_parts > 0

theorem shaded_area_of_partitioned_triangle
  (t : IsoscelesRightTriangle)
  (p : TrianglePartition)
  (h1 : t.leg_length = 10)
  (h2 : p.num_parts = 25)
  (num_shaded : ℕ)
  (h3 : num_shaded = 15) :
  num_shaded * (t.leg_length^2 / 2) / p.num_parts = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_partitioned_triangle_l3675_367557


namespace NUMINAMATH_CALUDE_chess_tournament_attendees_l3675_367500

theorem chess_tournament_attendees (total_students : ℕ) 
  (h1 : total_students = 24) 
  (chess_program_fraction : ℚ) 
  (h2 : chess_program_fraction = 1 / 3) 
  (tournament_fraction : ℚ) 
  (h3 : tournament_fraction = 1 / 2) : ℕ :=
  by
    sorry

#check chess_tournament_attendees

end NUMINAMATH_CALUDE_chess_tournament_attendees_l3675_367500


namespace NUMINAMATH_CALUDE_solution_of_equation_l3675_367504

theorem solution_of_equation (x : ℝ) : 
  (3 / (x + 2) - 1 / x = 0) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3675_367504


namespace NUMINAMATH_CALUDE_product_minus_one_divisible_by_ten_l3675_367546

theorem product_minus_one_divisible_by_ten :
  ∃ k : ℤ, 11 * 21 * 31 * 41 * 51 - 1 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_minus_one_divisible_by_ten_l3675_367546


namespace NUMINAMATH_CALUDE_investment_ratio_is_two_to_three_l3675_367552

/-- A partnership problem with three investors A, B, and C. -/
structure Partnership where
  /-- B's investment amount -/
  b_investment : ℝ
  /-- Total profit earned -/
  total_profit : ℝ
  /-- B's share of the profit -/
  b_share : ℝ
  /-- A's investment is 3 times B's investment -/
  a_investment_prop : ℝ := 3 * b_investment
  /-- Assumption that total_profit and b_share are positive -/
  h_positive : 0 < total_profit ∧ 0 < b_share

/-- The ratio of B's investment to C's investment in the partnership -/
def investment_ratio (p : Partnership) : ℚ × ℚ :=
  (2, 3)

/-- Theorem stating that the investment ratio is 2:3 given the partnership conditions -/
theorem investment_ratio_is_two_to_three (p : Partnership)
  (h1 : p.total_profit = 3300)
  (h2 : p.b_share = 600) :
  investment_ratio p = (2, 3) := by
  sorry

#check investment_ratio_is_two_to_three

end NUMINAMATH_CALUDE_investment_ratio_is_two_to_three_l3675_367552


namespace NUMINAMATH_CALUDE_mark_bananas_equal_mike_matt_fruits_l3675_367569

/-- Represents the number of fruits each child received -/
structure FruitDistribution where
  mike_oranges : ℕ
  matt_apples : ℕ
  mark_bananas : ℕ

/-- The fruit distribution problem -/
def annie_fruit_problem (fd : FruitDistribution) : Prop :=
  fd.mike_oranges = 3 ∧
  fd.matt_apples = 2 * fd.mike_oranges ∧
  fd.mike_oranges + fd.matt_apples + fd.mark_bananas = 18

/-- The theorem stating the relationship between Mark's bananas and the total fruits of Mike and Matt -/
theorem mark_bananas_equal_mike_matt_fruits (fd : FruitDistribution) 
  (h : annie_fruit_problem fd) : 
  fd.mark_bananas = fd.mike_oranges + fd.matt_apples := by
  sorry

#check mark_bananas_equal_mike_matt_fruits

end NUMINAMATH_CALUDE_mark_bananas_equal_mike_matt_fruits_l3675_367569


namespace NUMINAMATH_CALUDE_parity_of_sum_of_powers_l3675_367580

theorem parity_of_sum_of_powers : Even (1^1994 + 9^1994 + 8^1994 + 6^1994) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_sum_of_powers_l3675_367580


namespace NUMINAMATH_CALUDE_second_investment_rate_l3675_367532

theorem second_investment_rate
  (total_investment : ℝ)
  (first_rate : ℝ)
  (first_amount : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 6000)
  (h2 : first_rate = 0.09)
  (h3 : first_amount = 1800)
  (h4 : total_interest = 624)
  : (total_interest - first_amount * first_rate) / (total_investment - first_amount) = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_rate_l3675_367532


namespace NUMINAMATH_CALUDE_eggs_per_basket_l3675_367594

theorem eggs_per_basket (purple_eggs blue_eggs min_eggs : ℕ) 
  (h1 : purple_eggs = 30)
  (h2 : blue_eggs = 42)
  (h3 : min_eggs = 5) : 
  ∃ (eggs_per_basket : ℕ), 
    eggs_per_basket ≥ min_eggs ∧ 
    purple_eggs % eggs_per_basket = 0 ∧
    blue_eggs % eggs_per_basket = 0 ∧
    eggs_per_basket = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l3675_367594


namespace NUMINAMATH_CALUDE_problem_solution_l3675_367529

theorem problem_solution : 
  ((-1)^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9) ∧ 
  (2 * ((3 : ℝ)^(1/2) - (2 : ℝ)^(1/2)) - ((2 : ℝ)^(1/2) + (3 : ℝ)^(1/2)) = (3 : ℝ)^(1/2) - 3 * (2 : ℝ)^(1/2)) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l3675_367529


namespace NUMINAMATH_CALUDE_first_group_size_l3675_367528

/-- The number of beavers in the first group -/
def first_group : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def time_first_group : ℕ := 3

/-- The number of beavers in the second group -/
def second_group : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def time_second_group : ℕ := 5

/-- Theorem stating that the first group consists of 20 beavers -/
theorem first_group_size :
  first_group * time_first_group = second_group * time_second_group :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l3675_367528


namespace NUMINAMATH_CALUDE_sticker_count_l3675_367599

/-- The total number of stickers Ryan, Steven, and Terry have altogether -/
def total_stickers (ryan_stickers : ℕ) (steven_multiplier : ℕ) (terry_extra : ℕ) : ℕ :=
  ryan_stickers + 
  (steven_multiplier * ryan_stickers) + 
  (steven_multiplier * ryan_stickers + terry_extra)

/-- Proof that the total number of stickers is 230 -/
theorem sticker_count : total_stickers 30 3 20 = 230 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l3675_367599


namespace NUMINAMATH_CALUDE_maximum_marks_l3675_367513

/-- 
Given:
1. The passing mark is 36% of the maximum marks.
2. A student gets 130 marks and fails by 14 marks.
Prove that the maximum number of marks is 400.
-/
theorem maximum_marks (passing_percentage : ℚ) (student_marks : ℕ) (failing_margin : ℕ) :
  passing_percentage = 36 / 100 →
  student_marks = 130 →
  failing_margin = 14 →
  ∃ (max_marks : ℕ), max_marks = 400 ∧ 
    (student_marks + failing_margin : ℚ) = passing_percentage * max_marks :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_l3675_367513


namespace NUMINAMATH_CALUDE_one_real_zero_l3675_367541

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- State the theorem
theorem one_real_zero : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_real_zero_l3675_367541


namespace NUMINAMATH_CALUDE_animal_jumping_distances_l3675_367527

-- Define the jumping distances for each animal
def grasshopper_jump : ℕ := 36

def frog_jump : ℕ := grasshopper_jump + 17

def mouse_jump : ℕ := frog_jump + 15

def kangaroo_jump : ℕ := 2 * mouse_jump

def rabbit_jump : ℕ := kangaroo_jump / 2 - 12

-- Theorem to prove the jumping distances
theorem animal_jumping_distances :
  grasshopper_jump = 36 ∧
  frog_jump = 53 ∧
  mouse_jump = 68 ∧
  kangaroo_jump = 136 ∧
  rabbit_jump = 56 := by
  sorry


end NUMINAMATH_CALUDE_animal_jumping_distances_l3675_367527


namespace NUMINAMATH_CALUDE_imaginary_complex_implies_modulus_l3675_367597

/-- Given a real number t, if the complex number z = (1-ti)/(1+i) is purely imaginary, 
    then |√3 + ti| = 2 -/
theorem imaginary_complex_implies_modulus (t : ℝ) : 
  let z : ℂ := (1 - t * Complex.I) / (1 + Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (Real.sqrt 3 + t * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_complex_implies_modulus_l3675_367597


namespace NUMINAMATH_CALUDE_min_candy_removal_l3675_367516

def candy_distribution (total : ℕ) (sisters : ℕ) : ℕ :=
  total - sisters * (total / sisters)

theorem min_candy_removal (total : ℕ) (sisters : ℕ) 
  (h1 : total = 24) (h2 : sisters = 5) : 
  candy_distribution total sisters = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_candy_removal_l3675_367516


namespace NUMINAMATH_CALUDE_quadratic_equation_special_roots_l3675_367537

/-- 
Given a quadratic equation x^2 + px + q = 0 with roots D and 1-D, 
where D is the discriminant of the equation, 
prove that the only possible values for (p, q) are (-1, 0) and (-1, 3/16).
-/
theorem quadratic_equation_special_roots (p q D : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = D ∨ x = 1 - D) ∧ 
  D^2 = p^2 - 4*q →
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3/16) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_special_roots_l3675_367537


namespace NUMINAMATH_CALUDE_absolute_value_of_c_l3675_367561

theorem absolute_value_of_c (a b c : ℤ) : 
  a * (3 + I : ℂ)^4 + b * (3 + I : ℂ)^3 + c * (3 + I : ℂ)^2 + b * (3 + I : ℂ) + a = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 116 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_of_c_l3675_367561


namespace NUMINAMATH_CALUDE_gcd_462_330_l3675_367562

theorem gcd_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end NUMINAMATH_CALUDE_gcd_462_330_l3675_367562


namespace NUMINAMATH_CALUDE_sphere_volume_l3675_367567

theorem sphere_volume (r : ℝ) (d V : ℝ) (h1 : r = 1/3) (h2 : d = 2*r) (h3 : d = (16/9 * V)^(1/3)) : V = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l3675_367567


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l3675_367571

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let x1 := p1.1
  let y1 := p1.2
  let x2 := p2.1
  let y2 := p2.2
  let x3 := p3.1
  let y3 := p3.2
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC : 
  let A : ℝ × ℝ := (5, 3)
  let B : ℝ × ℝ := reflect_y_axis A
  let C : ℝ × ℝ := reflect_y_eq_x B
  triangle_area A B C = 40 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l3675_367571


namespace NUMINAMATH_CALUDE_three_part_division_l3675_367573

theorem three_part_division (total : ℚ) (p1 p2 p3 : ℚ) (h1 : total = 78)
  (h2 : p1 + p2 + p3 = total) (h3 : p2 = (1/3) * p1) (h4 : p3 = (1/6) * p1) :
  p2 = 17 + (1/3) :=
by sorry

end NUMINAMATH_CALUDE_three_part_division_l3675_367573


namespace NUMINAMATH_CALUDE_triangle_special_case_l3675_367570

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_special_case (t : Triangle) 
  (h1 : t.A - t.C = π / 2)  -- A - C = 90°
  (h2 : t.a + t.c = Real.sqrt 2 * t.b)  -- a + c = √2 * b
  : t.C = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_case_l3675_367570


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3675_367538

theorem quadratic_factorization_sum (a b c d : ℤ) : 
  (∀ x, x^2 + 13*x + 40 = (x + a) * (x + b)) →
  (∀ x, x^2 - 19*x + 88 = (x - c) * (x - d)) →
  a + b + c + d = 32 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3675_367538


namespace NUMINAMATH_CALUDE_bag_weight_problem_l3675_367563

theorem bag_weight_problem (sugar_weight salt_weight removed_weight : ℕ) 
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : removed_weight = 4) :
  sugar_weight + salt_weight - removed_weight = 42 := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_problem_l3675_367563


namespace NUMINAMATH_CALUDE_incorrect_equation_l3675_367575

/-- A repeating decimal with non-repeating part N and repeating part R -/
structure RepeatingDecimal where
  N : ℕ  -- non-repeating part
  R : ℕ  -- repeating part
  t : ℕ  -- number of digits in N
  u : ℕ  -- number of digits in R
  t_pos : t > 0
  u_pos : u > 0

/-- The value of the repeating decimal -/
noncomputable def RepeatingDecimal.value (M : RepeatingDecimal) : ℝ :=
  (M.N : ℝ) / 10^M.t + (M.R : ℝ) / (10^M.t * (10^M.u - 1))

/-- The theorem stating that the equation in option D is incorrect -/
theorem incorrect_equation (M : RepeatingDecimal) :
  ¬(10^M.t * (10^M.u - 1) * M.value = (M.R : ℝ) * ((M.N : ℝ) - 1)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_l3675_367575


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l3675_367577

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^2 * 3^3) = 45 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l3675_367577


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_equals_two_l3675_367588

-- Define the point P
def P (a : ℤ) : ℝ × ℝ := (a - 1, a - 3)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_implies_a_equals_two (a : ℤ) :
  in_fourth_quadrant (P a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_equals_two_l3675_367588


namespace NUMINAMATH_CALUDE_subcommittee_count_l3675_367566

def total_members : ℕ := 12
def num_teachers : ℕ := 5
def subcommittee_size : ℕ := 5

def valid_subcommittees : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - num_teachers) subcommittee_size

theorem subcommittee_count :
  valid_subcommittees = 771 :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3675_367566


namespace NUMINAMATH_CALUDE_store_revenue_calculation_l3675_367555

/-- Represents the revenue calculation for Linda's store --/
def store_revenue (jean_price tee_price_low tee_price_high jacket_price jacket_discount tee_count_low tee_count_high jean_count jacket_count_regular jacket_count_discount sales_tax : ℚ) : ℚ :=
  let tee_revenue := tee_price_low * tee_count_low + tee_price_high * tee_count_high
  let jean_revenue := jean_price * jean_count
  let jacket_revenue_regular := jacket_price * jacket_count_regular
  let jacket_revenue_discount := jacket_price * (1 - jacket_discount) * jacket_count_discount
  let total_revenue := tee_revenue + jean_revenue + jacket_revenue_regular + jacket_revenue_discount
  let total_with_tax := total_revenue * (1 + sales_tax)
  total_with_tax

/-- Theorem stating that the store revenue matches the calculated amount --/
theorem store_revenue_calculation :
  store_revenue 22 15 20 37 0.1 4 3 4 2 3 0.07 = 408.63 :=
by sorry

end NUMINAMATH_CALUDE_store_revenue_calculation_l3675_367555


namespace NUMINAMATH_CALUDE_simplification_and_constant_coefficient_l3675_367560

-- Define the expression as a function of x and square
def expression (x : ℝ) (square : ℝ) : ℝ :=
  (square * x^2 + 6*x + 8) - (6*x + 5*x^2 + 2)

theorem simplification_and_constant_coefficient :
  (∀ x : ℝ, expression x 3 = -2 * x^2 + 6) ∧
  (∃! square : ℝ, ∀ x : ℝ, expression x square = (expression 0 square)) :=
by sorry

end NUMINAMATH_CALUDE_simplification_and_constant_coefficient_l3675_367560


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3675_367515

theorem quadratic_root_problem (a b c : ℝ) (h : a * (b - c) ≠ 0) :
  (∀ x, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0 ↔ x = 1 ∨ x = (c * (a - b)) / (a * (b - c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3675_367515


namespace NUMINAMATH_CALUDE_geometric_sequence_10th_term_l3675_367531

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_10th_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_4th : a 4 = 16)
  (h_7th : a 7 = 128) :
  a 10 = 1024 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_10th_term_l3675_367531


namespace NUMINAMATH_CALUDE_division_with_equal_quotient_and_remainder_l3675_367533

theorem division_with_equal_quotient_and_remainder :
  {N : ℕ | ∃ k : ℕ, 2014 = N * k + k ∧ k < N} = {2013, 1006, 105, 52} := by
  sorry

end NUMINAMATH_CALUDE_division_with_equal_quotient_and_remainder_l3675_367533


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3675_367585

/-- Given that i² = -1, prove that (2 - 3i) / (4 - 5i) = 23/41 - (2/41)i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 3 * i) / (4 - 5 * i) = 23 / 41 - (2 / 41) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3675_367585


namespace NUMINAMATH_CALUDE_pizza_combinations_l3675_367598

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3675_367598


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3675_367582

-- Define the given conditions
def initial_distance : ℝ := 300
def initial_fuel : ℝ := 10
def new_fuel : ℝ := 15

-- Define the theorem
theorem truck_travel_distance :
  (initial_distance / initial_fuel) * new_fuel = 450 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l3675_367582


namespace NUMINAMATH_CALUDE_janet_jasmine_shampoo_l3675_367523

/-- The amount of rose shampoo Janet has, in bottles -/
def rose_shampoo : ℚ := 1/3

/-- The amount of shampoo Janet uses per day, in bottles -/
def daily_usage : ℚ := 1/12

/-- The number of days Janet's shampoo will last -/
def days : ℕ := 7

/-- The total amount of shampoo Janet has, in bottles -/
def total_shampoo : ℚ := daily_usage * days

/-- The amount of jasmine shampoo Janet has, in bottles -/
def jasmine_shampoo : ℚ := total_shampoo - rose_shampoo

theorem janet_jasmine_shampoo : jasmine_shampoo = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_janet_jasmine_shampoo_l3675_367523


namespace NUMINAMATH_CALUDE_alice_chicken_amount_l3675_367535

/-- Represents the grocery items in Alice's cart -/
structure GroceryCart where
  lettuce : ℕ
  cherryTomatoes : ℕ
  sweetPotatoes : ℕ
  broccoli : ℕ
  brusselSprouts : ℕ

/-- Calculates the total cost of items in the cart excluding chicken -/
def cartCost (cart : GroceryCart) : ℚ :=
  3 + 2.5 + (0.75 * cart.sweetPotatoes) + (2 * cart.broccoli) + 2.5

/-- Theorem: Alice has 1.5 pounds of chicken in her cart -/
theorem alice_chicken_amount (cart : GroceryCart) 
  (h1 : cart.lettuce = 1)
  (h2 : cart.cherryTomatoes = 1)
  (h3 : cart.sweetPotatoes = 4)
  (h4 : cart.broccoli = 2)
  (h5 : cart.brusselSprouts = 1)
  (h6 : 35 - (cartCost cart) - 11 = 6 * chicken_amount) :
  chicken_amount = 1.5 := by
  sorry

#check alice_chicken_amount

end NUMINAMATH_CALUDE_alice_chicken_amount_l3675_367535


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3675_367586

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 3 * y^2 + 5) ≤ Real.sqrt 28 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 3 * y^2 + 5) = Real.sqrt 28 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3675_367586


namespace NUMINAMATH_CALUDE_carnival_earnings_example_l3675_367551

/-- Represents the earnings of a carnival snack booth over a period of days -/
def carnival_earnings (popcorn_sales : ℕ) (cotton_candy_multiplier : ℕ) (days : ℕ) (rent : ℕ) (ingredients_cost : ℕ) : ℕ :=
  let daily_total := popcorn_sales + popcorn_sales * cotton_candy_multiplier
  let total_revenue := daily_total * days
  let total_expenses := rent + ingredients_cost
  total_revenue - total_expenses

/-- Theorem stating that the carnival snack booth's earnings after expenses for 5 days is $895 -/
theorem carnival_earnings_example : carnival_earnings 50 3 5 30 75 = 895 := by
  sorry

end NUMINAMATH_CALUDE_carnival_earnings_example_l3675_367551


namespace NUMINAMATH_CALUDE_rectangle_area_l3675_367550

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 5 → length = 4 * width → width * length = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3675_367550


namespace NUMINAMATH_CALUDE_die_roll_counts_l3675_367539

/-- Represents the number of sides on a standard die -/
def dieSides : ℕ := 6

/-- Calculates the number of three-digit numbers with all distinct digits -/
def distinctDigits : ℕ := dieSides * (dieSides - 1) * (dieSides - 2)

/-- Calculates the total number of different three-digit numbers -/
def totalNumbers : ℕ := dieSides ^ 3

/-- Calculates the number of three-digit numbers with exactly two digits the same -/
def twoSameDigits : ℕ := 3 * dieSides * (dieSides - 1)

theorem die_roll_counts :
  distinctDigits = 120 ∧ totalNumbers = 216 ∧ twoSameDigits = 90 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_counts_l3675_367539


namespace NUMINAMATH_CALUDE_x_squared_plus_2x_is_quadratic_l3675_367526

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 2x = 0 is a quadratic equation -/
theorem x_squared_plus_2x_is_quadratic :
  is_quadratic_equation (λ x => x^2 + 2*x) :=
by
  sorry


end NUMINAMATH_CALUDE_x_squared_plus_2x_is_quadratic_l3675_367526


namespace NUMINAMATH_CALUDE_half_times_x_times_three_fourths_l3675_367517

theorem half_times_x_times_three_fourths (x : ℚ) : x = 5/6 → (1/2 : ℚ) * x * (3/4 : ℚ) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_half_times_x_times_three_fourths_l3675_367517


namespace NUMINAMATH_CALUDE_stone_blocks_per_step_l3675_367572

theorem stone_blocks_per_step 
  (levels : ℕ) 
  (steps_per_level : ℕ) 
  (total_blocks : ℕ) 
  (h1 : levels = 4) 
  (h2 : steps_per_level = 8) 
  (h3 : total_blocks = 96) : 
  total_blocks / (levels * steps_per_level) = 3 := by
sorry

end NUMINAMATH_CALUDE_stone_blocks_per_step_l3675_367572


namespace NUMINAMATH_CALUDE_no_equidistant_points_l3675_367556

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the parallel tangents
def ParallelTangents (O : ℝ × ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {P | |P.2 - O.2| = d}

-- Define a point equidistant from circle and tangents
def IsEquidistant (P : ℝ × ℝ) (O : ℝ × ℝ) (r d : ℝ) : Prop :=
  abs (((P.1 - O.1)^2 + (P.2 - O.2)^2).sqrt - r) = abs (|P.2 - O.2| - d)

theorem no_equidistant_points (O : ℝ × ℝ) (r d : ℝ) (h : d > r) :
  ¬∃P, IsEquidistant P O r d :=
by sorry

end NUMINAMATH_CALUDE_no_equidistant_points_l3675_367556


namespace NUMINAMATH_CALUDE_stratified_sampling_red_balls_l3675_367506

/-- Given a set of 100 balls with 20 red balls, prove that a stratified sample of 10 balls should contain 2 red balls. -/
theorem stratified_sampling_red_balls 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (sample_size : ℕ) 
  (h_total : total_balls = 100) 
  (h_red : red_balls = 20) 
  (h_sample : sample_size = 10) : 
  (red_balls : ℚ) / total_balls * sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_red_balls_l3675_367506


namespace NUMINAMATH_CALUDE_age_difference_l3675_367548

/-- Given that the total age of a and b is 13 years more than the total age of b and c,
    prove that c is 13 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 13) : a = c + 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3675_367548


namespace NUMINAMATH_CALUDE_line_constant_value_l3675_367512

theorem line_constant_value (m n p : ℝ) (h : p = 1/3) :
  ∃ C : ℝ, (m = 6*n + C ∧ m + 2 = 6*(n + p) + C) → C = 0 :=
sorry

end NUMINAMATH_CALUDE_line_constant_value_l3675_367512


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l3675_367514

theorem perfect_square_pairs (a b : ℤ) : 
  (∃ k : ℤ, a^2 + 4*b = k^2) ∧ (∃ m : ℤ, b^2 + 4*a = m^2) ↔ 
  (a = 0 ∧ b = 0) ∨ 
  (a = -4 ∧ b = -4) ∨ 
  (a = 4 ∧ b = -4) ∨ 
  (∃ k : ℕ, (a = k^2 ∧ b = 0) ∨ (a = 0 ∧ b = k^2)) ∨
  (a = -6 ∧ b = -5) ∨ 
  (a = -5 ∧ b = -6) ∨ 
  (∃ t : ℕ, (a = t ∧ b = 1 - t) ∨ (a = 1 - t ∧ b = t)) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l3675_367514


namespace NUMINAMATH_CALUDE_no_squarish_numbers_l3675_367503

/-- A number is squarish if it satisfies all the given conditions -/
def is_squarish (n : ℕ) : Prop :=
  -- Six-digit number
  100000 ≤ n ∧ n < 1000000 ∧
  -- Each digit between 1 and 8
  (∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 8) ∧
  -- Perfect square
  ∃ x, n = x^2 ∧
  -- First two digits are a perfect square
  ∃ y, (n / 10000) = y^2 ∧
  -- Middle two digits are a perfect square and divisible by 2
  ∃ z, ((n / 100) % 100) = z^2 ∧ ((n / 100) % 100) % 2 = 0 ∧
  -- Last two digits are a perfect square
  ∃ w, (n % 100) = w^2

theorem no_squarish_numbers : ¬∃ n, is_squarish n := by
  sorry

end NUMINAMATH_CALUDE_no_squarish_numbers_l3675_367503


namespace NUMINAMATH_CALUDE_exists_counterexample_for_option_c_l3675_367578

theorem exists_counterexample_for_option_c (h : ∃ a b : ℝ, a > b ∧ b > 0) :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ¬(a > Real.sqrt b) :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_for_option_c_l3675_367578


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l3675_367502

theorem least_x_for_even_prime_fraction (x p : ℕ) : 
  x > 0 → 
  Prime p → 
  Prime (x / (12 * p)) → 
  Even (x / (12 * p)) → 
  (∀ y : ℕ, y > 0 ∧ (∃ q : ℕ, Prime q ∧ Prime (y / (12 * q)) ∧ Even (y / (12 * q))) → x ≤ y) → 
  x = 48 := by
sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l3675_367502


namespace NUMINAMATH_CALUDE_total_spent_usd_value_l3675_367540

/-- The total amount spent on souvenirs in US dollars -/
def total_spent_usd (key_chain_bracelet_cost : ℝ) (tshirt_cost_diff : ℝ) 
  (tshirt_discount : ℝ) (key_chain_tax : ℝ) (bracelet_tax : ℝ) 
  (conversion_rate : ℝ) : ℝ :=
  let tshirt_cost := key_chain_bracelet_cost - tshirt_cost_diff
  let tshirt_actual := tshirt_cost * (1 - tshirt_discount)
  let key_chain_bracelet_actual := key_chain_bracelet_cost * (1 + key_chain_tax + bracelet_tax)
  (tshirt_actual + key_chain_bracelet_actual) * conversion_rate

/-- Theorem stating the total amount spent on souvenirs in US dollars -/
theorem total_spent_usd_value :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_spent_usd 347 146 0.1 0.12 0.08 0.75 - 447.98| < ε :=
sorry

end NUMINAMATH_CALUDE_total_spent_usd_value_l3675_367540


namespace NUMINAMATH_CALUDE_min_value_2a6_plus_a5_l3675_367583

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ a 1 > 0 ∧ ∀ n, a (n + 1) = q * a n

/-- The theorem stating the minimum value of 2a_6 + a_5 for a specific geometric sequence -/
theorem min_value_2a6_plus_a5 (a : ℕ → ℝ) :
  PositiveGeometricSequence a →
  (2 * a 4 + a 3 = 2 * a 2 + a 1 + 8) →
  (∀ x, 2 * a 6 + a 5 ≥ x) →
  x = 32 := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a6_plus_a5_l3675_367583


namespace NUMINAMATH_CALUDE_sector_central_angle_l3675_367508

/-- Given a sector with circumference 12 and area 8, its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r : ℝ) (l : ℝ) (α : ℝ) : 
  l + 2 * r = 12 → 
  1 / 2 * l * r = 8 → 
  α = l / r → 
  α = 1 ∨ α = 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3675_367508


namespace NUMINAMATH_CALUDE_min_coin_tosses_l3675_367568

theorem min_coin_tosses (n : ℕ) : (1 - (1/2)^n ≥ 15/16) ↔ n ≥ 4 := by sorry

end NUMINAMATH_CALUDE_min_coin_tosses_l3675_367568


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3675_367554

/-- An ellipse passing through two given points with a focus on a coordinate axis. -/
structure Ellipse where
  -- The coefficients of the ellipse equation x²/a² + y²/b² = 1
  a : ℝ
  b : ℝ
  -- Condition: a > 0 and b > 0
  ha : a > 0
  hb : b > 0
  -- Condition: Passes through P1(-√6, 1)
  passes_p1 : 6 / a^2 + 1 / b^2 = 1
  -- Condition: Passes through P2(√3, -√2)
  passes_p2 : 3 / a^2 + 2 / b^2 = 1
  -- Condition: One focus on coordinate axis, perpendicular to minor axis vertices, passes through (-3, 3√2/2)
  focus_condition : 9 / a^2 + (9/2) / b^2 = 1

/-- The standard equation of the ellipse satisfies one of the given forms. -/
theorem ellipse_standard_equation (e : Ellipse) : 
  (e.a^2 = 9 ∧ e.b^2 = 3) ∨ 
  (e.a^2 = 18 ∧ e.b^2 = 9) ∨ 
  (e.a^2 = 45/4 ∧ e.b^2 = 45/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3675_367554


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l3675_367519

/-- Partnership investment problem -/
theorem partnership_investment_ratio :
  ∀ (x : ℝ) (m : ℝ),
  x > 0 →  -- A's investment is positive
  (12 * x) / (12 * x + 12 * x + 4 * m * x) = 1/3 →  -- A's share proportion
  m = 3 :=  -- Ratio of C's investment to A's
by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l3675_367519


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l3675_367565

theorem arithmetic_sequence_average (a₁ aₙ d : ℚ) (n : ℕ) (h₁ : a₁ = 15) (h₂ : aₙ = 35) (h₃ : d = 1/4) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  (n * (a₁ + aₙ)) / (2 * n) = 25 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l3675_367565


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3675_367581

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  let s := d / Real.sqrt 2
  s * s = 72 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3675_367581


namespace NUMINAMATH_CALUDE_video_votes_l3675_367574

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 3/4 ∧ 
  (∀ (total_votes : ℕ), 
    (↑total_votes : ℚ) * like_percentage - (↑total_votes : ℚ) * (1 - like_percentage) = score) →
  ∃ (total_votes : ℕ), total_votes = 240 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l3675_367574


namespace NUMINAMATH_CALUDE_square_land_area_l3675_367545

/-- The area of a square land plot with side length 25 units is 625 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 25) : side_length ^ 2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l3675_367545


namespace NUMINAMATH_CALUDE_sum_first_ten_natural_numbers_l3675_367518

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 10 natural numbers is 55 -/
theorem sum_first_ten_natural_numbers : triangular_number 10 = 55 := by
  sorry

#eval triangular_number 10  -- This should output 55

end NUMINAMATH_CALUDE_sum_first_ten_natural_numbers_l3675_367518


namespace NUMINAMATH_CALUDE_monotonicity_of_f_l3675_367558

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

theorem monotonicity_of_f (a : ℝ) :
  (∀ x y, x < 1 → y < 1 → x < y → f a x > f a y) ∧ 
  (∀ x y, x > 1 → y > 1 → x < y → f a x < f a y) ∨
  (a = -Real.exp 1 / 2 ∧ ∀ x y, x < y → f a x < f a y) ∨
  (a < -Real.exp 1 / 2 ∧ 
    (∀ x y, x < 1 → y < 1 → x < y → f a x < f a y) ∧
    (∀ x y, 1 < x → x < Real.log (-2*a) → 1 < y → y < Real.log (-2*a) → x < y → f a x > f a y) ∧
    (∀ x y, x > Real.log (-2*a) → y > Real.log (-2*a) → x < y → f a x < f a y)) ∨
  (-Real.exp 1 / 2 < a ∧ a < 0 ∧
    (∀ x y, x < Real.log (-2*a) → y < Real.log (-2*a) → x < y → f a x < f a y) ∧
    (∀ x y, Real.log (-2*a) < x → x < 1 → Real.log (-2*a) < y → y < 1 → x < y → f a x > f a y) ∧
    (∀ x y, x > 1 → y > 1 → x < y → f a x < f a y)) :=
sorry

end

end NUMINAMATH_CALUDE_monotonicity_of_f_l3675_367558


namespace NUMINAMATH_CALUDE_bond_coupon_income_is_135_l3675_367530

/-- Represents a bond with its characteristics -/
structure Bond where
  purchase_price : ℝ
  face_value : ℝ
  current_yield : ℝ
  duration : ℕ

/-- Calculates the annual coupon income for a given bond -/
def annual_coupon_income (b : Bond) : ℝ :=
  b.current_yield * b.purchase_price

/-- Theorem stating that for the given bond, the annual coupon income is 135 rubles -/
theorem bond_coupon_income_is_135 (b : Bond) 
  (h1 : b.purchase_price = 900)
  (h2 : b.face_value = 1000)
  (h3 : b.current_yield = 0.15)
  (h4 : b.duration = 3) :
  annual_coupon_income b = 135 := by
  sorry

end NUMINAMATH_CALUDE_bond_coupon_income_is_135_l3675_367530


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3675_367589

theorem fourth_root_equation_solutions :
  {x : ℝ | x > 0 ∧ (x^(1/4) : ℝ) = 15 / (8 - (x^(1/4) : ℝ))} = {625, 81} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3675_367589


namespace NUMINAMATH_CALUDE_complete_square_sum_l3675_367507

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (49 * x^2 + 70 * x - 81 = 0 ↔ (a * x + b)^2 = c) ∧ 
  a > 0 ∧ 
  a + b + c = -44 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3675_367507


namespace NUMINAMATH_CALUDE_inverse_proportionality_l3675_367536

theorem inverse_proportionality (X Y K : ℝ) (h1 : XY = K - 1) (h2 : K > 1) :
  ∃ c : ℝ, ∀ x y : ℝ, (x ≠ 0 ∧ y ≠ 0) → (X = x ∧ Y = y) → x * y = c :=
sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l3675_367536


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l3675_367576

-- Define the pie as 100%
def whole_pie : ℚ := 1

-- Carlos's share
def carlos_share : ℚ := 0.6

-- Maria takes half of the remainder
def maria_share_ratio : ℚ := 1/2

-- Theorem to prove
theorem remaining_pie_portion : 
  let remainder_after_carlos := whole_pie - carlos_share
  let maria_share := maria_share_ratio * remainder_after_carlos
  let final_remainder := remainder_after_carlos - maria_share
  final_remainder = 0.2 := by
sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l3675_367576


namespace NUMINAMATH_CALUDE_min_cuboid_height_l3675_367549

/-- Represents a cuboid with a square base -/
structure Cuboid where
  base_side : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- The minimum height of a cuboid that can contain given spheres -/
def min_height (base_side : ℝ) (small_spheres : List Sphere) (large_sphere : Sphere) : ℝ :=
  sorry

theorem min_cuboid_height :
  let cuboid : Cuboid := { base_side := 4, height := min_height 4 (List.replicate 8 { radius := 1 }) { radius := 2 } }
  let small_spheres : List Sphere := List.replicate 8 { radius := 1 }
  let large_sphere : Sphere := { radius := 2 }
  cuboid.height = 2 + 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_min_cuboid_height_l3675_367549


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3675_367590

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 4) :
  ∃ d : ℝ, d = -1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3675_367590


namespace NUMINAMATH_CALUDE_charity_distribution_l3675_367592

theorem charity_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) : 
  total_amount = 2500 →
  donation_percentage = 0.80 →
  num_organizations = 8 →
  (total_amount * donation_percentage) / num_organizations = 250 := by
sorry

end NUMINAMATH_CALUDE_charity_distribution_l3675_367592


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3675_367544

/-- A geometric progression with sum to infinity 8 and sum of first three terms 7 has first term 4 -/
theorem geometric_progression_first_term :
  ∀ (a r : ℝ),
  (a / (1 - r) = 8) →  -- sum to infinity
  (a + a*r + a*r^2 = 7) →  -- sum of first three terms
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3675_367544


namespace NUMINAMATH_CALUDE_score_difference_proof_l3675_367534

def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (85, 0.25),
  (90, 0.10),
  (100, 0.30)
]

def mean_score : ℝ := (score_distribution.map (fun (score, percent) => score * percent)).sum

def median_score : ℝ := 85

theorem score_difference_proof :
  mean_score - median_score = -0.75 := by sorry

end NUMINAMATH_CALUDE_score_difference_proof_l3675_367534


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3675_367522

def U : Finset Nat := {1, 2, 3, 4}
def A : Finset Nat := {1, 4}
def B : Finset Nat := {2, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3675_367522


namespace NUMINAMATH_CALUDE_simplify_expression_l3675_367520

theorem simplify_expression (a b : ℝ) : (2*a - b) - 2*(a - 2*b) = 3*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3675_367520


namespace NUMINAMATH_CALUDE_dice_cube_volume_l3675_367510

/-- The volume of a cube formed by stacking dice --/
theorem dice_cube_volume 
  (num_dice : ℕ) 
  (die_edge : ℝ) 
  (h1 : num_dice = 125) 
  (h2 : die_edge = 2) 
  (h3 : ∃ n : ℕ, n ^ 3 = num_dice) : 
  (die_edge * (num_dice : ℝ) ^ (1/3 : ℝ)) ^ 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_dice_cube_volume_l3675_367510


namespace NUMINAMATH_CALUDE_legs_on_queen_mary_ii_l3675_367587

/-- Calculates the total number of legs on a ship with cats and humans. -/
def total_legs (total_heads : ℕ) (num_cats : ℕ) : ℕ :=
  let num_humans := total_heads - num_cats
  let cat_legs := num_cats * 4
  let human_legs := (num_humans - 1) * 2 + 1
  cat_legs + human_legs

/-- Theorem stating that the total number of legs is 45 under given conditions. -/
theorem legs_on_queen_mary_ii :
  total_legs 16 7 = 45 := by
  sorry

end NUMINAMATH_CALUDE_legs_on_queen_mary_ii_l3675_367587


namespace NUMINAMATH_CALUDE_max_sundays_in_fifty_days_l3675_367595

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of days we're considering -/
def daysConsidered : ℕ := 50

/-- The maximum number of Sundays in the first 50 days of any year -/
def maxSundays : ℕ := daysConsidered / daysInWeek

theorem max_sundays_in_fifty_days :
  maxSundays = 7 := by sorry

end NUMINAMATH_CALUDE_max_sundays_in_fifty_days_l3675_367595


namespace NUMINAMATH_CALUDE_expression_evaluation_l3675_367501

theorem expression_evaluation : -1^4 - (1/6) * (|(-2)| - (-3)^2) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3675_367501


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3675_367509

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3675_367509


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3675_367525

-- Define the function representing the sum of distances
def f (x : ℝ) : ℝ := |x - 5| + |x + 1|

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 8} = Set.Ioo (-2 : ℝ) (6 : ℝ) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3675_367525


namespace NUMINAMATH_CALUDE_product_mod_nine_l3675_367543

theorem product_mod_nine : (98 * 102) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_nine_l3675_367543


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_105_l3675_367511

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive positive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

theorem largest_consecutive_sum_105 :
  (∃ (a : ℕ), a > 0 ∧ sum_consecutive a 14 = 105) ∧
  (∀ (n : ℕ), n > 14 → ¬∃ (a : ℕ), a > 0 ∧ sum_consecutive a n = 105) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_105_l3675_367511


namespace NUMINAMATH_CALUDE_triangle_properties_l3675_367564

/-- Given a triangle ABC with specific properties, prove its angle B and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b * Real.cos C = (2 * a - c) * Real.cos B →
  b = Real.sqrt 7 →
  a + c = 4 →
  B = π / 3 ∧ 
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3675_367564


namespace NUMINAMATH_CALUDE_quadratic_min_max_l3675_367542

theorem quadratic_min_max (x : ℝ) (n : ℝ) :
  (∀ x, x^2 - 4*x - 3 ≥ -7) ∧
  (n = 6 - x → ∀ x, Real.sqrt (x^2 - 2*n^2) ≤ 6 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_min_max_l3675_367542


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l3675_367553

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l3675_367553


namespace NUMINAMATH_CALUDE_unique_solution_system_l3675_367591

theorem unique_solution_system (x y z : ℝ) : 
  (Real.sqrt (x - 997) + Real.sqrt (y - 932) + Real.sqrt (z - 796) = 100) ∧
  (Real.sqrt (x - 1237) + Real.sqrt (y - 1121) + Real.sqrt (3045 - z) = 90) ∧
  (Real.sqrt (x - 1621) + Real.sqrt (2805 - y) + Real.sqrt (z - 997) = 80) ∧
  (Real.sqrt (2102 - x) + Real.sqrt (y - 1237) + Real.sqrt (z - 932) = 70) →
  x = 2021 ∧ y = 2021 ∧ z = 2021 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3675_367591


namespace NUMINAMATH_CALUDE_max_sum_of_three_numbers_l3675_367547

theorem max_sum_of_three_numbers (a b c : ℕ) : 
  a + b = 1014 → c - b = 497 → a > b → (∀ S : ℕ, S = a + b + c → S ≤ 2017) ∧ (∃ S : ℕ, S = a + b + c ∧ S = 2017) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_numbers_l3675_367547


namespace NUMINAMATH_CALUDE_x_value_proof_l3675_367584

theorem x_value_proof (x : ℚ) 
  (h1 : 8 * x^2 + 9 * x - 2 = 0) 
  (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3675_367584


namespace NUMINAMATH_CALUDE_counterexample_disproves_conjecture_l3675_367559

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def isPrime (p : ℤ) : Prop := p > 1 ∧ ∀ m : ℤ, m > 1 → m < p → ¬(p % m = 0)

def isSumOfThreePrimes (n : ℤ) : Prop :=
  ∃ p q r : ℤ, isPrime p ∧ isPrime q ∧ isPrime r ∧ n = p + q + r

theorem counterexample_disproves_conjecture :
  ∃ n : ℤ, n > 5 ∧ isOdd n ∧ ¬(isSumOfThreePrimes n) →
  ¬(∀ m : ℤ, m > 5 → isOdd m → isSumOfThreePrimes m) :=
sorry

end NUMINAMATH_CALUDE_counterexample_disproves_conjecture_l3675_367559


namespace NUMINAMATH_CALUDE_money_split_ratio_l3675_367593

theorem money_split_ratio (parker_share richie_share total : ℚ) : 
  parker_share / richie_share = 2 / 3 →
  parker_share = 50 →
  parker_share < richie_share →
  total = parker_share + richie_share →
  total = 125 := by
sorry

end NUMINAMATH_CALUDE_money_split_ratio_l3675_367593
