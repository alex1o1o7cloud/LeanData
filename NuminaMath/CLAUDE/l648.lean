import Mathlib

namespace palace_number_puzzle_l648_64888

theorem palace_number_puzzle :
  ∀ (x : ℕ),
    x < 15 →
    (15 - x) + (15 + x) = 30 →
    (15 + x) - (15 - x) = 2 * x →
    2 * x * 30 = 780 →
    x = 13 :=
by
  sorry

end palace_number_puzzle_l648_64888


namespace max_value_of_expression_l648_64805

theorem max_value_of_expression (x : ℝ) : 
  x^4 / (x^8 + 4*x^6 + 2*x^4 + 8*x^2 + 16) ≤ 1/31 ∧ 
  ∃ y : ℝ, y^4 / (y^8 + 4*y^6 + 2*y^4 + 8*y^2 + 16) = 1/31 :=
by sorry

end max_value_of_expression_l648_64805


namespace coffee_price_correct_l648_64835

/-- The price of a cup of coffee satisfying the given conditions -/
def coffee_price : ℝ := 6

/-- The price of a piece of cheesecake -/
def cheesecake_price : ℝ := 10

/-- The discount rate applied to the set of coffee and cheesecake -/
def discount_rate : ℝ := 0.25

/-- The final price of the set (coffee + cheesecake) with discount -/
def discounted_set_price : ℝ := 12

/-- Theorem stating that the coffee price satisfies the given conditions -/
theorem coffee_price_correct :
  (1 - discount_rate) * (coffee_price + cheesecake_price) = discounted_set_price := by
  sorry

end coffee_price_correct_l648_64835


namespace a_plus_b_equals_34_l648_64839

theorem a_plus_b_equals_34 (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 30) / (x - 3)) →
  A + B = 34 := by
sorry

end a_plus_b_equals_34_l648_64839


namespace sqrt_equation_solution_l648_64833

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 11) = 14 → x = 37 := by
  sorry

end sqrt_equation_solution_l648_64833


namespace bread_distribution_problem_l648_64852

theorem bread_distribution_problem :
  ∃! (m w c : ℕ),
    m + w + c = 12 ∧
    2 * m + (1/2) * w + (1/4) * c = 12 ∧
    m ≥ 0 ∧ w ≥ 0 ∧ c ≥ 0 ∧
    m = 5 ∧ w = 1 ∧ c = 6 :=
by sorry

end bread_distribution_problem_l648_64852


namespace calculation_proof_l648_64834

theorem calculation_proof : 8 - (7.14 * (1/3) - 2 * (2/9) / 2.5) + 0.1 = 6.62 := by
  sorry

end calculation_proof_l648_64834


namespace no_integer_roots_for_primes_l648_64895

theorem no_integer_roots_for_primes (p q : ℕ) : 
  Prime p → Prime q → ¬∃ (x : ℤ), x^2 + 3*p*x + 5*q = 0 := by
  sorry

end no_integer_roots_for_primes_l648_64895


namespace bus_capacity_l648_64802

/-- A bus with seats on both sides and a back seat -/
structure Bus where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  back_seat_capacity : Nat

/-- Calculate the total number of people that can sit in the bus -/
def total_capacity (b : Bus) : Nat :=
  (b.left_seats + b.right_seats) * b.people_per_seat + b.back_seat_capacity

/-- Theorem stating the total capacity of the bus -/
theorem bus_capacity :
  ∃ (b : Bus),
    b.left_seats = 15 ∧
    b.right_seats = b.left_seats - 3 ∧
    b.people_per_seat = 3 ∧
    b.back_seat_capacity = 7 ∧
    total_capacity b = 88 := by
  sorry

end bus_capacity_l648_64802


namespace salary_change_percentage_loss_l648_64863

theorem salary_change (original : ℝ) (h : original > 0) :
  let decreased := original * (1 - 0.5)
  let final := decreased * (1 + 0.5)
  final = original * 0.75 :=
by
  sorry

theorem percentage_loss : 
  1 - 0.75 = 0.25 :=
by
  sorry

end salary_change_percentage_loss_l648_64863


namespace shopkeeper_articles_sold_l648_64823

/-- Proves that the number of articles sold is 30, given the selling price and profit conditions -/
theorem shopkeeper_articles_sold (C : ℝ) (C_pos : C > 0) : 
  ∃ N : ℕ, 
    (35 : ℝ) * C = (N : ℝ) * C + (1 / 6 : ℝ) * ((N : ℝ) * C) ∧ 
    N = 30 := by
  sorry

end shopkeeper_articles_sold_l648_64823


namespace compound_interest_problem_l648_64877

theorem compound_interest_problem (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8820)
  (h2 : P * (1 + r)^3 = 9261) : 
  P = 8000 := by
  sorry

end compound_interest_problem_l648_64877


namespace female_students_count_l648_64891

theorem female_students_count (x : ℕ) : 
  (8 * x < 200) → 
  (9 * x > 200) → 
  (11 * (x + 4) > 300) → 
  x = 24 := by
sorry

end female_students_count_l648_64891


namespace pineapple_problem_l648_64807

theorem pineapple_problem (pineapple_cost : ℕ) (rings_per_pineapple : ℕ) 
  (rings_per_sale : ℕ) (sale_price : ℕ) (total_profit : ℕ) :
  pineapple_cost = 3 →
  rings_per_pineapple = 12 →
  rings_per_sale = 4 →
  sale_price = 5 →
  total_profit = 72 →
  ∃ (num_pineapples : ℕ),
    num_pineapples * (rings_per_pineapple / rings_per_sale * sale_price - pineapple_cost) = total_profit ∧
    num_pineapples = 6 := by
  sorry

end pineapple_problem_l648_64807


namespace sandra_theorem_l648_64878

def sandra_problem (savings : ℚ) (mother_gift : ℚ) (father_gift_multiplier : ℚ)
  (candy_cost : ℚ) (jelly_bean_cost : ℚ) (candy_count : ℕ) (jelly_bean_count : ℕ) : Prop :=
  let total_money := savings + mother_gift + (father_gift_multiplier * mother_gift)
  let total_cost := (candy_cost * candy_count) + (jelly_bean_cost * jelly_bean_count)
  let remaining_money := total_money - total_cost
  remaining_money = 11

theorem sandra_theorem :
  sandra_problem 10 4 2 (1/2) (1/5) 14 20 := by
  sorry

end sandra_theorem_l648_64878


namespace intersection_of_M_and_N_l648_64853

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end intersection_of_M_and_N_l648_64853


namespace first_nonzero_digit_of_1_over_127_l648_64810

theorem first_nonzero_digit_of_1_over_127 :
  ∃ (n : ℕ), n > 0 ∧ (1000 : ℚ) / 127 = 7 + n / 127 ∧ n < 127 :=
by sorry

end first_nonzero_digit_of_1_over_127_l648_64810


namespace arithmetic_mean_problem_l648_64871

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b) / 2 = 115 →
  (b + c) / 2 = 160 →
  (b + d) / 2 = 175 →
  a - d = -120 := by
sorry

end arithmetic_mean_problem_l648_64871


namespace union_of_intervals_l648_64850

open Set

theorem union_of_intervals (A B : Set ℝ) :
  A = {x : ℝ | -1 < x ∧ x < 4} →
  B = {x : ℝ | 2 < x ∧ x < 5} →
  A ∪ B = {x : ℝ | -1 < x ∧ x < 5} := by
  sorry

end union_of_intervals_l648_64850


namespace intersection_of_A_and_B_l648_64803

def A : Set ℝ := {x : ℝ | |x| ≤ 2}
def B : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l648_64803


namespace harriet_siblings_product_l648_64883

/-- Represents a family with a specific structure -/
structure Family where
  harry_sisters : Nat
  harry_brothers : Nat

/-- Calculates the number of Harriet's sisters (excluding herself) -/
def harriet_sisters (f : Family) : Nat :=
  f.harry_sisters - 1

/-- Calculates the number of Harriet's brothers -/
def harriet_brothers (f : Family) : Nat :=
  f.harry_brothers

/-- Theorem stating that the product of Harriet's siblings is 9 -/
theorem harriet_siblings_product (f : Family) 
  (h1 : f.harry_sisters = 4) 
  (h2 : f.harry_brothers = 3) : 
  (harriet_sisters f) * (harriet_brothers f) = 9 := by
  sorry


end harriet_siblings_product_l648_64883


namespace melanie_dimes_l648_64826

theorem melanie_dimes (initial_dimes : ℕ) : 
  (initial_dimes - 7 + 4 = 5) → initial_dimes = 8 := by
  sorry

end melanie_dimes_l648_64826


namespace part_one_part_two_l648_64890

-- Define the set M
def M (D : Set ℝ) : Set (ℝ → ℝ) :=
  {f | ∀ x y, (x + y) / 2 ∈ D → f ((x + y) / 2) ≥ (f x + f y) / 2 ∧
       (f ((x + y) / 2) = (f x + f y) / 2 ↔ x = y)}

-- Part 1
theorem part_one (f : ℝ → ℝ) (h : f ∈ M (Set.Ioi 0)) :
  f 3 + f 5 ≤ 2 * f 4 := by sorry

-- Part 2
def g : ℝ → ℝ := λ x ↦ -x^2

theorem part_two : g ∈ M Set.univ := by sorry

end part_one_part_two_l648_64890


namespace estimate_cube_of_331_l648_64820

/-- Proves that (.331)^3 is approximately equal to 0.037, given that .331 is close to 1/3 -/
theorem estimate_cube_of_331 (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |0.331 - (1/3)| < δ → |0.331^3 - 0.037| < ε :=
sorry

end estimate_cube_of_331_l648_64820


namespace range_of_a_plus_3b_l648_64816

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : 1 ≤ a - 2*b ∧ a - 2*b ≤ 3) : 
  -11/3 ≤ a + 3*b ∧ a + 3*b ≤ 7/3 := by
  sorry

end range_of_a_plus_3b_l648_64816


namespace complex_minimum_value_l648_64801

theorem complex_minimum_value (w : ℂ) (h : Complex.abs (w - (3 - 3•I)) = 4) :
  Complex.abs (w + (2 - I))^2 + Complex.abs (w - (7 - 2•I))^2 = 66 := by
  sorry

end complex_minimum_value_l648_64801


namespace line_parallel_theorem_l648_64851

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a line in 3D space -/
structure Line

/-- Defines when a line is contained in a plane -/
def Line.containedIn (l : Line) (p : Plane) : Prop :=
  sorry

/-- Defines when a line is parallel to a plane -/
def Line.parallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Defines when two lines are coplanar -/
def Line.coplanar (l1 l2 : Line) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem: If m is contained in plane a, n is parallel to plane a,
    and m and n are coplanar, then m is parallel to n -/
theorem line_parallel_theorem (a : Plane) (m n : Line) :
  m.containedIn a → n.parallelToPlane a → m.coplanar n → m.parallel n :=
by sorry

end line_parallel_theorem_l648_64851


namespace triangle_x_coordinate_l648_64830

/-- 
Given a triangle with vertices (x, 0), (7, 4), and (7, -4),
if the area of the triangle is 32, then x = -1.
-/
theorem triangle_x_coordinate (x : ℝ) : 
  let v1 : ℝ × ℝ := (x, 0)
  let v2 : ℝ × ℝ := (7, 4)
  let v3 : ℝ × ℝ := (7, -4)
  let base : ℝ := |v2.2 - v3.2|
  let height : ℝ := |7 - x|
  let area : ℝ := (1/2) * base * height
  area = 32 → x = -1 := by
sorry

end triangle_x_coordinate_l648_64830


namespace james_age_l648_64846

/-- Proves that James' current age is 11 years old, given the conditions of the problem. -/
theorem james_age (julio_age : ℕ) (years_later : ℕ) (james_age : ℕ) : 
  julio_age = 36 →
  years_later = 14 →
  julio_age + years_later = 2 * (james_age + years_later) →
  james_age = 11 := by
sorry

end james_age_l648_64846


namespace vector_relations_l648_64822

/-- Given plane vectors a, b, and c, prove parallel and perpendicular conditions. -/
theorem vector_relations (a b c : ℝ × ℝ) (t : ℝ) 
  (ha : a = (-2, 1)) 
  (hb : b = (4, 2)) 
  (hc : c = (2, t)) : 
  (∃ (k : ℝ), a = k • c → t = -1) ∧ 
  (b.1 * c.1 + b.2 * c.2 = 0 → t = -4) := by
  sorry


end vector_relations_l648_64822


namespace equation_equivalence_l648_64812

theorem equation_equivalence (a c x y : ℤ) (m n p : ℕ) : 
  (a^9*x*y - a^8*y - a^7*x = a^6*(c^3 - 1)) →
  ((a^m*x - a^n)*(a^p*y - a^3) = a^6*c^3) →
  m*n*p = 90 := by sorry

end equation_equivalence_l648_64812


namespace arithmetic_sequence_sum_l648_64829

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions,
    prove that the sum of its third and fourth terms is 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h1 : isArithmeticSequence a) 
    (h2 : a 1 + a 2 = 10) 
    (h3 : a 4 = a 3 + 2) : 
  a 3 + a 4 = 18 := by
sorry

end arithmetic_sequence_sum_l648_64829


namespace infinite_points_on_line_l648_64836

/-- A point on the line x + y = 4 with positive rational coordinates -/
structure PointOnLine where
  x : ℚ
  y : ℚ
  x_pos : 0 < x
  y_pos : 0 < y
  on_line : x + y = 4

/-- The set of all points on the line x + y = 4 with positive rational coordinates -/
def PointsOnLine : Set PointOnLine :=
  {p : PointOnLine | True}

/-- Theorem: There are infinitely many points on the line x + y = 4 with positive rational coordinates -/
theorem infinite_points_on_line : Set.Infinite PointsOnLine := by
  sorry

end infinite_points_on_line_l648_64836


namespace coloring_properties_l648_64886

/-- A coloring of natural numbers with N colors. -/
def Coloring (N : ℕ) := ℕ → Fin N

/-- Property that there are infinitely many numbers of each color. -/
def InfinitelyMany (c : Coloring N) : Prop :=
  ∀ (k : Fin N), ∀ (m : ℕ), ∃ (n : ℕ), n > m ∧ c n = k

/-- Property that the color of the half-sum of two different numbers of the same parity
    depends only on the colors of the summands. -/
def HalfSumProperty (c : Coloring N) : Prop :=
  ∀ (a b x y : ℕ), a ≠ b → x ≠ y → a % 2 = b % 2 → x % 2 = y % 2 →
    c a = c x → c b = c y → c ((a + b) / 2) = c ((x + y) / 2)

/-- Main theorem about the properties of the coloring. -/
theorem coloring_properties (N : ℕ) (c : Coloring N)
    (h1 : InfinitelyMany c) (h2 : HalfSumProperty c) :
  (∀ (a b : ℕ), a % 2 = b % 2 → c a = c b → c ((a + b) / 2) = c a) ∧
  (∃ (coloring : Coloring N), InfinitelyMany coloring ∧ HalfSumProperty coloring ↔ N % 2 = 1) :=
by sorry

end coloring_properties_l648_64886


namespace carpet_shaded_area_carpet_specific_shaded_area_l648_64818

/-- Calculates the total shaded area of a rectangular carpet with specific dimensions and shaded areas. -/
theorem carpet_shaded_area (carpet_length carpet_width : ℝ) 
  (num_small_squares : ℕ) (ratio_long_to_R ratio_R_to_S : ℝ) : ℝ :=
  let R := carpet_length / ratio_long_to_R
  let S := R / ratio_R_to_S
  let area_R := R * R
  let area_S := S * S
  let total_area := area_R + (num_small_squares : ℝ) * area_S
  total_area

/-- Proves that the total shaded area of the carpet with given specifications is 141.75 square feet. -/
theorem carpet_specific_shaded_area : 
  carpet_shaded_area 18 12 12 2 4 = 141.75 := by
  sorry

end carpet_shaded_area_carpet_specific_shaded_area_l648_64818


namespace power_of_product_l648_64862

theorem power_of_product (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_of_product_l648_64862


namespace nancy_bottle_caps_l648_64848

theorem nancy_bottle_caps (initial : ℕ) : initial + 88 = 179 → initial = 91 := by
  sorry

end nancy_bottle_caps_l648_64848


namespace fireworks_display_count_l648_64837

/-- The number of fireworks needed to display a single number. -/
def fireworks_per_number : ℕ := 6

/-- The number of fireworks needed to display a single letter. -/
def fireworks_per_letter : ℕ := 5

/-- The number of digits in the year display. -/
def year_digits : ℕ := 4

/-- The number of letters in "HAPPY NEW YEAR". -/
def phrase_letters : ℕ := 12

/-- The number of additional boxes of fireworks. -/
def additional_boxes : ℕ := 50

/-- The number of fireworks in each additional box. -/
def fireworks_per_box : ℕ := 8

/-- The total number of fireworks lit during the display. -/
def total_fireworks : ℕ := 
  year_digits * fireworks_per_number + 
  phrase_letters * fireworks_per_letter + 
  additional_boxes * fireworks_per_box

theorem fireworks_display_count : total_fireworks = 484 := by
  sorry

end fireworks_display_count_l648_64837


namespace solution_set_a_2_range_of_a_l648_64847

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of values for a
theorem range_of_a :
  ∀ x : ℝ, f x a ≥ 4 → a ∈ Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end solution_set_a_2_range_of_a_l648_64847


namespace sum_of_seven_place_values_l648_64868

theorem sum_of_seven_place_values (n : ℚ) (h : n = 87953.0727) :
  (7000 : ℚ) + (7 / 100 : ℚ) + (7 / 10000 : ℚ) = 7000.0707 := by
  sorry

end sum_of_seven_place_values_l648_64868


namespace sum_reciprocal_inequality_l648_64843

theorem sum_reciprocal_inequality (a b c : ℝ) (h : a + b + c = 3) :
  1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2) ≤ 3/2 := by
  sorry

end sum_reciprocal_inequality_l648_64843


namespace roundness_of_900000_l648_64857

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 900,000 is 12 -/
theorem roundness_of_900000 : roundness 900000 = 12 := by sorry

end roundness_of_900000_l648_64857


namespace waiter_dishes_served_l648_64855

theorem waiter_dishes_served : 
  let num_tables : ℕ := 7
  let women_per_table : ℕ := 7
  let men_per_table : ℕ := 2
  let courses_per_woman : ℕ := 3
  let courses_per_man : ℕ := 4
  let shared_courses_women : ℕ := 1
  let shared_courses_men : ℕ := 2

  let dishes_per_table : ℕ := 
    women_per_table * courses_per_woman + 
    men_per_table * courses_per_man - 
    shared_courses_women - 
    shared_courses_men

  num_tables * dishes_per_table = 182
  := by sorry

end waiter_dishes_served_l648_64855


namespace two_year_increase_l648_64844

/-- Calculates the final value after two years of percentage increases -/
def final_value (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating the final value after two years of specific increases -/
theorem two_year_increase (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : initial = 65000)
  (h2 : rate1 = 0.12)
  (h3 : rate2 = 0.08) :
  final_value initial rate1 rate2 = 78624 := by
  sorry

#eval final_value 65000 0.12 0.08

end two_year_increase_l648_64844


namespace tangent_segment_region_area_l648_64885

/-- The area of the region formed by all line segments of length 6 that are tangent to a circle of radius 3 at their midpoints -/
theorem tangent_segment_region_area : Real := by
  -- Define the circle radius
  let circle_radius : Real := 3
  
  -- Define the line segment length
  let segment_length : Real := 6
  
  -- Define the region area
  let region_area : Real := 9 * Real.pi
  
  -- State that the line segments are tangent to the circle at their midpoints
  -- (This is implicitly used in the proof, but we don't need to explicitly define it in Lean)
  
  -- Prove that the area of the region is equal to 9π
  sorry

#check tangent_segment_region_area

end tangent_segment_region_area_l648_64885


namespace product_of_fractions_l648_64870

theorem product_of_fractions : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end product_of_fractions_l648_64870


namespace probability_not_adjacent_l648_64875

def total_chairs : ℕ := 10
def broken_chairs : Finset ℕ := {5, 8}
def available_chairs : ℕ := total_chairs - broken_chairs.card

def adjacent_pairs : ℕ := 6

theorem probability_not_adjacent :
  (1 - (adjacent_pairs : ℚ) / (available_chairs.choose 2)) = 11/14 := by sorry

end probability_not_adjacent_l648_64875


namespace diagonals_of_adjacent_faces_perpendicular_l648_64800

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A face diagonal is a line segment connecting opposite corners of a face -/
structure FaceDiagonal (c : Cube) where
  -- We don't need to define the specifics of a face diagonal for this problem

/-- Two faces are adjacent if they share an edge -/
def adjacent_faces (c : Cube) (f1 f2 : FaceDiagonal c) : Prop :=
  sorry  -- Definition of adjacent faces

/-- The angle between two lines -/
def angle_between (l1 l2 : FaceDiagonal c) : ℝ :=
  sorry  -- Definition of angle between two lines

/-- Theorem: The angle between diagonals of adjacent faces of a cube is 90 degrees -/
theorem diagonals_of_adjacent_faces_perpendicular (c : Cube) (d1 d2 : FaceDiagonal c) 
  (h : adjacent_faces c d1 d2) : angle_between d1 d2 = 90 := by
  sorry

end diagonals_of_adjacent_faces_perpendicular_l648_64800


namespace imaginary_part_of_complex_fraction_l648_64856

theorem imaginary_part_of_complex_fraction :
  Complex.im ((2 - Complex.I) / (1 + 2 * Complex.I)) = -1 := by sorry

end imaginary_part_of_complex_fraction_l648_64856


namespace rhombus_sum_difference_l648_64808

theorem rhombus_sum_difference (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a + 1) * (b + 1) + (b + 1) * (c + 1) + (c + 1) * (d + 1) + (d + 1) * (a + 1)) -
  (a * b + b * c + c * d + d * a) = 4046 := by
  sorry

end rhombus_sum_difference_l648_64808


namespace expression_simplification_l648_64858

theorem expression_simplification :
  (-8 : ℚ) * (18 / 14) * (49 / 27) + 4 / 3 = -52 / 3 := by
  sorry

end expression_simplification_l648_64858


namespace math_books_count_l648_64882

/-- Given a shelf of books with the following properties:
  * There are 100 books in total
  * 32 of them are history books
  * 25 of them are geography books
  * The rest are math books
  This theorem proves that there are 43 math books. -/
theorem math_books_count (total : ℕ) (history : ℕ) (geography : ℕ) (math : ℕ) 
  (h_total : total = 100)
  (h_history : history = 32)
  (h_geography : geography = 25)
  (h_sum : total = history + geography + math) :
  math = 43 := by
  sorry

end math_books_count_l648_64882


namespace smallest_number_minus_one_in_list_minus_one_is_smallest_l648_64821

def numbers : List ℚ := [3, 0, -1, -1/2]

theorem smallest_number (n : ℚ) (hn : n ∈ numbers) :
  -1 ≤ n := by sorry

theorem minus_one_in_list : -1 ∈ numbers := by sorry

theorem minus_one_is_smallest : ∀ n ∈ numbers, -1 ≤ n ∧ ∃ m ∈ numbers, -1 = m := by sorry

end smallest_number_minus_one_in_list_minus_one_is_smallest_l648_64821


namespace jills_shopping_trip_tax_percentage_l648_64849

/-- Represents the spending and tax information for a shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ

/-- Calculates the total tax as a percentage of the total amount spent (excluding taxes) -/
def totalTaxPercentage (trip : ShoppingTrip) : ℝ :=
  (trip.clothing_percent * trip.clothing_tax_rate +
   trip.food_percent * trip.food_tax_rate +
   trip.other_percent * trip.other_tax_rate) * 100

/-- Theorem stating that the total tax percentage for Jill's shopping trip is 4.40% -/
theorem jills_shopping_trip_tax_percentage :
  let trip : ShoppingTrip := {
    clothing_percent := 0.50,
    food_percent := 0.20,
    other_percent := 0.30,
    clothing_tax_rate := 0.04,
    food_tax_rate := 0,
    other_tax_rate := 0.08
  }
  totalTaxPercentage trip = 4.40 := by
  sorry

end jills_shopping_trip_tax_percentage_l648_64849


namespace second_day_speed_l648_64817

/-- Proves that given the climbing conditions, the speed on the second day is 4 km/h -/
theorem second_day_speed (total_time : ℝ) (speed_difference : ℝ) (time_difference : ℝ) (total_distance : ℝ)
  (h1 : total_time = 14)
  (h2 : speed_difference = 0.5)
  (h3 : time_difference = 2)
  (h4 : total_distance = 52) :
  let first_day_time := (total_time + time_difference) / 2
  let second_day_time := total_time - first_day_time
  let first_day_speed := (total_distance - speed_difference * second_day_time) / total_time
  let second_day_speed := first_day_speed + speed_difference
  second_day_speed = 4 := by sorry

end second_day_speed_l648_64817


namespace randy_farm_trees_l648_64819

/-- Calculates the total number of trees on Randy's farm -/
def total_trees (mango_trees : ℕ) (coconut_trees : ℕ) : ℕ :=
  mango_trees + coconut_trees

/-- Theorem: Given Randy's farm conditions, the total number of trees is 85 -/
theorem randy_farm_trees :
  let mango_trees : ℕ := 60
  let coconut_trees : ℕ := mango_trees / 2 - 5
  total_trees mango_trees coconut_trees = 85 := by
  sorry

end randy_farm_trees_l648_64819


namespace third_circle_radius_l648_64831

/-- Given two internally tangent circles with radii R and r (R > r),
    the radius x of a third circle tangent to both circles and their common diameter
    is given by x = 4Rr / (R + r). -/
theorem third_circle_radius (R r : ℝ) (h : R > r) (h_pos_R : R > 0) (h_pos_r : r > 0) :
  ∃ x : ℝ, x > 0 ∧ x = (4 * R * r) / (R + r) ∧
    (∀ y : ℝ, y > 0 → y ≠ x →
      ¬(∃ p q : ℝ × ℝ,
        (p.1 - q.1)^2 + (p.2 - q.2)^2 = (R - r)^2 ∧
        (p.1 - 0)^2 + (p.2 - 0)^2 = R^2 ∧
        (q.1 - 0)^2 + (q.2 - 0)^2 = r^2 ∧
        ((p.1 + q.1)/2 - 0)^2 + ((p.2 + q.2)/2 - y)^2 = y^2)) :=
by sorry

end third_circle_radius_l648_64831


namespace gcd_12345_6789_l648_64889

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l648_64889


namespace solution_set_implies_m_value_l648_64874

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- State the theorem
theorem solution_set_implies_m_value :
  ∃ m : ℝ, (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end solution_set_implies_m_value_l648_64874


namespace m_arit_fib_seq_periodic_l648_64815

/-- An m-arithmetic Fibonacci sequence -/
def MAritFibSeq (m : ℕ) := ℕ → Fin m

/-- The period of an m-arithmetic Fibonacci sequence -/
def Period (m : ℕ) (v : MAritFibSeq m) (r : ℕ) : Prop :=
  ∀ n, v n = v (n + r)

theorem m_arit_fib_seq_periodic (m : ℕ) (v : MAritFibSeq m) :
  ∃ r : ℕ, r ≤ m^2 ∧ Period m v r := by
  sorry

end m_arit_fib_seq_periodic_l648_64815


namespace total_volume_of_four_cubes_l648_64859

theorem total_volume_of_four_cubes (edge_length : ℝ) (num_cubes : ℕ) : 
  edge_length = 5 → num_cubes = 4 → num_cubes * (edge_length ^ 3) = 500 := by
  sorry

end total_volume_of_four_cubes_l648_64859


namespace evaluate_polynomial_l648_64867

theorem evaluate_polynomial (x : ℤ) (h : x = -2) : x^3 + x^2 + x + 1 = -5 := by
  sorry

end evaluate_polynomial_l648_64867


namespace spinner_direction_l648_64845

-- Define the directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation := 3 + 3/4
  let counterclockwise_rotation := 2 + 2/4
  rotate initial_direction clockwise_rotation counterclockwise_rotation = Direction.East :=
sorry

end spinner_direction_l648_64845


namespace melinda_coffees_l648_64824

/-- The cost of one doughnut in dollars -/
def doughnut_cost : ℚ := 45/100

/-- The total cost of Harold's purchase on Monday in dollars -/
def harold_total : ℚ := 491/100

/-- The number of doughnuts Harold bought on Monday -/
def harold_doughnuts : ℕ := 3

/-- The number of coffees Harold bought on Monday -/
def harold_coffees : ℕ := 4

/-- The total cost of Melinda's purchase on Tuesday in dollars -/
def melinda_total : ℚ := 759/100

/-- The number of doughnuts Melinda bought on Tuesday -/
def melinda_doughnuts : ℕ := 5

/-- Theorem stating that Melinda bought 6 large coffees on Tuesday -/
theorem melinda_coffees : ℕ := by
  sorry


end melinda_coffees_l648_64824


namespace unique_intersection_point_l648_64872

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies the equation 3x + 2y - 9 = 0 -/
def satisfiesLine1 (p : Point) : Prop :=
  3 * p.x + 2 * p.y - 9 = 0

/-- Checks if a point satisfies the equation 5x - 2y - 10 = 0 -/
def satisfiesLine2 (p : Point) : Prop :=
  5 * p.x - 2 * p.y - 10 = 0

/-- Checks if a point satisfies the equation x = 3 -/
def satisfiesLine3 (p : Point) : Prop :=
  p.x = 3

/-- Checks if a point satisfies the equation y = 1 -/
def satisfiesLine4 (p : Point) : Prop :=
  p.y = 1

/-- Checks if a point satisfies the equation x + y = 4 -/
def satisfiesLine5 (p : Point) : Prop :=
  p.x + p.y = 4

/-- Checks if a point satisfies all five line equations -/
def satisfiesAllLines (p : Point) : Prop :=
  satisfiesLine1 p ∧ satisfiesLine2 p ∧ satisfiesLine3 p ∧ satisfiesLine4 p ∧ satisfiesLine5 p

/-- Theorem stating that there is exactly one point satisfying all five line equations -/
theorem unique_intersection_point : ∃! p : Point, satisfiesAllLines p := by
  sorry

end unique_intersection_point_l648_64872


namespace complement_of_union_is_five_l648_64832

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 4}

-- Theorem statement
theorem complement_of_union_is_five :
  (U \ (A ∪ B)) = {5} := by
  sorry

end complement_of_union_is_five_l648_64832


namespace kaylin_is_33_l648_64869

def freyja_age : ℕ := 10

def eli_age (freyja_age : ℕ) : ℕ := freyja_age + 9

def sarah_age (eli_age : ℕ) : ℕ := 2 * eli_age

def kaylin_age (sarah_age : ℕ) : ℕ := sarah_age - 5

theorem kaylin_is_33 : 
  kaylin_age (sarah_age (eli_age freyja_age)) = 33 := by
sorry

end kaylin_is_33_l648_64869


namespace smallest_x_with_remainders_l648_64814

theorem smallest_x_with_remainders : ∃ (x : ℕ), 
  (x % 5 = 4) ∧ 
  (x % 6 = 5) ∧ 
  (x % 7 = 6) ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(y % 5 = 4 ∧ y % 6 = 5 ∧ y % 7 = 6)) ∧
  x = 209 :=
by sorry

end smallest_x_with_remainders_l648_64814


namespace roses_in_vase_l648_64896

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 7

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 12

/-- The number of orchids in the vase now -/
def current_orchids : ℕ := 20

/-- The difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 9

/-- The number of roses in the vase now -/
def current_roses : ℕ := 11

theorem roses_in_vase :
  current_orchids = current_roses + orchid_rose_difference :=
by sorry

end roses_in_vase_l648_64896


namespace sin_eq_sin_sin_solution_count_l648_64854

theorem sin_eq_sin_sin_solution_count :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin 0.99 ∧ Real.sin x = Real.sin (Real.sin x) := by
  sorry

end sin_eq_sin_sin_solution_count_l648_64854


namespace song_performance_theorem_l648_64879

/-- Represents the number of songs performed by each kid -/
structure SongCounts where
  sarah : ℕ
  emily : ℕ
  daniel : ℕ
  oli : ℕ
  chris : ℕ

/-- The total number of songs performed -/
def totalSongs (counts : SongCounts) : ℕ :=
  (counts.sarah + counts.emily + counts.daniel + counts.oli + counts.chris) / 4

theorem song_performance_theorem (counts : SongCounts) :
  counts.chris = 9 →
  counts.sarah = 3 →
  counts.emily > counts.sarah →
  counts.daniel > counts.sarah →
  counts.oli > counts.sarah →
  counts.emily < counts.chris →
  counts.daniel < counts.chris →
  counts.oli < counts.chris →
  totalSongs counts = 6 :=
by sorry

end song_performance_theorem_l648_64879


namespace yellow_jacket_incident_l648_64861

theorem yellow_jacket_incident (total_students : ℕ) 
  (initial_cafeteria_fraction : ℚ) (final_cafeteria_count : ℕ) 
  (cafeteria_to_outside : ℕ) : 
  total_students = 90 →
  initial_cafeteria_fraction = 2/3 →
  final_cafeteria_count = 67 →
  cafeteria_to_outside = 3 →
  (final_cafeteria_count - (initial_cafeteria_fraction * total_students).floor + cafeteria_to_outside) / 
  (total_students - (initial_cafeteria_fraction * total_students).floor) = 1/3 := by
sorry

end yellow_jacket_incident_l648_64861


namespace arithmetic_sequence_ratio_l648_64884

/-- Given two arithmetic sequences {a_n} and {b_n}, S_n and T_n are the sums of their first n terms respectively -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

/-- The ratio of S_n to T_n is (7n + 2) / (n + 3) for all n -/
def ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n + 2) / (n + 3)

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h1 : arithmetic_sequences a b S T)
  (h2 : ratio_condition S T) :
  a 5 / b 5 = 65 / 12 :=
sorry

end arithmetic_sequence_ratio_l648_64884


namespace power_of_product_l648_64828

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end power_of_product_l648_64828


namespace crate_width_is_sixteen_l648_64864

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank can fit upright in a crate -/
def fitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  (tank.radius * 2 ≤ crate.length ∧ tank.radius * 2 ≤ crate.width) ∨
  (tank.radius * 2 ≤ crate.length ∧ tank.radius * 2 ≤ crate.height) ∨
  (tank.radius * 2 ≤ crate.width ∧ tank.radius * 2 ≤ crate.height)

/-- Theorem: The width of the crate must be 16 feet -/
theorem crate_width_is_sixteen
  (crate : CrateDimensions)
  (tank : GasTank)
  (h1 : crate.length = 12)
  (h2 : crate.height = 18)
  (h3 : tank.radius = 8)
  (h4 : fitsInCrate tank crate)
  (h5 : ∀ t : GasTank, fitsInCrate t crate → t.radius ≤ tank.radius) :
  crate.width = 16 := by
  sorry

end crate_width_is_sixteen_l648_64864


namespace invalid_triangle_after_transformation_l648_64813

theorem invalid_triangle_after_transformation (DE DF EF : ℝ) 
  (h_original_valid : DE + DF > EF ∧ DE + EF > DF ∧ DF + EF > DE)
  (h_DE : DE = 8)
  (h_DF : DF = 9)
  (h_EF : EF = 5)
  (DE' DF' EF' : ℝ)
  (h_DE' : DE' = 3 * DE)
  (h_DF' : DF' = 2 * DF)
  (h_EF' : EF' = EF) :
  ¬(DE' + DF' > EF' ∧ DE' + EF' > DF' ∧ DF' + EF' > DE') :=
by sorry

end invalid_triangle_after_transformation_l648_64813


namespace solve_equation_l648_64841

theorem solve_equation : ∃ x : ℚ, 5 * (x - 4) = 3 * (6 - 3 * x) + 9 ∧ x = 47 / 14 := by
  sorry

end solve_equation_l648_64841


namespace consecutive_negative_integers_product_sum_l648_64880

theorem consecutive_negative_integers_product_sum (n : ℤ) :
  n < 0 ∧ n * (n + 1) = 2184 → n + (n + 1) = -95 := by
  sorry

end consecutive_negative_integers_product_sum_l648_64880


namespace infinite_sum_equality_l648_64842

theorem infinite_sum_equality (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  let f : ℕ → ℝ := fun n => 1 / ((n * c - (n - 1) * d) * ((n + 1) * c - n * d))
  let series := ∑' n, f n
  series = 1 / ((c - d) * d) := by sorry

end infinite_sum_equality_l648_64842


namespace B_determines_xy_l648_64809

/-- Function B that determines x and y --/
def B (x y : ℕ) : ℕ := (x + y) * (x + y + 1) - y

/-- Theorem stating that B(x, y) uniquely determines x and y --/
theorem B_determines_xy (x y : ℕ) : 
  ∀ a b : ℕ, B x y = B a b → x = a ∧ y = b := by sorry

end B_determines_xy_l648_64809


namespace whatsis_whosis_equals_so_plus_so_l648_64881

/-- A structure representing the variables in the problem -/
structure Variables where
  whatsis : ℝ
  whosis : ℝ
  is : ℝ
  so : ℝ
  pos_whatsis : 0 < whatsis
  pos_whosis : 0 < whosis
  pos_is : 0 < is
  pos_so : 0 < so

/-- The main theorem representing the problem -/
theorem whatsis_whosis_equals_so_plus_so (v : Variables) 
  (h1 : v.whatsis = v.so)
  (h2 : v.whosis = v.is)
  (h3 : v.so + v.so = v.is * v.so)
  (h4 : v.whosis = v.so)
  (h5 : v.so + v.so = v.so * v.so)
  (h6 : v.is = 2) :
  v.whosis * v.whatsis = v.so + v.so := by
  sorry


end whatsis_whosis_equals_so_plus_so_l648_64881


namespace max_goals_scored_l648_64892

/-- Represents the number of goals scored by Marlon in a soccer game --/
def goals_scored (penalty_shots free_kicks : ℕ) : ℝ :=
  0.4 * penalty_shots + 0.5 * free_kicks

/-- Proves that the maximum number of goals Marlon could have scored is 20 --/
theorem max_goals_scored : 
  ∀ penalty_shots free_kicks : ℕ, 
  penalty_shots + free_kicks = 40 →
  goals_scored penalty_shots free_kicks ≤ 20 :=
by
  sorry

#check max_goals_scored

end max_goals_scored_l648_64892


namespace right_triangle_angle_A_l648_64804

theorem right_triangle_angle_A (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.cos B = Real.sqrt 3 / 2) : A = Real.pi / 3 := by
  sorry

end right_triangle_angle_A_l648_64804


namespace min_comparisons_correct_l648_64865

/-- Represents a set of coins with different weights -/
structure CoinSet (n : ℕ) where
  coins : Fin n → ℝ
  different_weights : ∀ i j, i ≠ j → coins i ≠ coins j

/-- Represents a set of balances, including one faulty balance -/
structure BalanceSet (n : ℕ) where
  balances : Fin n → Bool
  one_faulty : ∃ i, balances i = false

/-- The minimum number of comparisons needed to find the heaviest coin -/
def min_comparisons (n : ℕ) : ℕ := 2 * n - 1

/-- The main theorem: proving the minimum number of comparisons -/
theorem min_comparisons_correct (n : ℕ) (h : n > 2) 
  (coins : CoinSet n) (balances : BalanceSet n) :
  min_comparisons n = 2 * n - 1 :=
sorry

end min_comparisons_correct_l648_64865


namespace map_scale_proportion_l648_64825

/-- Represents the scale of a map -/
structure MapScale where
  cm : ℝ  -- centimeters on the map
  km : ℝ  -- kilometers in reality

/-- 
Given a map scale where 15 cm represents 90 km, 
proves that 20 cm represents 120 km on the same map
-/
theorem map_scale_proportion (scale : MapScale) 
  (h : scale.cm = 15 ∧ scale.km = 90) : 
  ∃ (new_scale : MapScale), 
    new_scale.cm = 20 ∧ 
    new_scale.km = 120 ∧
    new_scale.km / new_scale.cm = scale.km / scale.cm := by
  sorry


end map_scale_proportion_l648_64825


namespace arithmetic_sequence_common_difference_l648_64894

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 1)
  (h_sum : a 3 + a 4 = 8) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end arithmetic_sequence_common_difference_l648_64894


namespace irwin_two_point_baskets_l648_64873

/-- Represents the number of baskets scored for each point value -/
structure BasketCount where
  two_point : ℕ
  five_point : ℕ
  eleven_point : ℕ
  thirteen_point : ℕ

/-- Calculates the product of point values for a given BasketCount -/
def pointValueProduct (b : BasketCount) : ℕ :=
  2^b.two_point * 5^b.five_point * 11^b.eleven_point * 13^b.thirteen_point

theorem irwin_two_point_baskets :
  ∀ b : BasketCount,
    pointValueProduct b = 2420 →
    b.eleven_point = 2 →
    b.two_point = 2 := by
  sorry

end irwin_two_point_baskets_l648_64873


namespace least_addition_for_divisibility_l648_64811

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 29989 ∧ m = 73) :
  ∃ x : ℕ, x = 21 ∧ 
    (∀ y : ℕ, (n + y) % m = 0 → y ≥ x) ∧
    (n + x) % m = 0 :=
  sorry

end least_addition_for_divisibility_l648_64811


namespace subtraction_puzzle_sum_l648_64840

theorem subtraction_puzzle_sum :
  ∀ (P Q R S T : ℕ),
    P < 10 → Q < 10 → R < 10 → S < 10 → T < 10 →
    70000 + 1000 * Q + 200 + 10 * S + T - (10000 * P + 3000 + 100 * R + 90 + 6) = 22222 →
    P + Q + R + S + T = 29 := by
  sorry

end subtraction_puzzle_sum_l648_64840


namespace quadratic_inequality_l648_64876

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by sorry

end quadratic_inequality_l648_64876


namespace sean_shopping_cost_l648_64866

-- Define the prices and quantities
def soda_price : ℝ := 1
def soda_quantity : ℕ := 4
def soup_quantity : ℕ := 3
def sandwich_quantity : ℕ := 2
def salad_quantity : ℕ := 1

-- Define price relationships
def soup_price : ℝ := 2 * soda_price
def sandwich_price : ℝ := 4 * soup_price
def salad_price : ℝ := 2 * sandwich_price

-- Define discount and tax rates
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05

-- Calculate total cost before discount and tax
def total_cost : ℝ :=
  soda_price * soda_quantity +
  soup_price * soup_quantity +
  sandwich_price * sandwich_quantity +
  salad_price * salad_quantity

-- Calculate final cost after discount and tax
def final_cost : ℝ :=
  total_cost * (1 - discount_rate) * (1 + tax_rate)

-- Theorem to prove
theorem sean_shopping_cost :
  final_cost = 39.69 := by sorry

end sean_shopping_cost_l648_64866


namespace notebooks_left_l648_64827

theorem notebooks_left (notebooks_per_bundle : ℕ) (num_bundles : ℕ) (num_groups : ℕ) (students_per_group : ℕ) : 
  notebooks_per_bundle = 25 →
  num_bundles = 5 →
  num_groups = 8 →
  students_per_group = 13 →
  num_bundles * notebooks_per_bundle - num_groups * students_per_group = 21 := by
  sorry

end notebooks_left_l648_64827


namespace f_max_value_l648_64893

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + x + 1)

theorem f_max_value (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1/2 → ∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ Real.exp (-1/a)) ∧
  (a > 1/2 → ∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ Real.exp (-2) * (4*a - 1)) :=
sorry

end f_max_value_l648_64893


namespace max_value_sum_fractions_l648_64897

theorem max_value_sum_fractions (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 ∧
    a' * b' / (a' + b') + a' * c' / (a' + c') + b' * c' / (b' + c') = 1 :=
by sorry

end max_value_sum_fractions_l648_64897


namespace justin_flower_gathering_l648_64860

def minutes_per_flower : ℕ := 10
def gathering_hours : ℕ := 2
def lost_flowers : ℕ := 3
def classmates : ℕ := 30

def additional_minutes_needed : ℕ :=
  let gathered_flowers := gathering_hours * 60 / minutes_per_flower
  let remaining_flowers := classmates - (gathered_flowers - lost_flowers)
  remaining_flowers * minutes_per_flower

theorem justin_flower_gathering :
  additional_minutes_needed = 210 := by
  sorry

end justin_flower_gathering_l648_64860


namespace average_listening_time_approx_33_l648_64898

/-- Represents the distribution of audience members and their listening times --/
structure AudienceDistribution where
  total_audience : ℕ
  talk_duration : ℕ
  full_listeners_percent : ℚ
  sleepers_percent : ℚ
  half_listeners_percent : ℚ
  quarter_listeners_percent : ℚ

/-- Calculates the average listening time for the audience --/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  let full_listeners := (dist.full_listeners_percent * dist.total_audience) * dist.talk_duration
  let sleepers := 0
  let half_listeners := (dist.half_listeners_percent * dist.total_audience) * (dist.talk_duration / 2)
  let quarter_listeners := (dist.quarter_listeners_percent * dist.total_audience) * (dist.talk_duration / 4)
  (full_listeners + sleepers + half_listeners + quarter_listeners) / dist.total_audience

/-- The given audience distribution --/
def lecture_distribution : AudienceDistribution :=
  { total_audience := 200
  , talk_duration := 90
  , full_listeners_percent := 15 / 100
  , sleepers_percent := 15 / 100
  , half_listeners_percent := (1 / 4) * (70 / 100)
  , quarter_listeners_percent := (3 / 4) * (70 / 100)
  }

/-- Theorem stating that the average listening time is approximately 33 minutes --/
theorem average_listening_time_approx_33 :
  ∃ ε > 0, |average_listening_time lecture_distribution - 33| < ε :=
sorry

end average_listening_time_approx_33_l648_64898


namespace tens_digit_of_6_to_19_l648_64899

-- Define a function to calculate the tens digit
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

-- State the theorem
theorem tens_digit_of_6_to_19 :
  tens_digit (6^19) = 1 :=
sorry

end tens_digit_of_6_to_19_l648_64899


namespace pens_sold_in_garage_sale_l648_64806

/-- Given that Paul initially had 42 pens and after a garage sale had 19 pens left,
    prove that he sold 23 pens in the garage sale. -/
theorem pens_sold_in_garage_sale :
  let initial_pens : ℕ := 42
  let remaining_pens : ℕ := 19
  initial_pens - remaining_pens = 23 := by sorry

end pens_sold_in_garage_sale_l648_64806


namespace min_value_expression_l648_64838

theorem min_value_expression (x : ℝ) : 
  (15 - x) * (14 - x) * (15 + x) * (14 + x) ≥ -142.25 :=
by sorry

end min_value_expression_l648_64838


namespace iron_weight_is_11_16_l648_64887

/-- The weight of the piece of aluminum in pounds -/
def aluminum_weight : ℝ := 0.83

/-- The difference in weight between the piece of iron and the piece of aluminum in pounds -/
def weight_difference : ℝ := 10.33

/-- The weight of the piece of iron in pounds -/
def iron_weight : ℝ := aluminum_weight + weight_difference

/-- Theorem stating that the weight of the piece of iron is 11.16 pounds -/
theorem iron_weight_is_11_16 : iron_weight = 11.16 := by sorry

end iron_weight_is_11_16_l648_64887
