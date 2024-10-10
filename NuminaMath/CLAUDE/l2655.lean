import Mathlib

namespace functional_equation_implies_additive_l2655_265533

/-- A function satisfying the given functional equation is additive. -/
theorem functional_equation_implies_additive (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :
  ∀ x y : ℝ, f (x + y) = f x + f y := by
  sorry

end functional_equation_implies_additive_l2655_265533


namespace journey_speed_proof_l2655_265547

/-- Proves that given a journey in three equal parts with speeds 5 km/hr, v km/hr, and 15 km/hr,
    where the total time is 11 minutes and the total distance is 1.5 km, the value of v is 10 km/hr. -/
theorem journey_speed_proof (v : ℝ) : 
  let total_distance : ℝ := 1.5 -- km
  let part_distance : ℝ := total_distance / 3
  let total_time : ℝ := 11 / 60 -- hours
  let time1 : ℝ := part_distance / 5
  let time2 : ℝ := part_distance / v
  let time3 : ℝ := part_distance / 15
  time1 + time2 + time3 = total_time → v = 10 := by sorry

end journey_speed_proof_l2655_265547


namespace sqrt_sum_fractions_l2655_265503

theorem sqrt_sum_fractions : Real.sqrt (1/9 + 1/16) = 5/12 := by
  sorry

end sqrt_sum_fractions_l2655_265503


namespace prime_cube_difference_to_sum_of_squares_l2655_265568

theorem prime_cube_difference_to_sum_of_squares (p a b : ℕ) : 
  Prime p → (∃ a b : ℕ, p = a^3 - b^3) → (∃ c d : ℕ, p = c^2 + 3*d^2) := by
  sorry

end prime_cube_difference_to_sum_of_squares_l2655_265568


namespace total_pay_for_two_employees_l2655_265578

/-- Proves that the total amount paid to two employees X and Y is 770 units of currency,
    given that X is paid 120% of Y's pay and Y is paid 350 units per week. -/
theorem total_pay_for_two_employees (y_pay : ℝ) (x_pay : ℝ) : 
  y_pay = 350 → x_pay = 1.2 * y_pay → x_pay + y_pay = 770 := by
  sorry

end total_pay_for_two_employees_l2655_265578


namespace james_age_l2655_265555

/-- Represents the ages of Dan, James, and Lisa --/
structure Ages where
  dan : ℕ
  james : ℕ
  lisa : ℕ

/-- The conditions of the problem --/
def age_conditions (ages : Ages) : Prop :=
  ∃ (k : ℕ),
    ages.dan = 6 * k ∧
    ages.james = 5 * k ∧
    ages.lisa = 4 * k ∧
    ages.dan + 4 = 28 ∧
    ages.james + ages.lisa = 3 * (ages.james - ages.lisa)

/-- The theorem to prove --/
theorem james_age (ages : Ages) :
  age_conditions ages → ages.james = 20 := by
  sorry

end james_age_l2655_265555


namespace smallest_multiple_thirty_six_satisfies_smallest_positive_integer_l2655_265560

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 648 ∣ (450 * x) → x ≥ 36 := by
  sorry

theorem thirty_six_satisfies : 648 ∣ (450 * 36) := by
  sorry

theorem smallest_positive_integer : 
  ∃ (x : ℕ), x > 0 ∧ 648 ∣ (450 * x) ∧ ∀ (y : ℕ), y > 0 ∧ 648 ∣ (450 * y) → x ≤ y := by
  sorry

end smallest_multiple_thirty_six_satisfies_smallest_positive_integer_l2655_265560


namespace hyperbola_focal_distance_l2655_265557

theorem hyperbola_focal_distance (P F₁ F₂ : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2 / 64 - y^2 / 36 = 1) →  -- P is on the hyperbola
  (∃ c : ℝ, c > 0 ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0)) →  -- F₁ and F₂ are foci
  ‖P - F₁‖ = 15 →  -- |PF₁| = 15
  ‖P - F₂‖ = 31 :=  -- |PF₂| = 31
by sorry

end hyperbola_focal_distance_l2655_265557


namespace triangle_is_right_angle_l2655_265523

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (angle_sum : A + B + C = π)

-- State the theorem
theorem triangle_is_right_angle (t : Triangle) 
  (h : (Real.cos (t.A / 2))^2 = (t.b + t.c) / (2 * t.c)) : 
  t.c^2 = t.a^2 + t.b^2 := by
  sorry

end triangle_is_right_angle_l2655_265523


namespace stratified_sample_size_l2655_265584

/-- Proves that for a population of 600 with 250 young employees,
    a stratified sample with 5 young employees has a total size of 12 -/
theorem stratified_sample_size
  (total_population : ℕ)
  (young_population : ℕ)
  (young_sample : ℕ)
  (h1 : total_population = 600)
  (h2 : young_population = 250)
  (h3 : young_sample = 5)
  (h4 : young_population ≤ total_population)
  (h5 : young_sample > 0) :
  ∃ (sample_size : ℕ),
    sample_size * young_population = young_sample * total_population ∧
    sample_size = 12 := by
  sorry


end stratified_sample_size_l2655_265584


namespace marathon_time_proof_l2655_265599

theorem marathon_time_proof (dean_time jake_time micah_time : ℝ) : 
  dean_time = 9 →
  micah_time = (2/3) * dean_time →
  jake_time = micah_time + (1/3) * micah_time →
  micah_time + dean_time + jake_time = 23 := by
sorry

end marathon_time_proof_l2655_265599


namespace orange_juice_serving_size_l2655_265564

/-- Calculates the size of each serving of orange juice in ounces. -/
def serving_size (concentrate_cans : ℕ) (water_ratio : ℕ) (concentrate_oz : ℕ) (total_servings : ℕ) : ℚ :=
  let total_cans := concentrate_cans * (water_ratio + 1)
  let total_oz := total_cans * concentrate_oz
  (total_oz : ℚ) / total_servings

/-- Proves that the size of each serving is 6 ounces under the given conditions. -/
theorem orange_juice_serving_size :
  serving_size 34 3 12 272 = 6 := by
  sorry

end orange_juice_serving_size_l2655_265564


namespace chinese_dream_essay_contest_l2655_265553

theorem chinese_dream_essay_contest (total : ℕ) (seventh : ℕ) (eighth : ℕ) :
  total = 118 →
  seventh = eighth / 2 - 2 →
  total = seventh + eighth →
  seventh = 38 := by
sorry

end chinese_dream_essay_contest_l2655_265553


namespace probability_same_color_problem_l2655_265580

/-- Probability of drawing two balls of the same color with replacement -/
def probability_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green^2 + red^2 + blue^2) / total^2

/-- Theorem: The probability of drawing two balls of the same color is 29/81 -/
theorem probability_same_color_problem :
  probability_same_color 8 6 4 = 29 / 81 := by
  sorry

end probability_same_color_problem_l2655_265580


namespace reflection_in_first_quadrant_l2655_265585

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the reflection across y-axis
def reflect_y (p : Point) : Point :=
  (-p.1, p.2)

-- Define the first quadrant
def in_first_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem reflection_in_first_quadrant :
  let P : Point := (-3, 1)
  in_first_quadrant (reflect_y P) := by sorry

end reflection_in_first_quadrant_l2655_265585


namespace pond_animals_l2655_265509

/-- Given a pond with snails and frogs, calculate the total number of animals -/
theorem pond_animals (num_snails num_frogs : ℕ) : num_snails = 5 → num_frogs = 2 → num_snails + num_frogs = 7 := by
  sorry

end pond_animals_l2655_265509


namespace total_winter_clothing_l2655_265597

def number_of_boxes : ℕ := 8
def scarves_per_box : ℕ := 4
def mittens_per_box : ℕ := 6

theorem total_winter_clothing : 
  number_of_boxes * (scarves_per_box + mittens_per_box) = 80 := by
  sorry

end total_winter_clothing_l2655_265597


namespace sum_of_solutions_l2655_265514

theorem sum_of_solutions (x : ℝ) : 
  (18 * x^2 - 45 * x - 70 = 0) → 
  (∃ y : ℝ, 18 * y^2 - 45 * y - 70 = 0 ∧ x + y = 5/2) :=
by sorry

end sum_of_solutions_l2655_265514


namespace second_die_has_seven_sides_l2655_265565

/-- The number of sides on the first die -/
def first_die_sides : ℕ := 6

/-- The probability of rolling a sum of 13 with both dice -/
def prob_sum_13 : ℚ := 23809523809523808 / 1000000000000000000

/-- The number of sides on the second die -/
def second_die_sides : ℕ := sorry

theorem second_die_has_seven_sides :
  (1 : ℚ) / (first_die_sides * second_die_sides) = prob_sum_13 ∧ 
  second_die_sides ≥ 7 →
  second_die_sides = 7 := by sorry

end second_die_has_seven_sides_l2655_265565


namespace smallest_with_20_divisors_l2655_265575

def num_divisors (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem smallest_with_20_divisors : 
  ∃ (n : ℕ+), num_divisors n = 20 ∧ ∀ (m : ℕ+), m < n → num_divisors m ≠ 20 :=
by
  -- The proof goes here
  sorry

end smallest_with_20_divisors_l2655_265575


namespace nested_expression_value_l2655_265561

theorem nested_expression_value : (3*(3*(2*(2*(2*(3+2)+1)+1)+2)+1)+1) = 436 := by
  sorry

end nested_expression_value_l2655_265561


namespace matthew_crackers_l2655_265513

theorem matthew_crackers (initial_crackers remaining_crackers crackers_per_friend : ℕ) 
  (h1 : initial_crackers = 23)
  (h2 : remaining_crackers = 11)
  (h3 : crackers_per_friend = 6) :
  (initial_crackers - remaining_crackers) / crackers_per_friend = 2 :=
by sorry

end matthew_crackers_l2655_265513


namespace orange_bags_l2655_265593

def total_weight : ℝ := 45.0
def bag_capacity : ℝ := 23.0

theorem orange_bags : ⌊total_weight / bag_capacity⌋ = 1 := by sorry

end orange_bags_l2655_265593


namespace baker_cakes_theorem_l2655_265534

/-- Represents the number of cakes Baker made -/
def cakes_made : ℕ := sorry

/-- Represents the number of pastries Baker made -/
def pastries_made : ℕ := 153

/-- Represents the number of pastries Baker sold -/
def pastries_sold : ℕ := 8

/-- Represents the number of cakes Baker sold -/
def cakes_sold : ℕ := 97

/-- Represents the difference between cakes sold and pastries sold -/
def difference_sold : ℕ := 89

theorem baker_cakes_theorem : 
  pastries_made = 153 ∧ 
  pastries_sold = 8 ∧ 
  cakes_sold = 97 ∧ 
  difference_sold = 89 ∧ 
  cakes_sold - pastries_sold = difference_sold → 
  cakes_made = 97 :=
by sorry

end baker_cakes_theorem_l2655_265534


namespace hyperbola_equation_l2655_265512

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 → b > 0 →
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ 
    (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ (x - F.1)^2/a^2 - (y - F.2)^2/b^2 = 1) ∧
    (P.2^2 = 8*P.1) ∧
    ((P.1 - F.1)^2 + (P.2 - F.2)^2 = 25)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2 - y^2/3 = 1) :=
by sorry

end hyperbola_equation_l2655_265512


namespace prop1_prop2_prop3_prop4_l2655_265540

-- Define the function f
variable (f : ℝ → ℝ)

-- Proposition 1
theorem prop1 (h : ∀ x, f (1 + 2*x) = f (1 - 2*x)) :
  ∀ x, f (2 - x) = f x :=
sorry

-- Proposition 2
theorem prop2 :
  ∀ x, f (x - 2) = f (2 - x) :=
sorry

-- Proposition 3
theorem prop3 (h1 : ∀ x, f x = f (-x)) (h2 : ∀ x, f (2 + x) = -f x) :
  ∀ x, f (4 - x) = f x :=
sorry

-- Proposition 4
theorem prop4 (h1 : ∀ x, f x = -f (-x)) (h2 : ∀ x, f x = f (-x - 2)) :
  ∀ x, f (2 - x) = f x :=
sorry

end prop1_prop2_prop3_prop4_l2655_265540


namespace triangle_centroid_distance_sum_l2655_265516

/-- Given a triangle ABC with centroid G, if GA² + GB² + GC² = 72, then AB² + AC² + BC² = 216 -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  ((G.1 - A.1)^2 + (G.2 - A.2)^2 + 
   (G.1 - B.1)^2 + (G.2 - B.2)^2 + 
   (G.1 - C.1)^2 + (G.2 - C.2)^2 = 72) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 + 
   (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 216) := by
sorry

end triangle_centroid_distance_sum_l2655_265516


namespace binomial_coefficient_seven_three_l2655_265556

theorem binomial_coefficient_seven_three : Nat.choose 7 3 = 35 := by
  sorry

end binomial_coefficient_seven_three_l2655_265556


namespace yulgi_pocket_money_l2655_265517

/-- Proves that Yulgi's pocket money is 3600 won given the problem conditions -/
theorem yulgi_pocket_money :
  ∀ (y g : ℕ),
  y + g = 6000 →
  (y + g) - (y - g) = 4800 →
  y > g →
  y = 3600 := by
sorry

end yulgi_pocket_money_l2655_265517


namespace trebled_result_is_72_l2655_265590

theorem trebled_result_is_72 (x : ℕ) (h : x = 9) : 3 * (2 * x + 6) = 72 := by
  sorry

end trebled_result_is_72_l2655_265590


namespace blue_marble_ratio_l2655_265510

/-- Proves that the ratio of blue marbles to total marbles is 1:2 -/
theorem blue_marble_ratio (total : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 164 → 
  red = total / 4 →
  green = 27 →
  yellow = 14 →
  blue = total - (red + green + yellow) →
  (blue : ℚ) / total = 1 / 2 := by
  sorry

end blue_marble_ratio_l2655_265510


namespace min_distance_between_curves_l2655_265569

theorem min_distance_between_curves (a b c d : ℝ) 
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (∀ x y z w : ℝ, (x + 3 * Real.log x) / y = 1 → (w - 3) / (2 * z) = 1 → 
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) ∧
  (a - c)^2 + (b - d)^2 = 9/5 * Real.log (9/Real.exp 1) := by
  sorry

end min_distance_between_curves_l2655_265569


namespace min_triangle_area_l2655_265589

/-- The minimum non-zero area of a triangle with vertices (0,0), (50,20), and (p,q),
    where p and q are integers. -/
theorem min_triangle_area :
  ∀ p q : ℤ,
  let area := (1/2 : ℝ) * |20 * p - 50 * q|
  ∃ p' q' : ℤ,
    (area > 0 → area ≥ 15) ∧
    (∃ a : ℝ, a > 0 ∧ a < 15 → ¬∃ p'' q'' : ℤ, (1/2 : ℝ) * |20 * p'' - 50 * q''| = a) :=
by sorry

end min_triangle_area_l2655_265589


namespace sample_size_theorem_l2655_265537

/-- Represents the types of products produced by the factory -/
inductive ProductType
  | A
  | B
  | C

/-- Represents the quantity ratio of products -/
def quantity_ratio : ProductType → ℕ
  | ProductType.A => 2
  | ProductType.B => 3
  | ProductType.C => 5

/-- Calculates the total ratio sum -/
def total_ratio : ℕ := quantity_ratio ProductType.A + quantity_ratio ProductType.B + quantity_ratio ProductType.C

/-- Represents the number of Type B products in the sample -/
def type_b_sample : ℕ := 24

/-- Theorem: If 24 units of Type B are drawn in a stratified random sample 
    from a production with ratio 2:3:5, then the total sample size is 80 -/
theorem sample_size_theorem : 
  (type_b_sample * total_ratio) / quantity_ratio ProductType.B = 80 := by
  sorry

end sample_size_theorem_l2655_265537


namespace max_vector_sum_on_unit_circle_l2655_265508

theorem max_vector_sum_on_unit_circle :
  let A : ℝ × ℝ := (Real.sqrt 3, 1)
  let O : ℝ × ℝ := (0, 0)
  ∃ (max : ℝ), max = 3 ∧ 
    ∀ (B : ℝ × ℝ), (B.1 - O.1)^2 + (B.2 - O.2)^2 = 1 →
      Real.sqrt ((A.1 - O.1 + B.1 - O.1)^2 + (A.2 - O.2 + B.2 - O.2)^2) ≤ max :=
by sorry

end max_vector_sum_on_unit_circle_l2655_265508


namespace cos_225_degrees_l2655_265539

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l2655_265539


namespace average_marks_combined_classes_l2655_265579

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 90) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 71.25 := by
  sorry

end average_marks_combined_classes_l2655_265579


namespace no_solution_to_inequalities_l2655_265528

theorem no_solution_to_inequalities :
  ¬∃ (x y z t : ℝ),
    (abs x < abs (y - z + t)) ∧
    (abs y < abs (x - z + t)) ∧
    (abs z < abs (x - y + t)) ∧
    (abs t < abs (x - y + z)) :=
by sorry

end no_solution_to_inequalities_l2655_265528


namespace semicircle_square_properties_l2655_265574

-- Define the semicircle and inscribed square
def semicircle_with_square (a b : ℝ) : Prop :=
  ∃ (A B C D E F : ℝ × ℝ),
    -- A and B are endpoints of the diameter
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    -- CDEF is a square with side length 1
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 1 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = 1 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = 1 ∧
    (F.1 - C.1)^2 + (F.2 - C.2)^2 = 1 ∧
    -- AC = a and BC = b
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2

-- State the theorem
theorem semicircle_square_properties (a b : ℝ) (h : semicircle_with_square a b) :
  a - b = 1 ∧ a * b = 1 ∧ a + b = Real.sqrt 5 ∧ a^2 + b^2 ≠ 5 := by
  sorry

end semicircle_square_properties_l2655_265574


namespace arithmetic_progression_implies_equal_numbers_l2655_265591

theorem arithmetic_progression_implies_equal_numbers
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h_arith_prog : (a + b) / 2 = (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) :
  a = b :=
sorry

end arithmetic_progression_implies_equal_numbers_l2655_265591


namespace probability_sum_less_than_4_l2655_265511

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a point satisfies a condition within a given square --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The condition x + y < 4 --/
def sumLessThan4 (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_sum_less_than_4 :
  let s : Square := { bottomLeft := (0, 0), topRight := (3, 3) }
  probability s sumLessThan4 = 1/2 := by
  sorry

end probability_sum_less_than_4_l2655_265511


namespace deepak_age_l2655_265581

/-- Given that the ratio of Rahul's age to Deepak's age is 4:3, 
    and Rahul's age after 4 years will be 32, 
    prove that Deepak's current age is 21 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 4 = 32 →
  deepak_age = 21 := by
  sorry

end deepak_age_l2655_265581


namespace factor_in_range_l2655_265538

theorem factor_in_range : ∃ m : ℕ, 
  (201212200619 : ℕ) % m = 0 ∧ 
  (6 * 10^9 : ℕ) < m ∧ 
  m < (13 * 10^9 : ℕ) / 2 ∧
  m = 6490716149 := by
sorry

end factor_in_range_l2655_265538


namespace factorization_problem_1_factorization_problem_2_l2655_265548

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  a^2 * (a - b) + 4 * b^2 * (b - a) = (a - b) * (a + 2*b) * (a - 2*b) := by sorry

-- Problem 2
theorem factorization_problem_2 (m : ℝ) :
  m^4 - 1 = (m^2 + 1) * (m + 1) * (m - 1) := by sorry

end factorization_problem_1_factorization_problem_2_l2655_265548


namespace unique_solution_quadratic_l2655_265535

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 25 * x^2 + n * x + 4 = 0) ↔ n = 20 := by
sorry

end unique_solution_quadratic_l2655_265535


namespace solve_equation_l2655_265519

theorem solve_equation : ∃ x : ℚ, (3/4 : ℚ) - (1/2 : ℚ) = 1/x ∧ x = 4 := by sorry

end solve_equation_l2655_265519


namespace geometric_sequence_problem_l2655_265544

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  a 3 * a 5 = 4 * (a 6)^2 →
  a 3 = 1/2 := by
sorry

end geometric_sequence_problem_l2655_265544


namespace equation_solution_a_l2655_265518

theorem equation_solution_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) → 
  (a = 0 ∨ a = 1) := by
sorry

end equation_solution_a_l2655_265518


namespace unique_a_l2655_265546

/-- The equation is quadratic in x -/
def is_quadratic (a : ℝ) : Prop :=
  |a - 1| = 2

/-- The coefficient of the quadratic term is non-zero -/
def coeff_nonzero (a : ℝ) : Prop :=
  a - 3 ≠ 0

/-- The value of a that satisfies the conditions -/
theorem unique_a : ∃! a : ℝ, is_quadratic a ∧ coeff_nonzero a :=
  sorry

end unique_a_l2655_265546


namespace root_sum_reciprocal_l2655_265536

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (p^3 - 25*p^2 + 90*p - 73 = 0) →
  (q^3 - 25*q^2 + 90*q - 73 = 0) →
  (r^3 - 25*r^2 + 90*r - 73 = 0) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 25*s^2 + 90*s - 73) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 256 :=
by sorry

end root_sum_reciprocal_l2655_265536


namespace chefs_wage_difference_l2655_265520

theorem chefs_wage_difference (dishwasher1_wage dishwasher2_wage dishwasher3_wage : ℚ)
  (chef1_percentage chef2_percentage chef3_percentage : ℚ)
  (manager_wage : ℚ) :
  dishwasher1_wage = 6 →
  dishwasher2_wage = 7 →
  dishwasher3_wage = 8 →
  chef1_percentage = 1.2 →
  chef2_percentage = 1.25 →
  chef3_percentage = 1.3 →
  manager_wage = 12.5 →
  manager_wage - (dishwasher1_wage * chef1_percentage + 
                  dishwasher2_wage * chef2_percentage + 
                  dishwasher3_wage * chef3_percentage) = 13.85 := by
  sorry

end chefs_wage_difference_l2655_265520


namespace sample_xy_product_l2655_265566

theorem sample_xy_product (x y : ℝ) : 
  (9 + 10 + 11 + x + y) / 5 = 10 →
  ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 2 →
  x * y = 96 := by
sorry

end sample_xy_product_l2655_265566


namespace a_equals_two_l2655_265549

theorem a_equals_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) : a = 2 := by
  sorry

end a_equals_two_l2655_265549


namespace geometric_sequence_product_l2655_265529

theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 2 →                            -- First term is 2
  a 5 = 8 →                            -- Fifth term is 8
  a 2 * a 3 * a 4 = 64 := by            -- Product of middle terms is 64
sorry

end geometric_sequence_product_l2655_265529


namespace parabola_directrix_l2655_265558

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop :=
  y = -1/2

/-- Theorem: The directrix of the given parabola is y = -1/2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end parabola_directrix_l2655_265558


namespace discount_order_matters_l2655_265562

def original_price : ℚ := 50
def fixed_discount : ℚ := 10
def percentage_discount : ℚ := 0.25

def price_fixed_then_percentage : ℚ := (original_price - fixed_discount) * (1 - percentage_discount)
def price_percentage_then_fixed : ℚ := (original_price * (1 - percentage_discount)) - fixed_discount

theorem discount_order_matters :
  price_percentage_then_fixed < price_fixed_then_percentage ∧
  (price_fixed_then_percentage - price_percentage_then_fixed) * 100 = 250 := by
  sorry

end discount_order_matters_l2655_265562


namespace range_of_n_minus_m_l2655_265587

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x - 1 else (3/2) * x + 1

theorem range_of_n_minus_m (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  2/3 < n - m ∧ n - m ≤ Real.log (3/2) + 1/3 :=
sorry

end range_of_n_minus_m_l2655_265587


namespace second_digit_of_three_digit_number_l2655_265551

/-- Given a three-digit number xyz, if 100x + 10y + z - (x + y + z) = 261, then y = 7 -/
theorem second_digit_of_three_digit_number (x y z : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ z ≥ 0 ∧ z ≤ 9 →
  100 * x + 10 * y + z - (x + y + z) = 261 →
  y = 7 := by
  sorry

end second_digit_of_three_digit_number_l2655_265551


namespace day_after_2_pow_20_l2655_265582

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week after a given number of days from Monday -/
def dayAfter (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => DayOfWeek.Monday
  | 1 => DayOfWeek.Tuesday
  | 2 => DayOfWeek.Wednesday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Friday
  | 5 => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday

/-- Theorem: After 2^20 days from Monday, it will be Friday -/
theorem day_after_2_pow_20 : dayAfter (2^20) = DayOfWeek.Friday := by
  sorry


end day_after_2_pow_20_l2655_265582


namespace jane_vases_last_day_l2655_265598

/-- The number of vases Jane arranges on the last day given her daily rate, total vases, and total days --/
def vases_on_last_day (daily_rate : ℕ) (total_vases : ℕ) (total_days : ℕ) : ℕ :=
  if total_vases ≤ daily_rate * (total_days - 1)
  then 0
  else total_vases - daily_rate * (total_days - 1)

theorem jane_vases_last_day :
  vases_on_last_day 25 378 17 = 0 := by
  sorry

end jane_vases_last_day_l2655_265598


namespace table_sticks_prove_table_sticks_l2655_265541

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The theorem stating that a table makes 9 sticks of wood -/
theorem table_sticks : ℕ :=
  let total_sticks := hours_warm * sticks_per_hour
  let chair_total := chairs_chopped * chair_sticks
  let stool_total := stools_chopped * stool_sticks
  let table_total := total_sticks - chair_total - stool_total
  table_total / tables_chopped

/-- Proof of the theorem -/
theorem prove_table_sticks : table_sticks = 9 := by
  sorry


end table_sticks_prove_table_sticks_l2655_265541


namespace correct_bird_count_l2655_265505

/-- Given a number of feet on tree branches and the number of feet per bird,
    calculate the number of birds on the tree. -/
def birds_on_tree (total_feet : ℕ) (feet_per_bird : ℕ) : ℕ :=
  total_feet / feet_per_bird

theorem correct_bird_count : birds_on_tree 92 2 = 46 := by
  sorry

end correct_bird_count_l2655_265505


namespace total_money_is_correct_l2655_265595

/-- Calculates the total amount of money in Euros given the specified coins and bills and the conversion rate. -/
def total_money_in_euros : ℝ :=
  let pennies : ℕ := 9
  let nickels : ℕ := 4
  let dimes : ℕ := 3
  let quarters : ℕ := 7
  let half_dollars : ℕ := 5
  let one_dollar_coins : ℕ := 2
  let two_dollar_bills : ℕ := 1
  
  let penny_value : ℝ := 0.01
  let nickel_value : ℝ := 0.05
  let dime_value : ℝ := 0.10
  let quarter_value : ℝ := 0.25
  let half_dollar_value : ℝ := 0.50
  let one_dollar_value : ℝ := 1.00
  let two_dollar_value : ℝ := 2.00
  
  let usd_to_euro_rate : ℝ := 0.85
  
  let total_usd : ℝ := 
    pennies * penny_value +
    nickels * nickel_value +
    dimes * dime_value +
    quarters * quarter_value +
    half_dollars * half_dollar_value +
    one_dollar_coins * one_dollar_value +
    two_dollar_bills * two_dollar_value
  
  total_usd * usd_to_euro_rate

/-- Theorem stating that the total amount of money in Euros is equal to 7.514. -/
theorem total_money_is_correct : total_money_in_euros = 7.514 := by
  sorry

end total_money_is_correct_l2655_265595


namespace triangle_theorem_l2655_265588

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - t.a * t.b ∧
  t.b = 2 ∧
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ t.a = 3 := by
  sorry

end triangle_theorem_l2655_265588


namespace vectors_parallel_iff_y_eq_neg_one_l2655_265521

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- Vector a -/
def a : ℝ × ℝ := (1, 2)

/-- Vector b parameterized by y -/
def b (y : ℝ) : ℝ × ℝ := (1, -2*y)

/-- Theorem: Vectors a and b are parallel if and only if y = -1 -/
theorem vectors_parallel_iff_y_eq_neg_one :
  ∀ y : ℝ, are_parallel a (b y) ↔ y = -1 := by sorry

end vectors_parallel_iff_y_eq_neg_one_l2655_265521


namespace shopkeeper_gain_percentage_l2655_265576

/-- The gain percentage of a shopkeeper using false weights -/
theorem shopkeeper_gain_percentage 
  (claimed_weight : ℝ) 
  (actual_weight : ℝ) 
  (claimed_weight_is_kg : claimed_weight = 1000) 
  (actual_weight_used : actual_weight = 980) : 
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  (1000 - 980) / 980 * 100 := by
sorry

end shopkeeper_gain_percentage_l2655_265576


namespace angle_system_solution_l2655_265500

theorem angle_system_solution (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (eq1 : 2 * Real.sin (2 * β) = 3 * Real.sin (2 * α))
  (eq2 : Real.tan β = 3 * Real.tan α) :
  α = Real.arctan (Real.sqrt 7 / 7) ∧ β = Real.arctan (3 * Real.sqrt 7 / 7) := by
sorry

end angle_system_solution_l2655_265500


namespace symmetry_line_l2655_265594

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 7)^2 + (y + 4)^2 = 16
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + (y - 6)^2 = 16

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := 6*x - 5*y - 1 = 0

-- Theorem statement
theorem symmetry_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  circle1 x₁ y₁ → circle2 x₂ y₂ →
  ∃ (x y : ℝ),
  line_of_symmetry x y ∧
  (x = (x₁ + x₂) / 2) ∧
  (y = (y₁ + y₂) / 2) ∧
  ((x - x₁)^2 + (y - y₁)^2 = (x - x₂)^2 + (y - y₂)^2) :=
by sorry

end symmetry_line_l2655_265594


namespace cuboid_volumes_sum_l2655_265570

theorem cuboid_volumes_sum (length width height1 height2 : ℝ) 
  (h1 : length = 44)
  (h2 : width = 35)
  (h3 : height1 = 7)
  (h4 : height2 = 3) :
  length * width * height1 + length * width * height2 = 15400 := by
  sorry

end cuboid_volumes_sum_l2655_265570


namespace count_four_digit_numbers_l2655_265563

/-- The number of ways to select 3 different digits from 0 to 9 -/
def select_three_digits : ℕ := Nat.choose 10 3

/-- The number of four-digit numbers formed by selecting three different digits from 0 to 9,
    where one digit may appear twice -/
def four_digit_numbers : ℕ := 3888

/-- Theorem stating that the number of four-digit numbers formed by selecting
    three different digits from 0 to 9 (where one digit may appear twice) is 3888 -/
theorem count_four_digit_numbers :
  four_digit_numbers = 3888 :=
by sorry

end count_four_digit_numbers_l2655_265563


namespace stone_breadth_proof_l2655_265502

/-- Given a hall and stones with specific dimensions, prove the breadth of each stone. -/
theorem stone_breadth_proof (hall_length : ℝ) (hall_width : ℝ) (stone_length : ℝ) (stone_count : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_length = 0.8)
  (h4 : stone_count = 1350) :
  ∃ (stone_width : ℝ), 
    stone_width = 0.5 ∧ 
    (hall_length * hall_width * 100) = (stone_count : ℝ) * stone_length * stone_width * 100 := by
  sorry


end stone_breadth_proof_l2655_265502


namespace rational_solutions_count_l2655_265542

theorem rational_solutions_count (p : ℕ) (hp : Prime p) :
  let f : ℚ → ℚ := λ x => x^4 + (2 - p : ℚ)*x^3 + (2 - 2*p : ℚ)*x^2 + (1 - 2*p : ℚ)*x - p
  (∃ (s : Finset ℚ), s.card = 2 ∧ (∀ x ∈ s, f x = 0) ∧ (∀ x, f x = 0 → x ∈ s)) := by
  sorry

end rational_solutions_count_l2655_265542


namespace complex_equation_solution_l2655_265501

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I : ℂ) * 2 + 1 = (Complex.I + 1) * (Complex.I * b + a) →
  a = 3/2 ∧ b = 1/2 :=
sorry

end complex_equation_solution_l2655_265501


namespace work_multiple_proof_l2655_265577

/-- Represents the time taken to complete a job given the number of workers and the fraction of the job to be completed -/
def time_to_complete (num_workers : ℕ) (job_fraction : ℚ) (base_time : ℕ) : ℚ :=
  (job_fraction * base_time) / num_workers

theorem work_multiple_proof (base_workers : ℕ) (h : base_workers > 0) :
  time_to_complete base_workers 1 12 = 12 →
  time_to_complete (2 * base_workers) (1/2) 12 = 3 := by
sorry

end work_multiple_proof_l2655_265577


namespace inequalities_system_k_range_l2655_265592

theorem inequalities_system_k_range :
  ∀ k : ℚ,
  (∀ x : ℤ, x^2 - x - 2 > 0 ∧ 2*x^2 + (5 + 2*k)*x + 5 < 0 ↔ x = -2) →
  3/4 < k ∧ k ≤ 4/3 :=
by sorry

end inequalities_system_k_range_l2655_265592


namespace moon_mission_cost_share_l2655_265572

/-- Calculates the individual share of a total cost divided equally among a population -/
def individual_share (total_cost : ℕ) (population : ℕ) : ℚ :=
  (total_cost : ℚ) / (population : ℚ)

/-- Proves that the individual share of 40 billion dollars among 200 million people is 200 dollars -/
theorem moon_mission_cost_share :
  individual_share (40 * 10^9) (200 * 10^6) = 200 := by
  sorry

end moon_mission_cost_share_l2655_265572


namespace modulus_of_one_over_one_minus_i_l2655_265507

theorem modulus_of_one_over_one_minus_i :
  let z : ℂ := 1 / (1 - I)
  ‖z‖ = Real.sqrt 2 / 2 := by sorry

end modulus_of_one_over_one_minus_i_l2655_265507


namespace cherie_boxes_count_l2655_265552

/-- The number of boxes Koby bought -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in each of Cherie's boxes -/
def cherie_sparklers_per_box : ℕ := 8

/-- The number of whistlers in each of Cherie's boxes -/
def cherie_whistlers_per_box : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of boxes Cherie bought -/
def cherie_boxes : ℕ := 1

theorem cherie_boxes_count : 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) + 
  cherie_boxes * (cherie_sparklers_per_box + cherie_whistlers_per_box) = 
  total_fireworks :=
by sorry

end cherie_boxes_count_l2655_265552


namespace max_area_fence_enclosure_l2655_265524

/-- Represents a rectangular fence enclosure --/
structure FenceEnclosure where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 200
  length_constraint : length ≥ 90
  width_constraint : width ≥ 50
  ratio_constraint : length ≤ 2 * width

/-- The area of a fence enclosure --/
def area (f : FenceEnclosure) : ℝ := f.length * f.width

/-- Theorem stating the maximum area of the fence enclosure --/
theorem max_area_fence_enclosure :
  ∃ (f : FenceEnclosure), ∀ (g : FenceEnclosure), area f ≥ area g ∧ area f = 10000 :=
sorry

end max_area_fence_enclosure_l2655_265524


namespace union_equals_A_l2655_265532

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}

theorem union_equals_A (a : ℝ) : (A ∪ B a = A) → (a = 2 ∨ a = 3) := by
  sorry

end union_equals_A_l2655_265532


namespace power_zero_eq_one_l2655_265571

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_eq_one_l2655_265571


namespace airline_passengers_l2655_265543

/-- Given an airline where each passenger can take 8 pieces of luggage,
    and a total of 32 bags, prove that 4 people were flying. -/
theorem airline_passengers (bags_per_person : ℕ) (total_bags : ℕ) (num_people : ℕ) : 
  bags_per_person = 8 →
  total_bags = 32 →
  num_people * bags_per_person = total_bags →
  num_people = 4 := by
sorry

end airline_passengers_l2655_265543


namespace probability_of_two_primes_l2655_265527

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of integers from 1 to 30 inclusive -/
def integerSet : Finset ℕ := sorry

/-- The set of prime numbers from 1 to 30 inclusive -/
def primeSet : Finset ℕ := sorry

/-- The number of ways to choose 2 items from a set of size n -/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

theorem probability_of_two_primes :
  (choose (Finset.card primeSet) 2 : ℚ) / (choose (Finset.card integerSet) 2) = 10 / 87 := by
  sorry

end probability_of_two_primes_l2655_265527


namespace festival_allowance_days_l2655_265550

/-- Calculates the maximum number of full days for festival allowance --/
def maxAllowanceDays (staffCount : Nat) (dailyRate : Nat) (totalAmount : Nat) (pettyCashAmount : Nat) : Nat :=
  let totalAvailable := totalAmount + pettyCashAmount
  (totalAvailable - pettyCashAmount) / (staffCount * dailyRate)

theorem festival_allowance_days :
  maxAllowanceDays 20 100 65000 1000 = 32 := by
  sorry

end festival_allowance_days_l2655_265550


namespace puzzle_border_pieces_l2655_265559

theorem puzzle_border_pieces (total_pieces : ℕ) (trevor_pieces : ℕ) (missing_pieces : ℕ) : 
  total_pieces = 500 → 
  trevor_pieces = 105 → 
  missing_pieces = 5 → 
  (total_pieces - missing_pieces - trevor_pieces - 3 * trevor_pieces) = 75 :=
by sorry

end puzzle_border_pieces_l2655_265559


namespace blue_balls_unchanged_l2655_265530

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  red : Nat
  blue : Nat
  yellow : Nat

/-- The operation of adding yellow balls to the box -/
def addYellowBalls (initial : BallCount) (added : Nat) : BallCount :=
  { red := initial.red,
    blue := initial.blue,
    yellow := initial.yellow + added }

theorem blue_balls_unchanged (initial : BallCount) (added : Nat) :
  (addYellowBalls initial added).blue = initial.blue :=
by sorry

end blue_balls_unchanged_l2655_265530


namespace probability_of_b_in_rabbit_l2655_265522

def word : String := "rabbit"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_of_b_in_rabbit :
  (count_letter word 'b' : ℚ) / word.length = 1 / 3 := by
  sorry

end probability_of_b_in_rabbit_l2655_265522


namespace expression_evaluation_l2655_265583

theorem expression_evaluation (x : ℝ) (h : x = -2) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = -1 := by sorry

end expression_evaluation_l2655_265583


namespace probability_at_least_one_diamond_or_ace_l2655_265506

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of cards that are either diamonds or aces -/
def targetCards : ℕ := 16

/-- The probability of drawing a card that is neither a diamond nor an ace -/
def probNonTarget : ℚ := (deckSize - targetCards) / deckSize

/-- The number of draws -/
def numDraws : ℕ := 3

theorem probability_at_least_one_diamond_or_ace :
  1 - probNonTarget ^ numDraws = 1468 / 2197 := by sorry

end probability_at_least_one_diamond_or_ace_l2655_265506


namespace brick_wall_theorem_l2655_265573

/-- Calculates the total number of bricks in a wall with a given number of rows,
    where each row has one less brick than the row below it. -/
def total_bricks (rows : ℕ) (bottom_row_bricks : ℕ) : ℕ :=
  (2 * bottom_row_bricks - rows + 1) * rows / 2

/-- Theorem stating that a wall with 5 rows, 18 bricks in the bottom row,
    and each row having one less brick than the row below it,
    has a total of 80 bricks. -/
theorem brick_wall_theorem :
  total_bricks 5 18 = 80 := by
  sorry

end brick_wall_theorem_l2655_265573


namespace existence_of_integers_l2655_265596

theorem existence_of_integers : ∃ (list : List Int), 
  (list.length = 2016) ∧ 
  (list.prod = 9) ∧ 
  (list.sum = 0) := by
  sorry

end existence_of_integers_l2655_265596


namespace isosceles_triangle_from_cosine_condition_l2655_265586

/-- Given a triangle ABC where a*cos(B) = b*cos(A), prove that the triangle is isosceles -/
theorem isosceles_triangle_from_cosine_condition (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_cosine : a * Real.cos B = b * Real.cos A) : 
  a = b ∨ b = c ∨ a = c :=
sorry

end isosceles_triangle_from_cosine_condition_l2655_265586


namespace bug_meeting_time_l2655_265554

theorem bug_meeting_time (r₁ r₂ v₁ v₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) : ∃ t : ℝ, t = 2.5 ∧ 
  (∃ n₁ n₂ : ℕ, t * v₁ = 2 * Real.pi * r₁ * n₁ ∧ 
   t * v₂ = 2 * Real.pi * r₂ * (n₂ + 1/4)) := by
  sorry

end bug_meeting_time_l2655_265554


namespace only_parallel_converse_true_l2655_265567

-- Define the basic concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define properties and relations
def parallel (l1 l2 : Line) : Prop := sorry
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def isosceles (t : Triangle) : Prop := sorry
def acute (t : Triangle) : Prop := sorry
def rightAngle (a : Angle) : Prop := sorry
def correspondingAngles (a1 a2 : Angle) (t1 t2 : Triangle) : Prop := sorry
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that only the converse of proposition B is true
theorem only_parallel_converse_true :
  (∀ t : Triangle, acute t → isosceles t) = False ∧
  (∀ l1 l2 : Line, ∀ a1 a2 : Angle, alternateInteriorAngles a1 a2 l1 l2 → parallel l1 l2) = True ∧
  (∀ t1 t2 : Triangle, ∀ a1 a2 : Angle, correspondingAngles a1 a2 t1 t2 → congruent t1 t2) = False ∧
  (∀ a1 a2 : Angle, a1 = a2 → rightAngle a1 ∧ rightAngle a2) = False :=
sorry

end only_parallel_converse_true_l2655_265567


namespace triangle_problem_l2655_265545

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute angles
  A + B + C = π →  -- angles sum to π
  b = Real.sqrt 2 * a * Real.sin B →  -- given condition
  (A = π/4 ∧ 
   (b = Real.sqrt 6 ∧ c = Real.sqrt 3 + 1 → a = 2)) :=
by sorry

end triangle_problem_l2655_265545


namespace regular_polygon_perimeter_l2655_265531

/-- A regular polygon with side length 6 units and exterior angle 60 degrees has a perimeter of 36 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (h1 : s = 6) (h2 : θ = 60) :
  let n : ℝ := 360 / θ
  s * n = 36 := by
sorry

end regular_polygon_perimeter_l2655_265531


namespace decagon_diagonal_intersections_l2655_265526

def regular_decagon : Nat := 10

/-- The number of distinct interior points where two or more diagonals intersect in a regular decagon -/
def intersection_points (n : Nat) : Nat :=
  Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points regular_decagon = 210 := by
  sorry

end decagon_diagonal_intersections_l2655_265526


namespace old_clock_slower_by_12_minutes_l2655_265525

/-- Represents the time interval between consecutive coincidences of hour and minute hands -/
def coincidence_interval : ℕ := 66

/-- Represents the number of coincidences in a 24-hour period -/
def coincidences_per_day : ℕ := 22

/-- Represents the number of minutes in a standard 24-hour day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Represents the number of minutes in the old clock's 24-hour period -/
def old_clock_day_minutes : ℕ := coincidence_interval * coincidences_per_day

theorem old_clock_slower_by_12_minutes :
  old_clock_day_minutes - standard_day_minutes = 12 := by sorry

end old_clock_slower_by_12_minutes_l2655_265525


namespace smallest_base_for_256_is_correct_l2655_265504

/-- The smallest base in which 256 (decimal) has exactly 4 digits -/
def smallest_base_for_256 : ℕ := 5

/-- Predicate to check if a number has exactly 4 digits in a given base -/
def has_exactly_four_digits (n : ℕ) (base : ℕ) : Prop :=
  base ^ 3 ≤ n ∧ n < base ^ 4

theorem smallest_base_for_256_is_correct :
  (has_exactly_four_digits 256 smallest_base_for_256) ∧
  (∀ b : ℕ, 0 < b → b < smallest_base_for_256 → ¬(has_exactly_four_digits 256 b)) :=
by sorry

end smallest_base_for_256_is_correct_l2655_265504


namespace perpendicular_and_parallel_lines_planes_l2655_265515

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perp (l1 l2 : Line) : Prop := sorry
def para (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_and_parallel_lines_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : contained_in n β) :
  (parallel α β → perp m n) ∧ (para m n → perpendicular α β) := by
  sorry

end perpendicular_and_parallel_lines_planes_l2655_265515
