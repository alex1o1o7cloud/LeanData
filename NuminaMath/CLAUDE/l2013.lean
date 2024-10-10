import Mathlib

namespace sum_set_cardinality_l2013_201331

/-- A function that generates an arithmetic sequence with a given first term, common difference, and length. -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  fun i => a₁ + d * (i : ℝ)

/-- The set of all sums of pairs of elements from the arithmetic sequence. -/
def sumSet (a₁ : ℝ) (d : ℝ) (n : ℕ) : Set ℝ :=
  {x | ∃ (i j : Fin n), i ≤ j ∧ x = arithmeticSequence a₁ d n i + arithmeticSequence a₁ d n j}

/-- The theorem stating that the number of elements in the sum set is 2n - 3. -/
theorem sum_set_cardinality (a₁ : ℝ) (d : ℝ) (n : ℕ) (h₁ : n ≥ 3) (h₂ : d > 0) :
  Nat.card (sumSet a₁ d n) = 2 * n - 3 :=
sorry

end sum_set_cardinality_l2013_201331


namespace cost_per_flower_is_15_l2013_201388

/-- Represents the number of centerpieces -/
def num_centerpieces : ℕ := 6

/-- Represents the number of roses per centerpiece -/
def roses_per_centerpiece : ℕ := 8

/-- Represents the number of lilies per centerpiece -/
def lilies_per_centerpiece : ℕ := 6

/-- Represents the total budget in dollars -/
def total_budget : ℕ := 2700

/-- Calculates the total number of roses -/
def total_roses : ℕ := num_centerpieces * roses_per_centerpiece

/-- Calculates the total number of orchids -/
def total_orchids : ℕ := 2 * total_roses

/-- Calculates the total number of lilies -/
def total_lilies : ℕ := num_centerpieces * lilies_per_centerpiece

/-- Calculates the total number of flowers -/
def total_flowers : ℕ := total_roses + total_orchids + total_lilies

/-- Theorem: The cost per flower is $15 -/
theorem cost_per_flower_is_15 : total_budget / total_flowers = 15 := by
  sorry


end cost_per_flower_is_15_l2013_201388


namespace f_odd_and_decreasing_l2013_201302

def f (x : ℝ) : ℝ := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end f_odd_and_decreasing_l2013_201302


namespace rectangle_dimensions_l2013_201341

theorem rectangle_dimensions (x : ℝ) : 
  (x + 1 > 0) → 
  (3*x - 4 > 0) → 
  (x + 1) * (3*x - 4) = 12*x - 19 → 
  x = (13 + Real.sqrt 349) / 6 := by
sorry

end rectangle_dimensions_l2013_201341


namespace stream_speed_l2013_201332

/-- 
Given a canoe that rows upstream at 8 km/hr and downstream at 12 km/hr, 
this theorem proves that the speed of the stream is 2 km/hr.
-/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 8)
  (h_downstream : downstream_speed = 12) :
  let canoe_speed := (upstream_speed + downstream_speed) / 2
  let stream_speed := (downstream_speed - upstream_speed) / 2
  stream_speed = 2 := by sorry

end stream_speed_l2013_201332


namespace airplane_seating_l2013_201300

/-- A proof problem about airplane seating --/
theorem airplane_seating (first_class business_class economy_class : ℕ) 
  (h1 : first_class = 10)
  (h2 : business_class = 30)
  (h3 : economy_class = 50)
  (h4 : economy_class / 2 = first_class + (business_class - (business_class - x)))
  (h5 : first_class - 7 = 3)
  (x : ℕ) :
  x = 8 := by sorry

end airplane_seating_l2013_201300


namespace beaker_liquid_distribution_l2013_201353

/-- Proves that if 5 ml of liquid is removed from a beaker and 35 ml remains, 
    then the original amount of liquid would have been 8 ml per cup if equally 
    distributed among 5 cups. -/
theorem beaker_liquid_distribution (initial_volume : ℝ) : 
  initial_volume - 5 = 35 → initial_volume / 5 = 8 := by
  sorry

end beaker_liquid_distribution_l2013_201353


namespace wire_length_l2013_201384

/-- Represents the lengths of five wire pieces in a specific ratio --/
structure WirePieces where
  ratio : Fin 5 → ℕ
  shortest : ℝ
  total : ℝ

/-- The wire pieces satisfy the given conditions --/
def satisfies_conditions (w : WirePieces) : Prop :=
  w.ratio 0 = 4 ∧
  w.ratio 1 = 5 ∧
  w.ratio 2 = 7 ∧
  w.ratio 3 = 3 ∧
  w.ratio 4 = 2 ∧
  w.shortest = 16

/-- Theorem stating the total length of the wire --/
theorem wire_length (w : WirePieces) (h : satisfies_conditions w) : w.total = 84 := by
  sorry


end wire_length_l2013_201384


namespace amp_five_two_l2013_201303

-- Define the & operation
def amp (a b : ℤ) : ℤ := ((a + b) * (a - b))^2

-- Theorem statement
theorem amp_five_two : amp 5 2 = 441 := by
  sorry

end amp_five_two_l2013_201303


namespace shaded_fraction_is_seven_sixteenths_l2013_201321

/-- Represents a square divided into smaller squares and triangles -/
structure DividedSquare where
  /-- The number of smaller squares the large square is divided into -/
  num_small_squares : ℕ
  /-- The number of triangles each smaller square is divided into -/
  triangles_per_small_square : ℕ
  /-- The total number of shaded triangles -/
  shaded_triangles : ℕ

/-- Calculates the fraction of the square that is shaded -/
def shaded_fraction (s : DividedSquare) : ℚ :=
  s.shaded_triangles / (s.num_small_squares * s.triangles_per_small_square)

/-- Theorem stating that the shaded fraction of the given square is 7/16 -/
theorem shaded_fraction_is_seven_sixteenths (s : DividedSquare) 
  (h1 : s.num_small_squares = 4)
  (h2 : s.triangles_per_small_square = 4)
  (h3 : s.shaded_triangles = 7) : 
  shaded_fraction s = 7 / 16 := by
  sorry

end shaded_fraction_is_seven_sixteenths_l2013_201321


namespace function_value_comparison_l2013_201333

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem function_value_comparison
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f a b c (x + 1) = f a b c (1 - x)) :
  f a b c (Real.arcsin (1/3)) > f a b c (Real.arcsin (2/3)) :=
by sorry

end function_value_comparison_l2013_201333


namespace a_3_eq_35_l2013_201365

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℕ := 5 * n ^ 2 + 10 * n

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := S n - S (n - 1)

/-- Theorem: The third term of the sequence is 35 -/
theorem a_3_eq_35 : a 3 = 35 := by
  sorry

end a_3_eq_35_l2013_201365


namespace max_value_complex_expression_l2013_201393

theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  ∃ (M : ℝ), M = 12 ∧ ∀ z : ℂ, Complex.abs z = 2 →
    Complex.abs ((z - 2)^2 * (z + 2)) ≤ M ∧
    ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = M :=
by sorry

end max_value_complex_expression_l2013_201393


namespace gcd_90_405_l2013_201375

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l2013_201375


namespace painted_cube_problem_l2013_201317

theorem painted_cube_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 →
  n = 2 :=
by
  sorry

end painted_cube_problem_l2013_201317


namespace cubic_inequality_l2013_201373

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ a^2*b + b^2*c + c^2*a ∧
  (a^3 + b^3 + c^3 = a^2*b + b^2*c + c^2*a ↔ a = b ∧ b = c) :=
by sorry

end cubic_inequality_l2013_201373


namespace missy_additional_capacity_l2013_201310

/-- Proves that Missy can handle 15 more claims than John given the conditions -/
theorem missy_additional_capacity (jan_capacity : ℕ) (john_capacity : ℕ) (missy_capacity : ℕ) :
  jan_capacity = 20 →
  john_capacity = jan_capacity + (jan_capacity * 3 / 10) →
  missy_capacity = 41 →
  missy_capacity - john_capacity = 15 := by
  sorry

#check missy_additional_capacity

end missy_additional_capacity_l2013_201310


namespace ball_probability_l2013_201309

/-- Given a bag of balls with the specified conditions, prove the probability of choosing a ball that is neither red nor purple -/
theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h_total : total = 60)
  (h_red : red = 5)
  (h_purple : purple = 7) :
  (total - (red + purple)) / total = 4 / 5 := by
sorry

end ball_probability_l2013_201309


namespace arithmetic_sequence_problem_l2013_201324

/-- Given an arithmetic sequence {a_n} where a_5 = 3 and a_9 = 6, prove that a_13 = 9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_a5 : a 5 = 3) 
  (h_a9 : a 9 = 6) : 
  a 13 = 9 := by
sorry

end arithmetic_sequence_problem_l2013_201324


namespace equation_solution_l2013_201370

theorem equation_solution : ∃ x : ℚ, x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end equation_solution_l2013_201370


namespace quadratic_condition_necessary_not_sufficient_l2013_201363

theorem quadratic_condition_necessary_not_sufficient :
  (∀ b : ℝ, (∀ x : ℝ, x^2 - b*x + 1 > 0) → b ∈ Set.Icc 0 1) ∧
  ¬(∀ b : ℝ, b ∈ Set.Icc 0 1 → (∀ x : ℝ, x^2 - b*x + 1 > 0)) :=
by sorry

end quadratic_condition_necessary_not_sufficient_l2013_201363


namespace lily_received_35_books_l2013_201364

/-- The number of books Lily received -/
def books_lily_received (mike_books_tuesday : ℕ) (corey_books_tuesday : ℕ) (mike_gave : ℕ) (corey_gave_extra : ℕ) : ℕ :=
  mike_gave + (mike_gave + corey_gave_extra)

/-- Theorem stating that Lily received 35 books -/
theorem lily_received_35_books :
  ∀ (mike_books_tuesday corey_books_tuesday mike_gave corey_gave_extra : ℕ),
    mike_books_tuesday = 45 →
    corey_books_tuesday = 2 * mike_books_tuesday →
    mike_gave = 10 →
    corey_gave_extra = 15 →
    books_lily_received mike_books_tuesday corey_books_tuesday mike_gave corey_gave_extra = 35 := by
  sorry

#eval books_lily_received 45 90 10 15

end lily_received_35_books_l2013_201364


namespace nonzero_matrix_squared_zero_l2013_201396

theorem nonzero_matrix_squared_zero : 
  ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), A ≠ 0 ∧ A ^ 2 = 0 := by
  sorry

end nonzero_matrix_squared_zero_l2013_201396


namespace solve_for_y_l2013_201357

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 64) (h2 : x = 8) : y = 1 := by
  sorry

end solve_for_y_l2013_201357


namespace odd_increasing_function_inequality_l2013_201342

-- Define the properties of function f
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- State the theorem
theorem odd_increasing_function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_incr : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (x^2 - x + 1)} = 
  Set.union (Set.Iio 1) (Set.Ioi 2) := by sorry

end odd_increasing_function_inequality_l2013_201342


namespace budget_allocation_home_electronics_l2013_201313

theorem budget_allocation_home_electronics : 
  ∀ (total_budget : ℝ) (microphotonics food_additives gmo industrial_lubricants astrophysics home_electronics : ℝ),
  total_budget > 0 →
  microphotonics = 0.14 * total_budget →
  food_additives = 0.15 * total_budget →
  gmo = 0.19 * total_budget →
  industrial_lubricants = 0.08 * total_budget →
  astrophysics = (72 / 360) * total_budget →
  home_electronics + microphotonics + food_additives + gmo + industrial_lubricants + astrophysics = total_budget →
  home_electronics = 0.24 * total_budget :=
by
  sorry

end budget_allocation_home_electronics_l2013_201313


namespace function_max_min_difference_l2013_201344

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a^x else -x + a

-- State the theorem
theorem function_max_min_difference (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 2, f a x ≤ max) ∧
    (∀ x ∈ Set.Icc 0 2, f a x ≥ min) ∧
    (max - min = 5/2)) →
  (a = 1/2 ∨ a = 7/2) :=
by sorry

end function_max_min_difference_l2013_201344


namespace modular_inverse_seven_mod_thirtysix_l2013_201372

theorem modular_inverse_seven_mod_thirtysix : 
  ∃ x : ℤ, 0 ≤ x ∧ x < 36 ∧ (7 * x) % 36 = 1 :=
by
  use 31
  sorry

end modular_inverse_seven_mod_thirtysix_l2013_201372


namespace tree_house_wood_theorem_l2013_201374

/-- The total amount of wood needed for John's tree house -/
def total_wood_needed : ℝ :=
  let pillar_short := 4
  let pillar_long := 5 * pillar_short
  let wall_long := 6
  let wall_short := wall_long - 3
  let floor_wood := 5.5
  let roof_long := 2 * floor_wood
  let roof_short := 1.5 * floor_wood
  4 * pillar_short + 4 * pillar_long +
  10 * wall_long + 10 * wall_short +
  8 * floor_wood +
  6 * roof_long + 6 * roof_short

/-- Theorem stating the total amount of wood needed for John's tree house -/
theorem tree_house_wood_theorem : total_wood_needed = 345.5 := by
  sorry

end tree_house_wood_theorem_l2013_201374


namespace sqrt_product_equals_two_l2013_201387

theorem sqrt_product_equals_two : Real.sqrt 12 * Real.sqrt (1/3) = 2 := by
  sorry

end sqrt_product_equals_two_l2013_201387


namespace position_from_front_l2013_201316

theorem position_from_front (total : ℕ) (position_from_back : ℕ) (h1 : total = 22) (h2 : position_from_back = 13) :
  total - position_from_back + 1 = 10 := by
sorry

end position_from_front_l2013_201316


namespace button_probability_l2013_201334

/-- Represents the number of buttons of each color in a jar -/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- Represents the action of removing buttons from one jar to another -/
structure ButtonRemoval where
  removed : ℕ

theorem button_probability (initial_jar_a : JarContents) 
  (removal : ButtonRemoval) (final_jar_a : JarContents) :
  initial_jar_a.red = 4 →
  initial_jar_a.blue = 8 →
  removal.removed + removal.removed = initial_jar_a.red + initial_jar_a.blue - (final_jar_a.red + final_jar_a.blue) →
  3 * (final_jar_a.red + final_jar_a.blue) = 2 * (initial_jar_a.red + initial_jar_a.blue) →
  (final_jar_a.red / (final_jar_a.red + final_jar_a.blue : ℚ)) * 
  (removal.removed / ((initial_jar_a.red + initial_jar_a.blue - (final_jar_a.red + final_jar_a.blue)) : ℚ)) = 1/8 := by
  sorry


end button_probability_l2013_201334


namespace complex_equation_implication_l2013_201328

theorem complex_equation_implication (a b : ℝ) :
  let z : ℂ := a + b * Complex.I
  (z * (z + 2 * Complex.I) * (z + 4 * Complex.I) = 5000 * Complex.I) →
  (a^3 - a * (b^2 + 6*b + 8) - (b+6) * (b^2 + 6*b + 8) = 0 ∧
   a * (b+6) - b * (b^2 + 6*b + 8) = 5000) :=
by sorry

end complex_equation_implication_l2013_201328


namespace range_of_a_range_of_m_l2013_201394

/-- The function f(x) = |2x+1| + |2x-3| -/
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

/-- Theorem for the range of a -/
theorem range_of_a (a : ℝ) : (∀ x, f x > |1 - 3*a|) → -1 < a ∧ a < 5/3 := by sorry

/-- Theorem for the range of m -/
theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 4*Real.sqrt 2*t + f m = 0) → 
  -3/2 ≤ m ∧ m ≤ 5/2 := by sorry

end range_of_a_range_of_m_l2013_201394


namespace sum_equality_l2013_201345

theorem sum_equality (a b : Fin 2016 → ℝ) 
  (h1 : ∀ n ∈ Finset.range 2015, a (n + 1) = (1 / 65) * Real.sqrt (2 * (n + 1) + 2) + a n)
  (h2 : ∀ n ∈ Finset.range 2015, b (n + 1) = (1 / 1009) * Real.sqrt (2 * (n + 1) + 2) - b n)
  (h3 : a 0 = b 2015)
  (h4 : b 0 = a 2015) :
  (Finset.range 2015).sum (λ k => a (k + 1) * b k - a k * b (k + 1)) = 62 := by
sorry

end sum_equality_l2013_201345


namespace triangle_perimeter_l2013_201351

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 4000) (h2 : b = 3500) 
  (h3 : c^2 = a^2 - b^2) : a + b + c = 9437 := by
  sorry

end triangle_perimeter_l2013_201351


namespace all_sides_equal_l2013_201319

/-- A convex n-gon with equal interior angles and ordered sides -/
structure ConvexNGon (n : ℕ) where
  -- The sides of the n-gon
  sides : Fin n → ℝ
  -- All sides are non-negative
  sides_nonneg : ∀ i, 0 ≤ sides i
  -- The sides are ordered in descending order
  sides_ordered : ∀ i j, i ≤ j → sides j ≤ sides i
  -- The n-gon is convex
  convex : True
  -- All interior angles are equal
  equal_angles : True

/-- Theorem: In a convex n-gon with equal interior angles and ordered sides, all sides are equal -/
theorem all_sides_equal (n : ℕ) (ngon : ConvexNGon n) :
  ∀ i j : Fin n, ngon.sides i = ngon.sides j :=
sorry

end all_sides_equal_l2013_201319


namespace square_root_of_one_fourth_l2013_201399

theorem square_root_of_one_fourth : ∃ x : ℚ, x^2 = (1/4 : ℚ) ∧ x = -1/2 := by
  sorry

end square_root_of_one_fourth_l2013_201399


namespace correct_statements_l2013_201382

-- Define the statements
inductive Statement
| Synthesis1
| Synthesis2
| Analysis1
| Analysis2
| Contradiction

-- Define a function to check if a statement is correct
def is_correct (s : Statement) : Prop :=
  match s with
  | Statement.Synthesis1 => True  -- Synthesis is a method of cause and effect
  | Statement.Synthesis2 => True  -- Synthesis is a forward reasoning method
  | Statement.Analysis1 => True   -- Analysis is a method of seeking cause from effect
  | Statement.Analysis2 => False  -- Analysis is NOT an indirect proof method
  | Statement.Contradiction => False  -- Contradiction is NOT a backward reasoning method

-- Theorem to prove
theorem correct_statements :
  (is_correct Statement.Synthesis1) ∧
  (is_correct Statement.Synthesis2) ∧
  (is_correct Statement.Analysis1) ∧
  ¬(is_correct Statement.Analysis2) ∧
  ¬(is_correct Statement.Contradiction) :=
by sorry

#check correct_statements

end correct_statements_l2013_201382


namespace sphere_surface_area_rectangular_solid_l2013_201371

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 50 * Real.pi := by
  sorry

end sphere_surface_area_rectangular_solid_l2013_201371


namespace sum_of_integers_l2013_201311

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c - e = 7)
  (eq2 : b - c + d + e = 8)
  (eq3 : c - d + a - e = 4)
  (eq4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
sorry

end sum_of_integers_l2013_201311


namespace square_eq_four_implies_x_values_l2013_201368

theorem square_eq_four_implies_x_values (x : ℝ) :
  (x - 1)^2 = 4 → x = 3 ∨ x = -1 := by
  sorry

end square_eq_four_implies_x_values_l2013_201368


namespace cubic_equation_root_l2013_201376

/-- Given that 3 + √5 is a root of x³ + cx² + dx + 15 = 0 where c and d are rational,
    prove that d = -18.5 -/
theorem cubic_equation_root (c d : ℚ) 
  (h : (3 + Real.sqrt 5)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 15 = 0) :
  d = -37/2 := by
sorry

end cubic_equation_root_l2013_201376


namespace set_equality_implies_a_plus_minus_one_l2013_201308

theorem set_equality_implies_a_plus_minus_one (a : ℝ) :
  ({0, -1, 2*a} : Set ℝ) = {a-1, -abs a, a+1} →
  (a = 1 ∨ a = -1) :=
by sorry

end set_equality_implies_a_plus_minus_one_l2013_201308


namespace special_arrangements_count_l2013_201392

/-- The number of ways to arrange 3 boys and 3 girls in a row, 
    where one specific boy is not adjacent to the other two boys -/
def special_arrangements : ℕ :=
  let n_boys := 3
  let n_girls := 3
  let arrangements_with_boys_separated := n_girls.factorial * (n_girls + 1).factorial
  let arrangements_with_two_boys_adjacent := 2 * (n_girls + 1).factorial * n_girls.factorial
  arrangements_with_boys_separated + arrangements_with_two_boys_adjacent

theorem special_arrangements_count : special_arrangements = 288 := by
  sorry

end special_arrangements_count_l2013_201392


namespace manager_employee_ratio_l2013_201379

theorem manager_employee_ratio (total_employees : ℕ) (female_managers : ℕ) 
  (h1 : total_employees = 750) (h2 : female_managers = 300) :
  (female_managers : ℚ) / total_employees = 2 / 5 := by
  sorry

end manager_employee_ratio_l2013_201379


namespace complex_equation_solution_l2013_201325

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 + i) * z = 2 * i

-- Theorem statement
theorem complex_equation_solution :
  ∀ z : ℂ, equation z → z = 1 + i :=
by
  sorry

end complex_equation_solution_l2013_201325


namespace successive_numbers_product_l2013_201336

theorem successive_numbers_product (n : ℕ) : 
  n * (n + 1) = 2652 → n = 51 := by
  sorry

end successive_numbers_product_l2013_201336


namespace schedule_arrangements_eq_192_l2013_201369

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 6

/-- Represents the number of subjects to be scheduled -/
def num_subjects : ℕ := 6

/-- Represents the number of morning periods -/
def num_morning_periods : ℕ := 4

/-- Represents the number of afternoon periods -/
def num_afternoon_periods : ℕ := 2

/-- Calculates the number of ways to arrange the schedule given the constraints -/
def schedule_arrangements : ℕ :=
  (num_morning_periods.choose 1) * (num_afternoon_periods.choose 1) * (num_subjects - 2).factorial

theorem schedule_arrangements_eq_192 : schedule_arrangements = 192 := by
  sorry

end schedule_arrangements_eq_192_l2013_201369


namespace salt_concentration_after_dilution_l2013_201315

/-- Calculates the final salt concentration after adding water to a salt solution -/
theorem salt_concentration_after_dilution
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 56)
  (h2 : initial_concentration = 0.1)
  (h3 : water_added = 14) :
  let salt_amount := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := salt_amount / final_volume
  final_concentration = 0.08 := by sorry

end salt_concentration_after_dilution_l2013_201315


namespace box_length_l2013_201383

/-- Given a box with width 16 units and height 13 units, which can contain 3120 unit cubes (1 x 1 x 1), prove that the length of the box is 15 units. -/
theorem box_length (width : ℕ) (height : ℕ) (volume : ℕ) (length : ℕ) : 
  width = 16 → height = 13 → volume = 3120 → volume = length * width * height → length = 15 := by
  sorry

end box_length_l2013_201383


namespace eight_hash_six_l2013_201339

/-- Definition of the # operation -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- First condition: r # 0 = r + 1 -/
axiom hash_zero (r : ℝ) : hash r 0 = r + 1

/-- Second condition: r # s = s # r -/
axiom hash_comm (r s : ℝ) : hash r s = hash s r

/-- Third condition: (r + 1) # s = (r # s) + s + 2 -/
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

/-- The main theorem to prove -/
theorem eight_hash_six : hash 8 6 = 69 :=
  sorry

end eight_hash_six_l2013_201339


namespace smallest_four_digit_solution_l2013_201362

theorem smallest_four_digit_solution (x : ℕ) : x = 1094 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y, y ≥ 1000 ∧ y < 10000 →
    (9 * y ≡ 27 [ZMOD 18] ∧
     3 * y + 5 ≡ 11 [ZMOD 7] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 16]) →
    x ≤ y) ∧
  (9 * x ≡ 27 [ZMOD 18]) ∧
  (3 * x + 5 ≡ 11 [ZMOD 7]) ∧
  (-3 * x + 2 ≡ 2 * x [ZMOD 16]) := by
sorry

end smallest_four_digit_solution_l2013_201362


namespace no_integer_solution_l2013_201335

theorem no_integer_solution : ∀ x y : ℤ, 2 * x^2 - 5 * y^2 ≠ 7 := by sorry

end no_integer_solution_l2013_201335


namespace trigonometric_product_equals_one_l2013_201395

theorem trigonometric_product_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end trigonometric_product_equals_one_l2013_201395


namespace largest_undefined_value_l2013_201312

theorem largest_undefined_value (x : ℝ) :
  let f (x : ℝ) := (x + 2) / (9 * x^2 - 74 * x + 9)
  let roots := { x | 9 * x^2 - 74 * x + 9 = 0 }
  ∃ (max_root : ℝ), max_root ∈ roots ∧ ∀ (y : ℝ), y ∈ roots → y ≤ max_root ∧
  ∀ (z : ℝ), z > max_root → f z ≠ 0 := by
  sorry

end largest_undefined_value_l2013_201312


namespace zeros_equality_l2013_201343

/-- 
  f(n) represents the number of 0's in the binary representation of a positive integer n
-/
def f (n : ℕ+) : ℕ := sorry

/-- 
  Theorem: For all positive integers n, 
  the number of 0's in the binary representation of 8n+7 
  is equal to the number of 0's in the binary representation of 4n+3
-/
theorem zeros_equality (n : ℕ+) : f (8*n+7) = f (4*n+3) := by sorry

end zeros_equality_l2013_201343


namespace two_digit_congruent_to_two_mod_four_count_l2013_201338

theorem two_digit_congruent_to_two_mod_four_count : 
  (Finset.filter (fun n => n ≥ 10 ∧ n ≤ 99 ∧ n % 4 = 2) (Finset.range 100)).card = 23 := by
  sorry

end two_digit_congruent_to_two_mod_four_count_l2013_201338


namespace seventeen_in_sample_l2013_201356

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize : ℕ) (first : ℕ) : List ℕ :=
  let interval := populationSize / sampleSize
  List.range sampleSize |>.map (fun i => first + i * interval)

/-- Theorem: In a systematic sample of size 4 from a population of 56, 
    if 3 is the first sampled number, then 17 will also be in the sample -/
theorem seventeen_in_sample :
  let sample := systematicSample 56 4 3
  17 ∈ sample := by
  sorry

end seventeen_in_sample_l2013_201356


namespace intersection_of_A_and_B_l2013_201320

-- Define the sets A and B
def A : Set ℝ := {x | x - 3 > 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l2013_201320


namespace phone_number_prime_factorization_l2013_201314

theorem phone_number_prime_factorization :
  ∃ (p q r s : ℕ), 
    (Nat.Prime p) ∧ 
    (Nat.Prime q) ∧ 
    (Nat.Prime r) ∧ 
    (Nat.Prime s) ∧
    (q = p + 2) ∧ 
    (r = q + 2) ∧ 
    (s = r + 2) ∧
    (p * q * r * s = 27433619) ∧
    (p + q + r + s = 290) := by
  sorry

end phone_number_prime_factorization_l2013_201314


namespace x_cubed_greater_y_squared_l2013_201348

theorem x_cubed_greater_y_squared (x y : ℝ) 
  (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_greater_y_squared_l2013_201348


namespace otimes_difference_l2013_201380

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b^2

-- State the theorem
theorem otimes_difference : 
  (otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6)) = -23327/288 := by
  sorry

end otimes_difference_l2013_201380


namespace unique_solution_iff_a_eq_one_or_neg_one_l2013_201318

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x ≤ 1 ∧ x - y = -a

-- Define what it means for the system to have a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x y, system x y a

-- Theorem statement
theorem unique_solution_iff_a_eq_one_or_neg_one :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 1 ∨ a = -1) :=
by sorry

end unique_solution_iff_a_eq_one_or_neg_one_l2013_201318


namespace least_integer_divisible_by_three_primes_l2013_201323

theorem least_integer_divisible_by_three_primes : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n % p = 0 ∧ n % q = 0 ∧ n % r = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ m % p = 0 ∧ m % q = 0 ∧ m % r = 0) → 
    m ≥ 30) :=
by
  sorry

end least_integer_divisible_by_three_primes_l2013_201323


namespace square_area_probability_square_area_probability_proof_l2013_201350

/-- The probability of a randomly chosen point P on a line segment AB of length 10 cm
    resulting in a square with side length AP having an area between 25 cm² and 49 cm² -/
theorem square_area_probability : ℝ :=
  let AB : ℝ := 10
  let lower_bound : ℝ := 25
  let upper_bound : ℝ := 49
  1 / 5

/-- Proof of the theorem -/
theorem square_area_probability_proof :
  square_area_probability = 1 / 5 := by
  sorry

end square_area_probability_square_area_probability_proof_l2013_201350


namespace minimize_y_l2013_201389

variable (a b : ℝ)
def y (x : ℝ) := (x - a)^2 + (x - b)^2

theorem minimize_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y a b x ≤ y a b z ∧ x = (a + b) / 2 :=
sorry

end minimize_y_l2013_201389


namespace probability_theorem_l2013_201326

def total_balls : ℕ := 9
def red_balls : ℕ := 2
def black_balls : ℕ := 3
def white_balls : ℕ := 4

def prob_black_then_white : ℚ := 1/6

def prob_red_within_three : ℚ := 7/12

theorem probability_theorem :
  (black_balls / total_balls * white_balls / (total_balls - 1) = prob_black_then_white) ∧
  (red_balls / total_balls + 
   (total_balls - red_balls) / total_balls * red_balls / (total_balls - 1) + 
   (total_balls - red_balls) / total_balls * (total_balls - red_balls - 1) / (total_balls - 1) * red_balls / (total_balls - 2) = prob_red_within_three) :=
by sorry

end probability_theorem_l2013_201326


namespace cosine_translation_symmetry_l2013_201340

theorem cosine_translation_symmetry (x : ℝ) (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 - π / 12
  (∀ t, f (axis + t) = f (axis - t)) := by sorry

end cosine_translation_symmetry_l2013_201340


namespace polygon_sides_l2013_201367

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 4 * 360 - 180) →
  n = 9 := by
  sorry

end polygon_sides_l2013_201367


namespace trader_profit_above_goal_l2013_201366

theorem trader_profit_above_goal 
  (total_profit : ℕ) 
  (goal_amount : ℕ) 
  (donation_amount : ℕ) 
  (h1 : total_profit = 960)
  (h2 : goal_amount = 610)
  (h3 : donation_amount = 310) :
  (total_profit / 2 + donation_amount) - goal_amount = 180 :=
by sorry

end trader_profit_above_goal_l2013_201366


namespace exists_k_no_carry_l2013_201322

/-- 
There exists a positive integer k such that 3993·k is a number 
consisting only of the digit 9.
-/
theorem exists_k_no_carry : ∃ k : ℕ+, 
  ∃ n : ℕ+, (3993 * k.val : ℕ) = (10^n.val - 1) := by sorry

end exists_k_no_carry_l2013_201322


namespace unique_solution_trigonometric_equation_l2013_201378

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧ 
  Real.tan (150 * π / 180 - x * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 120 := by
  sorry

end unique_solution_trigonometric_equation_l2013_201378


namespace power_function_through_point_l2013_201361

/-- A power function that passes through the point (2, √2) -/
def f (x : ℝ) : ℝ := x ^ (1/2)

/-- Theorem: The power function f(x) that passes through (2, √2) satisfies f(8) = 2√2 -/
theorem power_function_through_point (x : ℝ) :
  f 2 = Real.sqrt 2 → f 8 = 2 * Real.sqrt 2 := by
  sorry

#check power_function_through_point

end power_function_through_point_l2013_201361


namespace solution_range_l2013_201377

theorem solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2/x + 1/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2/x + 1/y = 1 ∧ 2*x + y < m^2 - 8*m) ↔ 
  (m < -1 ∨ m > 9) :=
sorry

end solution_range_l2013_201377


namespace investment_comparison_l2013_201355

def initial_AA : ℝ := 200
def initial_BB : ℝ := 150
def initial_CC : ℝ := 100

def year1_AA_change : ℝ := 1.30
def year1_BB_change : ℝ := 0.80
def year1_CC_change : ℝ := 1.10

def year2_AA_change : ℝ := 0.85
def year2_BB_change : ℝ := 1.30
def year2_CC_change : ℝ := 0.95

def final_A : ℝ := initial_AA * year1_AA_change * year2_AA_change
def final_B : ℝ := initial_BB * year1_BB_change * year2_BB_change
def final_C : ℝ := initial_CC * year1_CC_change * year2_CC_change

theorem investment_comparison : final_A > final_B ∧ final_B > final_C := by
  sorry

end investment_comparison_l2013_201355


namespace stating_chess_tournament_players_l2013_201304

/-- The number of players in a chess tournament -/
def num_players : ℕ := 17

/-- The total number of games played in the tournament -/
def total_games : ℕ := 272

/-- 
Theorem stating that the number of players in the chess tournament is correct,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  (2 * num_players * (num_players - 1) = total_games) ∧ 
  (∀ n : ℕ, 2 * n * (n - 1) = total_games → n = num_players) := by
  sorry

#check chess_tournament_players

end stating_chess_tournament_players_l2013_201304


namespace expand_and_equate_l2013_201391

theorem expand_and_equate : 
  (∀ x : ℝ, (x - 5) * (x + 2) = x^2 + p * x + q) → p = -3 ∧ q = -10 := by
sorry

end expand_and_equate_l2013_201391


namespace coefficient_x_squared_in_expansion_l2013_201385

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (∃ c : ℝ, (x - 2/x)^4 = c*x^2 + (terms_without_x_squared : ℝ)) → 
  (∃ c : ℝ, (x - 2/x)^4 = 8*x^2 + (terms_without_x_squared : ℝ)) :=
by sorry

end coefficient_x_squared_in_expansion_l2013_201385


namespace wire_length_ratio_l2013_201346

/-- Represents the length of a single piece of wire used by Bonnie -/
def bonnie_wire_length : ℝ := 4

/-- Represents the number of wire pieces used by Bonnie to construct her cube -/
def bonnie_wire_count : ℕ := 12

/-- Represents the length of a single piece of wire used by Roark -/
def roark_wire_length : ℝ := 1

/-- Represents the side length of Bonnie's cube -/
def bonnie_cube_side : ℝ := bonnie_wire_length

/-- Represents the volume of a single unit cube constructed by Roark -/
def unit_cube_volume : ℝ := 1

/-- Theorem stating that the ratio of Bonnie's total wire length to Roark's total wire length is 1/16 -/
theorem wire_length_ratio :
  (bonnie_wire_length * bonnie_wire_count) / 
  (roark_wire_length * (12 * (bonnie_cube_side ^ 3))) = 1 / 16 := by
  sorry

end wire_length_ratio_l2013_201346


namespace central_symmetry_intersection_condition_l2013_201381

/-- Two functions are centrally symmetric and intersect at one point -/
def centrally_symmetric_one_intersection (a b c d : ℝ) : Prop :=
  let f := fun x => 2 * a + 1 / (x - b)
  let g := fun x => 2 * c + 1 / (x - d)
  let center := ((b + d) / 2, a + c)
  ∃! x, f x = g x ∧ 
    ∀ y, f ((b + d) - y) = g y ∧ 
         g ((b + d) - y) = f y

/-- The main theorem -/
theorem central_symmetry_intersection_condition (a b c d : ℝ) :
  centrally_symmetric_one_intersection a b c d ↔ (a - c) * (b - d) = 2 :=
by sorry


end central_symmetry_intersection_condition_l2013_201381


namespace exists_function_satisfying_condition_l2013_201398

theorem exists_function_satisfying_condition :
  ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (n : ℝ)^2 - 1 < (f (f n) : ℝ) ∧ (f (f n) : ℝ) < (n : ℝ)^2 + 2 := by
  sorry

end exists_function_satisfying_condition_l2013_201398


namespace rectangular_plot_width_l2013_201352

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (h1 : length = 90)
  (h2 : num_poles = 70)
  (h3 : pole_distance = 4)
  : ∃ width : ℝ, width = 48 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance :=
by
  sorry

end rectangular_plot_width_l2013_201352


namespace angles_on_axes_l2013_201307

def TerminalSideOnAxes (α : Real) : Prop :=
  ∃ k : ℤ, α = k * (Real.pi / 2)

theorem angles_on_axes :
  {α : Real | TerminalSideOnAxes α} = {α : Real | ∃ k : ℤ, α = k * (Real.pi / 2)} := by
  sorry

end angles_on_axes_l2013_201307


namespace initial_volume_calculation_l2013_201390

theorem initial_volume_calculation (initial_milk_percentage : Real)
                                   (final_milk_percentage : Real)
                                   (added_water : Real) :
  initial_milk_percentage = 0.84 →
  final_milk_percentage = 0.64 →
  added_water = 18.75 →
  ∃ (initial_volume : Real),
    initial_volume * initial_milk_percentage = 
    final_milk_percentage * (initial_volume + added_water) ∧
    initial_volume = 225 := by
  sorry

end initial_volume_calculation_l2013_201390


namespace prob_at_least_one_six_given_different_outcomes_prob_at_least_one_six_is_one_third_l2013_201354

/-- The probability of rolling at least one 6 given two fair dice with different outcomes -/
theorem prob_at_least_one_six_given_different_outcomes : ℝ :=
let total_outcomes := 30  -- 6 * 5, as outcomes are different
let favorable_outcomes := 10  -- 5 (first die is 6) + 5 (second die is 6)
favorable_outcomes / total_outcomes

/-- Proof that the probability is 1/3 -/
theorem prob_at_least_one_six_is_one_third :
  prob_at_least_one_six_given_different_outcomes = 1 / 3 := by
  sorry

end prob_at_least_one_six_given_different_outcomes_prob_at_least_one_six_is_one_third_l2013_201354


namespace largest_x_value_l2013_201359

theorem largest_x_value : ∃ (x_max : ℝ), 
  (∀ x : ℝ, (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 8 * x - 2 → x ≤ x_max) ∧
  ((15 * x_max^2 - 40 * x_max + 18) / (4 * x_max - 3) + 7 * x_max = 8 * x_max - 2) ∧
  x_max = 4 :=
by sorry

end largest_x_value_l2013_201359


namespace f_satisfies_conditions_l2013_201386

def f (x : ℝ) : ℝ := -x + 1

theorem f_satisfies_conditions :
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) :=
sorry

end f_satisfies_conditions_l2013_201386


namespace company_p_employee_count_l2013_201360

theorem company_p_employee_count (jan_employees : ℝ) : 
  jan_employees * 1.10 * 1.15 * 1.20 = 470 →
  ⌊jan_employees⌋ = 310 := by
  sorry

end company_p_employee_count_l2013_201360


namespace bianca_drawing_time_l2013_201327

/-- The total time Bianca spent drawing is equal to 41 minutes, given that she spent 22 minutes drawing at school and 19 minutes drawing at home. -/
theorem bianca_drawing_time (time_at_school time_at_home : ℕ) 
  (h1 : time_at_school = 22)
  (h2 : time_at_home = 19) :
  time_at_school + time_at_home = 41 := by
  sorry

end bianca_drawing_time_l2013_201327


namespace definite_integral_cos_zero_l2013_201329

theorem definite_integral_cos_zero : 
  ∫ x in (π/4)..(9*π/4), Real.sqrt 2 * Real.cos (2*x + π/4) = 0 := by sorry

end definite_integral_cos_zero_l2013_201329


namespace sum_of_a_values_l2013_201337

/-- The equation for which we need to find the values of 'a' -/
def equation (a x : ℝ) : Prop := 4 * x^2 + a * x + 8 * x + 9 = 0

/-- The condition for the equation to have only one solution -/
def has_one_solution (a : ℝ) : Prop :=
  ∃! x, equation a x

/-- The theorem stating that the sum of 'a' values is -16 -/
theorem sum_of_a_values :
  ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ has_one_solution a₁ ∧ has_one_solution a₂ ∧ a₁ + a₂ = -16 :=
sorry

end sum_of_a_values_l2013_201337


namespace jack_head_circumference_l2013_201358

theorem jack_head_circumference :
  ∀ (J C B : ℝ),
  C = J / 2 + 9 →
  B = 2 / 3 * C →
  B = 10 →
  J = 12 :=
by
  sorry

end jack_head_circumference_l2013_201358


namespace area_of_ABCD_l2013_201301

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The problem statement -/
theorem area_of_ABCD (r1 r2 r3 : Rectangle) : 
  r1.area + r2.area + r3.area = 8 ∧ r1.area = 2 → 
  ∃ (ABCD : Rectangle), ABCD.area = 8 := by
  sorry

end area_of_ABCD_l2013_201301


namespace inequality_equivalence_l2013_201305

theorem inequality_equivalence (x : ℝ) :
  (x + 1) / (x - 5) ≥ 3 ↔ x ≥ 8 ∧ x ≠ 5 :=
by sorry

end inequality_equivalence_l2013_201305


namespace tshirt_sale_ratio_l2013_201306

/-- Prove that the ratio of black shirts to white shirts is 1:1 given the conditions -/
theorem tshirt_sale_ratio :
  ∀ (black white : ℕ),
  black + white = 200 →
  30 * black + 25 * white = 5500 →
  black = white :=
by sorry

end tshirt_sale_ratio_l2013_201306


namespace right_triangle_leg_length_l2013_201397

theorem right_triangle_leg_length : ∀ (a b c : ℝ),
  a = 8 →
  c = 17 →
  a^2 + b^2 = c^2 →
  b = 15 :=
by
  sorry

end right_triangle_leg_length_l2013_201397


namespace lynne_spent_75_l2013_201349

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books solar_books magazines book_price magazine_price : ℕ) : ℕ :=
  cat_books * book_price + solar_books * book_price + magazines * magazine_price

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_spent_75 :
  total_spent 7 2 3 7 4 = 75 := by
  sorry

end lynne_spent_75_l2013_201349


namespace sarah_apple_ratio_l2013_201330

theorem sarah_apple_ratio : 
  let sarah_apples : ℝ := 45.0
  let brother_apples : ℝ := 9.0
  sarah_apples / brother_apples = 5 := by
sorry

end sarah_apple_ratio_l2013_201330


namespace jelly_bean_probability_l2013_201347

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.25 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_red ≥ 0 ∧ p_orange ≥ 0 ∧ p_yellow ≥ 0 ∧ p_green ≥ 0 →
  p_yellow = 0.25 := by
  sorry

end jelly_bean_probability_l2013_201347
