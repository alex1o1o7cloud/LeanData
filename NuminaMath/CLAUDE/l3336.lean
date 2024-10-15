import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_centers_collinearity_l3336_333677

-- Define the points
variable (A B C D E U H V K : Euclidean_plane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Euclidean_plane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D E : Euclidean_plane) : Prop := sorry

-- Define the circumcenter
def is_circumcenter (U A B E : Euclidean_plane) : Prop := sorry

-- Define the orthocenter
def is_orthocenter (H A B E : Euclidean_plane) : Prop := sorry

-- Define collinearity
def collinear (P Q R : Euclidean_plane) : Prop := sorry

-- State the theorem
theorem quadrilateral_centers_collinearity 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_intersect_at A B C D E)
  (h3 : is_circumcenter U A B E)
  (h4 : is_orthocenter H A B E)
  (h5 : is_circumcenter V C D E)
  (h6 : is_orthocenter K C D E) :
  collinear U E K ↔ collinear V E H := by sorry

end NUMINAMATH_CALUDE_quadrilateral_centers_collinearity_l3336_333677


namespace NUMINAMATH_CALUDE_equation_solutions_l3336_333616

theorem equation_solutions :
  (∃ x : ℚ, 1 - 1 / (x - 5) = x / (x + 5) ∧ x = 15 / 2) ∧
  (∃ x : ℚ, 3 / (x - 1) - 2 / (x + 1) = 1 / (x^2 - 1) ∧ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3336_333616


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l3336_333654

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 18*a*b) :
  |((a+b)/(a-b))| = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l3336_333654


namespace NUMINAMATH_CALUDE_base_with_final_digit_one_l3336_333608

theorem base_with_final_digit_one : 
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by sorry

end NUMINAMATH_CALUDE_base_with_final_digit_one_l3336_333608


namespace NUMINAMATH_CALUDE_horizontal_grid_lines_length_6_10_l3336_333680

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the total length of horizontal grid lines inside a right-angled triangle -/
def horizontalGridLinesLength (t : GridTriangle) : ℕ :=
  (t.base * (t.height - 1)) / 2

/-- The theorem stating the total length of horizontal grid lines for the specific triangle -/
theorem horizontal_grid_lines_length_6_10 :
  horizontalGridLinesLength { base := 10, height := 6 } = 27 := by
  sorry

#eval horizontalGridLinesLength { base := 10, height := 6 }

end NUMINAMATH_CALUDE_horizontal_grid_lines_length_6_10_l3336_333680


namespace NUMINAMATH_CALUDE_train_length_l3336_333685

/-- Calculates the length of a train given its speed, time to pass a station, and the station's length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (station_length : ℝ) :
  train_speed = 36 * (1000 / 3600) →
  time_to_pass = 45 →
  station_length = 200 →
  train_speed * time_to_pass - station_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3336_333685


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_polynomial_l3336_333653

theorem max_value_x_cubed_over_polynomial (x : ℝ) :
  x^3 / (x^6 + x^4 + x^3 - 3*x^2 + 9) ≤ 1/7 ∧
  ∃ y : ℝ, y^3 / (y^6 + y^4 + y^3 - 3*y^2 + 9) = 1/7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_polynomial_l3336_333653


namespace NUMINAMATH_CALUDE_smallest_room_length_l3336_333606

/-- Given two rectangular rooms, where the larger room has dimensions 45 feet by 30 feet,
    and the smaller room has a width of 15 feet, if the difference in area between
    these two rooms is 1230 square feet, then the length of the smaller room is 8 feet. -/
theorem smallest_room_length
  (larger_width : ℝ) (larger_length : ℝ)
  (smaller_width : ℝ) (smaller_length : ℝ)
  (area_difference : ℝ) :
  larger_width = 45 →
  larger_length = 30 →
  smaller_width = 15 →
  area_difference = 1230 →
  larger_width * larger_length - smaller_width * smaller_length = area_difference →
  smaller_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_room_length_l3336_333606


namespace NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l3336_333652

theorem product_of_squares_and_fourth_powers (r s : ℝ) 
  (h_positive_r : r > 0) (h_positive_s : s > 0)
  (h_sum_squares : r^2 + s^2 = 1) 
  (h_sum_fourth_powers : r^4 + s^4 = 7/8) : r * s = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l3336_333652


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3336_333651

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 7 * p - 6 = 0 →
  3 * q^2 - 7 * q - 6 = 0 →
  p ≠ q →
  (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3336_333651


namespace NUMINAMATH_CALUDE_farm_animals_l3336_333676

theorem farm_animals (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 5 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 17 / 7 →
  (initial_horses - 15) - (initial_cows + 15) = 50 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l3336_333676


namespace NUMINAMATH_CALUDE_hidden_primes_average_l3336_333670

/-- Given three cards with numbers on both sides, this theorem proves that
    the average of the hidden prime numbers is 46/3, given the conditions
    specified in the problem. -/
theorem hidden_primes_average (card1_visible card2_visible card3_visible : ℕ)
  (card1_hidden card2_hidden card3_hidden : ℕ)
  (h1 : card1_visible = 68)
  (h2 : card2_visible = 39)
  (h3 : card3_visible = 57)
  (h4 : Nat.Prime card1_hidden)
  (h5 : Nat.Prime card2_hidden)
  (h6 : Nat.Prime card3_hidden)
  (h7 : card1_visible + card1_hidden = card2_visible + card2_hidden)
  (h8 : card2_visible + card2_hidden = card3_visible + card3_hidden) :
  (card1_hidden + card2_hidden + card3_hidden : ℚ) / 3 = 46 / 3 := by
  sorry

#eval (46 : ℚ) / 3

end NUMINAMATH_CALUDE_hidden_primes_average_l3336_333670


namespace NUMINAMATH_CALUDE_distinct_terms_in_expansion_l3336_333617

/-- The number of distinct terms in the expansion of (a+b+c+d)(e+f+g+h+i),
    given that terms involving the product of a and e, and b and f are identical
    and combine into a single term. -/
theorem distinct_terms_in_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_distinct_terms_in_expansion_l3336_333617


namespace NUMINAMATH_CALUDE_range_of_a_l3336_333655

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + a

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 0 2, f a x ≥ 0
def q (a : ℝ) : Prop := ∃ x > 0, g a x = 0

-- State the theorem
theorem range_of_a :
  {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} =
  Set.Icc (-1) (-1/2) ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3336_333655


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l3336_333645

theorem quadratic_function_bounds (a : ℝ) (m : ℝ) : 
  a ≠ 0 → a < 0 → 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → 
    -2 ≤ a * x^2 + 2 * x + 1 ∧ a * x^2 + 2 * x + 1 ≤ 2) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l3336_333645


namespace NUMINAMATH_CALUDE_winner_received_62_percent_l3336_333657

/-- Represents an election with two candidates -/
structure Election where
  winner_votes : ℕ
  winning_margin : ℕ

/-- Calculates the percentage of votes received by the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / ((e.winner_votes + (e.winner_votes - e.winning_margin)) : ℚ) * 100

/-- Theorem stating that in the given election scenario, the winner received 62% of votes -/
theorem winner_received_62_percent :
  let e : Election := { winner_votes := 775, winning_margin := 300 }
  winner_percentage e = 62 := by sorry

end NUMINAMATH_CALUDE_winner_received_62_percent_l3336_333657


namespace NUMINAMATH_CALUDE_proportion_sum_condition_l3336_333699

theorem proportion_sum_condition 
  (a b c d a₁ b₁ c₁ d₁ : ℚ) 
  (h1 : a / b = c / d) 
  (h2 : a₁ / b₁ = c₁ / d₁) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) 
  (h5 : b₁ ≠ 0) 
  (h6 : d₁ ≠ 0) 
  (h7 : b + b₁ ≠ 0) 
  (h8 : d + d₁ ≠ 0) : 
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ :=
by sorry

end NUMINAMATH_CALUDE_proportion_sum_condition_l3336_333699


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_5_l3336_333658

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Given vectors a and b, if they are parallel, then x = 5 -/
theorem parallel_vectors_imply_x_equals_5 :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  are_parallel a b → x = 5 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_5_l3336_333658


namespace NUMINAMATH_CALUDE_abc_plus_def_equals_zero_l3336_333692

/-- Represents the transformation of numbers in the circle --/
def transform (v : Fin 6 → ℝ) : Fin 6 → ℝ := fun i =>
  v i + v (i - 1) + v (i + 1)

/-- The condition that after 2022 iterations, the numbers return to their initial values --/
def returns_to_initial (v : Fin 6 → ℝ) : Prop :=
  (transform^[2022]) v = v

theorem abc_plus_def_equals_zero 
  (v : Fin 6 → ℝ) 
  (h : returns_to_initial v) : 
  v 0 * v 1 * v 2 + v 3 * v 4 * v 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_plus_def_equals_zero_l3336_333692


namespace NUMINAMATH_CALUDE_bucket_ratio_l3336_333663

theorem bucket_ratio (small_bucket : ℚ) (large_bucket : ℚ) : 
  (∃ (n : ℚ), large_bucket = n * small_bucket + 3) →
  2 * small_bucket + 5 * large_bucket = 63 →
  large_bucket = 4 →
  large_bucket / small_bucket = 4 := by
sorry

end NUMINAMATH_CALUDE_bucket_ratio_l3336_333663


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3336_333667

theorem triangle_third_side_length 
  (a b : ℝ) 
  (angle : ℝ) 
  (ha : a = 9) 
  (hb : b = 10) 
  (hangle : angle = Real.pi * 3 / 4) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * Real.cos angle ∧ 
            c = Real.sqrt (181 + 90 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3336_333667


namespace NUMINAMATH_CALUDE_freds_dimes_l3336_333624

/-- Fred's dime problem -/
theorem freds_dimes (initial_dimes borrowed_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_freds_dimes_l3336_333624


namespace NUMINAMATH_CALUDE_cost_price_determination_l3336_333614

theorem cost_price_determination (loss_percentage : Real) (gain_percentage : Real) (price_increase : Real) :
  loss_percentage = 0.1 →
  gain_percentage = 0.1 →
  price_increase = 50 →
  ∃ (cost_price : Real),
    cost_price * (1 - loss_percentage) + price_increase = cost_price * (1 + gain_percentage) ∧
    cost_price = 250 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_determination_l3336_333614


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3336_333686

theorem fraction_to_decimal (h : 160 = 2^5 * 5) : 7 / 160 = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3336_333686


namespace NUMINAMATH_CALUDE_arrangements_not_adjacent_l3336_333698

theorem arrangements_not_adjacent (n : ℕ) (h : n = 6) : 
  (n.factorial : ℕ) - 2 * ((n - 1).factorial : ℕ) = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_not_adjacent_l3336_333698


namespace NUMINAMATH_CALUDE_probability_in_specific_club_l3336_333662

/-- A club with members of different genders and seniority levels -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  senior_boys : ℕ
  junior_boys : ℕ
  senior_girls : ℕ
  junior_girls : ℕ

/-- The probability of selecting two girls, one senior and one junior, from the club -/
def probability_two_girls_diff_seniority (c : Club) : ℚ :=
  (c.senior_girls.choose 1 * c.junior_girls.choose 1 : ℚ) / c.total_members.choose 2

/-- Theorem stating the probability for the given club configuration -/
theorem probability_in_specific_club : 
  ∃ c : Club, 
    c.total_members = 12 ∧ 
    c.boys = 6 ∧ 
    c.girls = 6 ∧ 
    c.senior_boys = 3 ∧ 
    c.junior_boys = 3 ∧ 
    c.senior_girls = 3 ∧ 
    c.junior_girls = 3 ∧ 
    probability_two_girls_diff_seniority c = 9 / 66 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_specific_club_l3336_333662


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l3336_333697

theorem complex_modulus_equation :
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (9 + t * Complex.I) = 15 ∧ t = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l3336_333697


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3336_333644

theorem solution_set_of_inequality (a : ℝ) (h : a > 1) :
  {x : ℝ | |x| + a > 1} = Set.univ :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3336_333644


namespace NUMINAMATH_CALUDE_two_digit_number_ratio_l3336_333639

theorem two_digit_number_ratio (a b : ℕ) : 
  a ≤ 9 ∧ b ≤ 9 ∧ a ≠ 0 → -- Ensure a and b are single digits and a is not 0
  (10 * a + b) * 6 = (10 * b + a) * 5 → -- Ratio condition
  10 * a + b = 45 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_ratio_l3336_333639


namespace NUMINAMATH_CALUDE_large_pizzas_purchased_l3336_333619

/-- Represents the number of slices in a small pizza -/
def small_pizza_slices : ℕ := 4

/-- Represents the number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- Represents the number of small pizzas purchased -/
def small_pizzas_purchased : ℕ := 3

/-- Represents the total number of slices consumed by all people -/
def total_slices_consumed : ℕ := 18

/-- Represents the number of slices left over -/
def slices_left_over : ℕ := 10

theorem large_pizzas_purchased :
  ∃ (n : ℕ), n * large_pizza_slices + small_pizzas_purchased * small_pizza_slices =
    total_slices_consumed + slices_left_over ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_large_pizzas_purchased_l3336_333619


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3336_333625

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def SimilarTriangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_side_length 
  (P Q R S T U : ℝ × ℝ) 
  (h_similar : SimilarTriangles {P, Q, R} {S, T, U}) 
  (h_PQ : dist P Q = 10) 
  (h_QR : dist Q R = 15) 
  (h_ST : dist S T = 6) : 
  dist T U = 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3336_333625


namespace NUMINAMATH_CALUDE_methane_moles_required_l3336_333634

-- Define the chemical species involved
structure ChemicalSpecies where
  methane : ℝ
  chlorine : ℝ
  chloromethane : ℝ
  hydrochloric_acid : ℝ

-- Define the reaction conditions
def reaction_conditions (reactants products : ChemicalSpecies) : Prop :=
  reactants.chlorine = 2 ∧
  products.chloromethane = 2 ∧
  products.hydrochloric_acid = 2

-- Define the stoichiometric relationship
def stoichiometric_relationship (reactants products : ChemicalSpecies) : Prop :=
  reactants.methane = products.chloromethane ∧
  reactants.methane = products.hydrochloric_acid

-- Theorem statement
theorem methane_moles_required 
  (reactants products : ChemicalSpecies) 
  (h_conditions : reaction_conditions reactants products) 
  (h_stoichiometry : stoichiometric_relationship reactants products) : 
  reactants.methane = 2 := by
  sorry

end NUMINAMATH_CALUDE_methane_moles_required_l3336_333634


namespace NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l3336_333603

theorem gcf_of_7_factorial_and_8_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l3336_333603


namespace NUMINAMATH_CALUDE_trees_in_yard_l3336_333604

/-- Given a yard of length 275 meters with trees planted at equal distances,
    one tree at each end, and 11 meters between consecutive trees,
    prove that there are 26 trees in total. -/
theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) : 
  yard_length = 275 → 
  tree_distance = 11 → 
  (yard_length - tree_distance) % tree_distance = 0 →
  (yard_length - tree_distance) / tree_distance + 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l3336_333604


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_successor_l3336_333659

def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_digit_sum_of_successor (n : ℕ) (h : digit_sum n = 2017) :
  ∃ (m : ℕ), digit_sum (n + 1) = m ∧ ∀ (k : ℕ), digit_sum (n + 1) ≤ k := by sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_successor_l3336_333659


namespace NUMINAMATH_CALUDE_equilateral_not_obtuse_l3336_333632

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define properties of a triangle
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angleA > 90 ∨ t.angleB > 90 ∨ t.angleC > 90

-- Theorem: An equilateral triangle cannot be obtuse
theorem equilateral_not_obtuse (t : Triangle) :
  t.isEquilateral → ¬t.isObtuse := by
  sorry

end NUMINAMATH_CALUDE_equilateral_not_obtuse_l3336_333632


namespace NUMINAMATH_CALUDE_f_triple_eq_f_solutions_bound_l3336_333641

noncomputable def f (x : ℝ) : ℝ := -3 * Real.sin (Real.pi * x)

theorem f_triple_eq_f_solutions_bound :
  ∃ (S : Finset ℝ), (∀ x ∈ S, -1 ≤ x ∧ x ≤ 1) ∧ 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (f (f x)) = f x → x ∈ S) ∧
  Finset.card S ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_f_triple_eq_f_solutions_bound_l3336_333641


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l3336_333687

theorem min_value_sum_of_fractions (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 ∧
  (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) = 3 * Real.sqrt 3 / 2 ↔ 
   x = Real.sqrt 3 / 3 ∧ y = Real.sqrt 3 / 3 ∧ z = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l3336_333687


namespace NUMINAMATH_CALUDE_fourth_guard_theorem_l3336_333631

/-- Represents a rectangular facility with guards at each corner -/
structure Facility :=
  (length : ℝ)
  (width : ℝ)
  (guard_distance : ℝ)

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (f : Facility) : ℝ :=
  2 * (f.length + f.width) - f.guard_distance

/-- Theorem stating the distance run by the fourth guard -/
theorem fourth_guard_theorem (f : Facility) 
  (h1 : f.length = 200)
  (h2 : f.width = 300)
  (h3 : f.guard_distance = 850) :
  fourth_guard_distance f = 150 := by
  sorry

#eval fourth_guard_distance { length := 200, width := 300, guard_distance := 850 }

end NUMINAMATH_CALUDE_fourth_guard_theorem_l3336_333631


namespace NUMINAMATH_CALUDE_unique_solution_x_squared_minus_two_factorial_y_l3336_333642

theorem unique_solution_x_squared_minus_two_factorial_y : 
  ∃! (x y : ℕ+), x^2 - 2 * Nat.factorial y.val = 2021 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_squared_minus_two_factorial_y_l3336_333642


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3336_333672

theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3336_333672


namespace NUMINAMATH_CALUDE_ratio_problem_l3336_333623

/-- Given two positive integers A and B, where A < B, if A = 36 and LCM(A, B) = 180, then A:B = 1:5 -/
theorem ratio_problem (A B : ℕ) (h1 : 0 < A) (h2 : A < B) (h3 : A = 36) (h4 : Nat.lcm A B = 180) :
  A * 5 = B * 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3336_333623


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3336_333683

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3336_333683


namespace NUMINAMATH_CALUDE_fraction_simplification_l3336_333610

theorem fraction_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3336_333610


namespace NUMINAMATH_CALUDE_gift_bags_production_time_l3336_333605

theorem gift_bags_production_time (total_bags : ℕ) (rate_per_day : ℕ) (h1 : total_bags = 519) (h2 : rate_per_day = 42) :
  (total_bags + rate_per_day - 1) / rate_per_day = 13 :=
sorry

end NUMINAMATH_CALUDE_gift_bags_production_time_l3336_333605


namespace NUMINAMATH_CALUDE_cubic_coefficient_b_is_zero_l3336_333633

-- Define the cubic function
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_coefficient_b_is_zero
  (a b c d : ℝ) :
  (g a b c d (-2) = 0) →
  (g a b c d 0 = 0) →
  (g a b c d 2 = 0) →
  (g a b c d 1 = -1) →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_coefficient_b_is_zero_l3336_333633


namespace NUMINAMATH_CALUDE_canteen_distance_l3336_333656

theorem canteen_distance (road_distance : ℝ) (perpendicular_distance : ℝ) 
  (hypotenuse_distance : ℝ) (canteen_distance : ℝ) :
  road_distance = 400 ∧ 
  perpendicular_distance = 300 ∧ 
  hypotenuse_distance = 500 ∧
  canteen_distance^2 = perpendicular_distance^2 + (road_distance - canteen_distance)^2 →
  canteen_distance = 312.5 := by
sorry

end NUMINAMATH_CALUDE_canteen_distance_l3336_333656


namespace NUMINAMATH_CALUDE_coupon_collection_probability_l3336_333650

theorem coupon_collection_probability (n m k : ℕ) (hn : n = 17) (hm : m = 9) (hk : k = 6) :
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_collection_probability_l3336_333650


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l3336_333691

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 10

-- Define the line y = -x + m
def line (x y m : ℝ) : Prop := y = -x + m

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 1)
def point_B : ℝ × ℝ := (1, 3)

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ line x₁ y₁ m ∧
    circle_C x₂ y₂ ∧ line x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the condition for MN to pass through the origin
def passes_through_origin (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ line x₁ y₁ m ∧
    circle_C x₂ y₂ ∧ line x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ + x₂)^2 + (y₁ + y₂)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem circle_intersection_theorem :
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 ∧
  (∀ m : ℝ, intersection_points m) →
  passes_through_origin (1 + Real.sqrt 7) ∧
  passes_through_origin (1 - Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l3336_333691


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3336_333648

theorem book_pages_calculation (chapters : ℕ) (days : ℕ) (pages_per_day : ℕ) 
  (h1 : chapters = 41)
  (h2 : days = 30)
  (h3 : pages_per_day = 15) :
  chapters * (days * pages_per_day / chapters) = days * pages_per_day :=
by
  sorry

#check book_pages_calculation

end NUMINAMATH_CALUDE_book_pages_calculation_l3336_333648


namespace NUMINAMATH_CALUDE_gina_sister_choice_ratio_l3336_333635

/-- The ratio of Gina's choices to her sister's choices on Netflix --/
theorem gina_sister_choice_ratio :
  ∀ (sister_shows : ℕ) (show_length : ℕ) (gina_minutes : ℕ),
  sister_shows = 24 →
  show_length = 50 →
  gina_minutes = 900 →
  (gina_minutes : ℚ) / (sister_shows * show_length : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_gina_sister_choice_ratio_l3336_333635


namespace NUMINAMATH_CALUDE_remainder_problem_l3336_333689

theorem remainder_problem (n : ℕ) (a b c d : ℕ) : 
  n > 0 → 
  n = 102 * a + b → 
  n = 103 * c + d → 
  0 ≤ b → b < 102 → 
  0 ≤ d → d < 103 → 
  a + d = 20 → 
  b = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3336_333689


namespace NUMINAMATH_CALUDE_lucy_lovely_age_difference_l3336_333678

/-- Represents the ages of Lucy and Lovely at different points in time -/
structure Ages where
  lucy_current : ℕ
  lovely_current : ℕ
  years_ago : ℕ

/-- Conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  a.lucy_current = 50 ∧
  a.lucy_current - a.years_ago = 3 * (a.lovely_current - a.years_ago) ∧
  a.lucy_current + 10 = 2 * (a.lovely_current + 10)

/-- Theorem stating the solution to the problem -/
theorem lucy_lovely_age_difference :
  ∃ (a : Ages), problem_conditions a ∧ a.years_ago = 5 :=
sorry

end NUMINAMATH_CALUDE_lucy_lovely_age_difference_l3336_333678


namespace NUMINAMATH_CALUDE_logical_equivalences_l3336_333618

theorem logical_equivalences (A B C : Prop) : 
  ((A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C)) ∧ 
   (A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C))) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalences_l3336_333618


namespace NUMINAMATH_CALUDE_no_solution_when_m_negative_four_point_five_l3336_333643

/-- The vector equation has no solutions when m = -4.5 -/
theorem no_solution_when_m_negative_four_point_five :
  let m : ℝ := -4.5
  let v1 : ℝ × ℝ := (1, 3)
  let v2 : ℝ × ℝ := (2, -3)
  let v3 : ℝ × ℝ := (-1, 4)
  let v4 : ℝ × ℝ := (3, m)
  ¬∃ (t s : ℝ), v1 + t • v2 = v3 + s • v4 :=
by sorry


end NUMINAMATH_CALUDE_no_solution_when_m_negative_four_point_five_l3336_333643


namespace NUMINAMATH_CALUDE_function_inequality_l3336_333661

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 1

-- State the theorem
theorem function_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 1| < b → |f x + 4| < a) ↔ b ≤ a / 4 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l3336_333661


namespace NUMINAMATH_CALUDE_boat_current_speed_l3336_333690

/-- Proves that given a boat with a speed of 20 km/hr in still water, 
    traveling 9.2 km downstream in 24 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : downstream_distance = 9.2)
  (h3 : time_minutes = 24) :
  let time_hours : ℝ := time_minutes / 60
  let current_speed : ℝ := downstream_distance / time_hours - boat_speed
  current_speed = 3 := by sorry

end NUMINAMATH_CALUDE_boat_current_speed_l3336_333690


namespace NUMINAMATH_CALUDE_lizzys_money_l3336_333674

theorem lizzys_money (mother_gave : ℕ) (spent_on_candy : ℕ) (uncle_gave : ℕ) (final_amount : ℕ) :
  mother_gave = 80 →
  spent_on_candy = 50 →
  uncle_gave = 70 →
  final_amount = 140 →
  ∃ (father_gave : ℕ), father_gave = 40 ∧ mother_gave + father_gave - spent_on_candy + uncle_gave = final_amount :=
by sorry

end NUMINAMATH_CALUDE_lizzys_money_l3336_333674


namespace NUMINAMATH_CALUDE_mode_of_scores_l3336_333666

def scores : List ℕ := [35, 37, 39, 37, 38, 38, 37]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_scores :
  mode scores = 37 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_scores_l3336_333666


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3336_333664

theorem quadratic_equation_solution (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x : ℝ, x^2 + 14*x = 96 ∧ x > 0 ∧ x = Real.sqrt a - b) → a + b = 152 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3336_333664


namespace NUMINAMATH_CALUDE_smallest_with_8_odd_10_even_divisors_l3336_333601

/-- A function that returns the number of positive odd integer divisors of a natural number -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the number of positive even integer divisors of a natural number -/
def num_even_divisors (n : ℕ) : ℕ := sorry

/-- The theorem stating that 53760 is the smallest positive integer with 8 odd divisors and 10 even divisors -/
theorem smallest_with_8_odd_10_even_divisors :
  ∀ n : ℕ, n > 0 →
    (num_odd_divisors n = 8 ∧ num_even_divisors n = 10) →
    n ≥ 53760 ∧
    (num_odd_divisors 53760 = 8 ∧ num_even_divisors 53760 = 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_8_odd_10_even_divisors_l3336_333601


namespace NUMINAMATH_CALUDE_min_fraction_sum_l3336_333660

theorem min_fraction_sum (A B C D : ℕ) : 
  A ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  B ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  C ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  D ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (A : ℚ) / B + (C : ℚ) / D ≥ 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l3336_333660


namespace NUMINAMATH_CALUDE_tyrones_money_l3336_333600

def one_dollar_bills : ℕ := 2
def five_dollar_bills : ℕ := 1
def quarters : ℕ := 13
def dimes : ℕ := 20
def nickels : ℕ := 8
def pennies : ℕ := 35

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01

def total_money : ℚ := 
  one_dollar_bills + 
  5 * five_dollar_bills + 
  quarter_value * quarters + 
  dime_value * dimes + 
  nickel_value * nickels + 
  penny_value * pennies

theorem tyrones_money : total_money = 13 := by
  sorry

end NUMINAMATH_CALUDE_tyrones_money_l3336_333600


namespace NUMINAMATH_CALUDE_ab_inequality_and_minimum_l3336_333684

theorem ab_inequality_and_minimum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 8) : 
  (a * b ≥ 16) ∧ 
  (a + 4 * b ≥ 17) ∧ 
  (a + 4 * b = 17 → a = 7) := by
sorry

end NUMINAMATH_CALUDE_ab_inequality_and_minimum_l3336_333684


namespace NUMINAMATH_CALUDE_rectangle_width_is_six_l3336_333602

/-- A rectangle with given properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  diagonal_squares : ℕ

/-- The properties of our specific rectangle -/
def my_rectangle : Rectangle where
  length := 8
  width := 6
  area := 48
  diagonal_squares := 12

/-- Theorem stating that the width of the rectangle is 6 inches -/
theorem rectangle_width_is_six (r : Rectangle) 
  (h1 : r.length = 8)
  (h2 : r.area = 48)
  (h3 : r.diagonal_squares = 12) : 
  r.width = 6 := by
  sorry

#check rectangle_width_is_six

end NUMINAMATH_CALUDE_rectangle_width_is_six_l3336_333602


namespace NUMINAMATH_CALUDE_video_upvotes_l3336_333622

theorem video_upvotes (up_to_down_ratio : Rat) (down_votes : ℕ) (up_votes : ℕ) : 
  up_to_down_ratio = 9 / 2 → down_votes = 4 → up_votes = 18 := by
  sorry

end NUMINAMATH_CALUDE_video_upvotes_l3336_333622


namespace NUMINAMATH_CALUDE_solve_equation_l3336_333681

theorem solve_equation (p q : ℝ) (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3336_333681


namespace NUMINAMATH_CALUDE_gdp_growth_problem_l3336_333628

/-- The GDP growth over a period of years -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ years

/-- The GDP growth problem -/
theorem gdp_growth_problem :
  let initial_gdp := 9593.3
  let growth_rate := 0.073
  let years := 4
  ∃ ε > 0, |gdp_growth initial_gdp growth_rate years - 127165| < ε :=
by sorry

end NUMINAMATH_CALUDE_gdp_growth_problem_l3336_333628


namespace NUMINAMATH_CALUDE_final_salary_proof_l3336_333665

def original_salary : ℝ := 20000
def reduction_rate : ℝ := 0.1
def increase_rate : ℝ := 0.1

def salary_after_changes (s : ℝ) (r : ℝ) (i : ℝ) : ℝ :=
  s * (1 - r) * (1 + i)

theorem final_salary_proof :
  salary_after_changes original_salary reduction_rate increase_rate = 19800 := by
  sorry

end NUMINAMATH_CALUDE_final_salary_proof_l3336_333665


namespace NUMINAMATH_CALUDE_number_line_inequalities_l3336_333673

theorem number_line_inequalities (a b c d : ℝ) 
  (ha_neg : a < 0) (hb_neg : b < 0) (hc_pos : c > 0) (hd_pos : d > 0)
  (hc_bounds : 0 < |c| ∧ |c| < 1)
  (hb_bounds : 1 < |b| ∧ |b| < 2)
  (ha_bounds : 2 < |a| ∧ |a| < 4)
  (hd_bounds : 1 < |d| ∧ |d| < 2) : 
  (|a| < 4) ∧ 
  (|b| < 2) ∧ 
  (|c| < 2) ∧ 
  (|a| > |b|) ∧ 
  (|c| < |d|) ∧ 
  (|a - b| < 4) ∧ 
  (|b - c| < 2) ∧ 
  (|c - a| > 1) := by
sorry

end NUMINAMATH_CALUDE_number_line_inequalities_l3336_333673


namespace NUMINAMATH_CALUDE_bicyclist_effective_speed_l3336_333621

/-- Calculates the effective speed of a bicyclist considering headwind -/
def effective_speed (initial_speed_ms : ℝ) (headwind_kmh : ℝ) : ℝ :=
  initial_speed_ms * 3.6 - headwind_kmh

/-- Proves that the effective speed of a bicyclist with an initial speed of 18 m/s
    and a headwind of 10 km/h is 54.8 km/h -/
theorem bicyclist_effective_speed :
  effective_speed 18 10 = 54.8 := by sorry

end NUMINAMATH_CALUDE_bicyclist_effective_speed_l3336_333621


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_main_theorem_l3336_333647

/-- Proves that the number of cards Nell gave to Jeff is 276 --/
theorem cards_given_to_jeff : ℕ → ℕ → ℕ → Prop :=
  fun nell_initial nell_remaining cards_given =>
    nell_initial = 528 →
    nell_remaining = 252 →
    cards_given = nell_initial - nell_remaining →
    cards_given = 276

/-- The main theorem --/
theorem main_theorem : ∃ (cards_given : ℕ), cards_given_to_jeff 528 252 cards_given :=
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_main_theorem_l3336_333647


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l3336_333611

/-- Represents a triangle with sides divided into segments -/
structure DividedTriangle where
  sides : ℕ  -- number of segments each side is divided into

/-- Counts the number of distinct triangles formed from division points -/
def count_distinct_triangles (t : DividedTriangle) : ℕ :=
  let total_points := (t.sides - 1) * 3
  let total_triangles := (total_points.choose 3)
  let parallel_sided := 3 * (t.sides - 1)^2
  let double_parallel := 3 * (t.sides - 1)
  let triple_parallel := 1
  total_triangles - parallel_sided + double_parallel - triple_parallel

/-- The main theorem stating the number of distinct triangles -/
theorem distinct_triangles_count (t : DividedTriangle) (h : t.sides = 8) :
  count_distinct_triangles t = 216 := by
  sorry

#eval count_distinct_triangles ⟨8⟩

end NUMINAMATH_CALUDE_distinct_triangles_count_l3336_333611


namespace NUMINAMATH_CALUDE_car_journey_time_l3336_333671

theorem car_journey_time (distance : ℝ) (new_speed : ℝ) (initial_time : ℝ) : 
  distance = 360 →
  new_speed = 40 →
  distance / new_speed = (3/2) * initial_time →
  initial_time = 6 := by
sorry

end NUMINAMATH_CALUDE_car_journey_time_l3336_333671


namespace NUMINAMATH_CALUDE_production_value_range_l3336_333688

-- Define the production value function
def f (x : ℝ) : ℝ := x * (220 - 2 * x)

-- Define the theorem
theorem production_value_range :
  ∀ x : ℝ, f x ≥ 6000 ↔ 50 < x ∧ x < 60 :=
by sorry

end NUMINAMATH_CALUDE_production_value_range_l3336_333688


namespace NUMINAMATH_CALUDE_expression_value_l3336_333638

theorem expression_value : 
  let a := 2021
  (a^3 - 3*a^2*(a+1) + 4*a*(a+1)^2 - (a+1)^3 + 2) / (a*(a+1)) = 1 + 1/a := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3336_333638


namespace NUMINAMATH_CALUDE_f_monotone_and_F_lower_bound_l3336_333630

noncomputable section

variables (m : ℝ) (x x₀ : ℝ)

def f (x : ℝ) : ℝ := x * Real.exp x - m * x

def F (x : ℝ) : ℝ := f m x - m * Real.log x

theorem f_monotone_and_F_lower_bound (hm : m < -Real.exp (-2)) 
  (h_crit : deriv (F m) x₀ = 0) (h_pos : F m x₀ > 0) :
  (∀ x y, x < y → f m x < f m y) ∧ F m x₀ > -2 * x₀^3 + 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_and_F_lower_bound_l3336_333630


namespace NUMINAMATH_CALUDE_lcm_of_48_and_180_l3336_333696

theorem lcm_of_48_and_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_48_and_180_l3336_333696


namespace NUMINAMATH_CALUDE_minute_hand_angle_for_2h40m_l3336_333675

/-- Represents the angle turned by the minute hand of a clock -/
def minuteHandAngle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- 
Theorem: When the hour hand of a clock moves for 2 hours and 40 minutes 
in a clockwise direction, the minute hand turns through an angle of -960°
-/
theorem minute_hand_angle_for_2h40m : 
  minuteHandAngle 2 40 = -960 := by
  sorry

end NUMINAMATH_CALUDE_minute_hand_angle_for_2h40m_l3336_333675


namespace NUMINAMATH_CALUDE_smallest_x_with_natural_percentages_l3336_333612

theorem smallest_x_with_natural_percentages :
  ∀ x : ℝ, x > 0 →
    (∃ n : ℕ, (45 / 100) * x = n) →
    (∃ m : ℕ, (24 / 100) * x = m) →
    x ≥ 100 / 3 ∧
    (∃ a b : ℕ, (45 / 100) * (100 / 3) = a ∧ (24 / 100) * (100 / 3) = b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_natural_percentages_l3336_333612


namespace NUMINAMATH_CALUDE_problem_solution_l3336_333620

open Real

noncomputable def f (x : ℝ) : ℝ :=
  ((1 + cos (2*x))^2 - 2*cos (2*x) - 1) / (sin (π/4 + x) * sin (π/4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  (1/2) * f x + sin (2*x)

theorem problem_solution :
  (f (-11*π/12) = Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (π/4), g x ≤ Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc 0 (π/4), g x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 (π/4), g x = Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π/4), g x = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3336_333620


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3336_333668

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℚ) :
  is_geometric_sequence a →
  a 2 * a 5 = -3/4 →
  a 2 + a 3 + a 4 + a 5 = 5/4 →
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3336_333668


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3336_333613

theorem roots_sum_powers (c d : ℝ) : 
  c^2 - 5*c + 6 = 0 → d^2 - 5*d + 6 = 0 → c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3336_333613


namespace NUMINAMATH_CALUDE_vector_parallelism_l3336_333640

theorem vector_parallelism (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, x]
  (∃ (k : ℝ), (a + b) = k • (a - b)) → x = -4 := by
sorry

end NUMINAMATH_CALUDE_vector_parallelism_l3336_333640


namespace NUMINAMATH_CALUDE_line_vector_proof_l3336_333626

def line_vector (t : ℚ) : ℚ × ℚ × ℚ := sorry

theorem line_vector_proof :
  (line_vector (-2) = (2, 6, 16)) ∧
  (line_vector 1 = (0, -1, -2)) ∧
  (line_vector 4 = (-2, -8, -18)) →
  (line_vector 0 = (2/3, 4/3, 4)) ∧
  (line_vector 5 = (-8, -19, -26)) := by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l3336_333626


namespace NUMINAMATH_CALUDE_divisibility_by_1897_l3336_333636

theorem divisibility_by_1897 (n : ℕ) : 
  (1897 : ℤ) ∣ (2903^n - 803^n - 464^n + 261^n) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_1897_l3336_333636


namespace NUMINAMATH_CALUDE_decryption_theorem_l3336_333693

/-- Represents a character in the Russian alphabet --/
inductive RussianChar : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | AA | AB | AC | AD | AE | AF | AG

/-- Represents an encrypted message --/
def EncryptedMessage := List Char

/-- Represents a decrypted message --/
def DecryptedMessage := List RussianChar

/-- Converts a base-7 number to base-10 --/
def baseSevenToTen (n : Int) : Int :=
  sorry

/-- Applies Caesar cipher shift to a character --/
def applyCaesarShift (c : Char) (shift : Int) : RussianChar :=
  sorry

/-- Decrypts a message using Caesar cipher and base-7 to base-10 conversion --/
def decryptMessage (msg : EncryptedMessage) (shift : Int) : DecryptedMessage :=
  sorry

/-- Checks if a decrypted message is valid Russian text --/
def isValidRussianText (msg : DecryptedMessage) : Prop :=
  sorry

/-- The main theorem: decrypting the messages with shift 22 results in valid Russian text --/
theorem decryption_theorem (messages : List EncryptedMessage) :
  ∀ msg ∈ messages, isValidRussianText (decryptMessage msg 22) :=
  sorry

end NUMINAMATH_CALUDE_decryption_theorem_l3336_333693


namespace NUMINAMATH_CALUDE_inverse_function_constraint_l3336_333629

theorem inverse_function_constraint (a b c d h : ℝ) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →
  (∀ x, x ∈ Set.range (fun x => (a * (x + h) + b) / (c * (x + h) + d)) →
    (a * ((a * (x + h) + b) / (c * (x + h) + d) + h) + b) / 
    (c * ((a * (x + h) + b) / (c * (x + h) + d) + h) + d) = x) →
  a + d - 2 * c * h = 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_constraint_l3336_333629


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l3336_333682

theorem proposition_false_iff_a_in_range (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l3336_333682


namespace NUMINAMATH_CALUDE_log_equation_solution_l3336_333609

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 + Real.log x / Real.log 27 = 7 →
  x = 3 ^ (42 / 11) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3336_333609


namespace NUMINAMATH_CALUDE_total_age_calculation_l3336_333637

def family_gathering (K : ℕ) : Prop :=
  let father_age : ℕ := 60
  let mother_age : ℕ := father_age - 2
  let brother_age : ℕ := father_age / 2
  let sister_age : ℕ := 40
  let elder_cousin_age : ℕ := brother_age + 2 * sister_age
  let younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
  let grandmother_age : ℕ := 3 * mother_age - 5
  let T : ℕ := father_age + mother_age + brother_age + sister_age + 
               elder_cousin_age + younger_cousin_age + grandmother_age + K
  T = 525 + K

theorem total_age_calculation (K : ℕ) : family_gathering K :=
  sorry

end NUMINAMATH_CALUDE_total_age_calculation_l3336_333637


namespace NUMINAMATH_CALUDE_polynomial_identities_l3336_333646

theorem polynomial_identities (x y : ℝ) : 
  ((x + y)^3 - x^3 - y^3 = 3*x*y*(x + y)) ∧ 
  ((x + y)^5 - x^5 - y^5 = 5*x*y*(x + y)*(x^2 + x*y + y^2)) ∧ 
  ((x + y)^7 - x^7 - y^7 = 7*x*y*(x + y)*(x^2 + x*y + y^2)^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identities_l3336_333646


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l3336_333627

/-- Two lines in the x-y plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- The condition for two lines to be distinct (not coincident) -/
def distinct (l1 l2 : Line) : Prop := l1.intercept ≠ l2.intercept

theorem parallel_lines_a_equals_3 (a : ℝ) :
  let l1 : Line := { slope := a^2, intercept := 3*a - a^2 }
  let l2 : Line := { slope := 4*a - 3, intercept := 2 }
  parallel l1 l2 ∧ distinct l1 l2 → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l3336_333627


namespace NUMINAMATH_CALUDE_ball_probability_l3336_333694

theorem ball_probability (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 3 →
  (red_balls : ℚ) / (red_balls + white_balls : ℚ) = 3 / 7 →
  white_balls = 4 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l3336_333694


namespace NUMINAMATH_CALUDE_x_squared_y_squared_range_l3336_333679

theorem x_squared_y_squared_range (x y : ℝ) (h : x^2 + y^2 = 2*x) :
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27/16 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_y_squared_range_l3336_333679


namespace NUMINAMATH_CALUDE_tan_315_eq_neg_one_l3336_333695

/-- Prove that the tangent of 315 degrees is equal to -1 -/
theorem tan_315_eq_neg_one : Real.tan (315 * π / 180) = -1 := by
  sorry


end NUMINAMATH_CALUDE_tan_315_eq_neg_one_l3336_333695


namespace NUMINAMATH_CALUDE_garage_sale_items_count_l3336_333607

theorem garage_sale_items_count 
  (prices : Finset ℕ) 
  (radio_price : ℕ) 
  (h_distinct : prices.card = prices.toList.length)
  (h_ninth_highest : (prices.filter (· > radio_price)).card = 8)
  (h_thirty_fifth_lowest : (prices.filter (· < radio_price)).card = 34)
  (h_radio_in_prices : radio_price ∈ prices) :
  prices.card = 43 := by
sorry

end NUMINAMATH_CALUDE_garage_sale_items_count_l3336_333607


namespace NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l3336_333649

/-- The number of kids from Lawrence county who went to camp -/
def kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) : ℕ :=
  total_kids - kids_at_home

/-- Theorem: Given the total number of kids in Lawrence county and the number of kids who stayed home,
    prove that the number of kids who went to camp is 893,835 -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1538832 644997 = 893835 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l3336_333649


namespace NUMINAMATH_CALUDE_root_equation_problem_l3336_333669

theorem root_equation_problem (m r s a b : ℝ) : 
  (a^2 - m*a + 4 = 0) →
  (b^2 - m*b + 4 = 0) →
  ((a^2 + 1/b)^2 - r*(a^2 + 1/b) + s = 0) →
  ((b^2 + 1/a)^2 - r*(b^2 + 1/a) + s = 0) →
  s = m + 16.25 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3336_333669


namespace NUMINAMATH_CALUDE_claire_photos_l3336_333615

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 10) :
  claire = 5 := by
sorry

end NUMINAMATH_CALUDE_claire_photos_l3336_333615
