import Mathlib

namespace y_coordinate_of_C_l2511_251139

-- Define the pentagon ABCDE
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the properties of the pentagon
def symmetricPentagon (p : Pentagon) : Prop :=
  p.A.1 = 0 ∧ p.A.2 = 0 ∧
  p.B.1 = 0 ∧ p.B.2 = 5 ∧
  p.D.1 = 5 ∧ p.D.2 = 5 ∧
  p.E.1 = 5 ∧ p.E.2 = 0 ∧
  p.C.1 = 2.5 -- Vertical line of symmetry

-- Define the area of the pentagon
def pentagonArea (p : Pentagon) : ℝ :=
  50 -- Given area

-- Theorem: The y-coordinate of vertex C is 15
theorem y_coordinate_of_C (p : Pentagon) 
  (h1 : symmetricPentagon p) 
  (h2 : pentagonArea p = 50) : 
  p.C.2 = 15 := by
  sorry

end y_coordinate_of_C_l2511_251139


namespace inequality_proof_l2511_251191

theorem inequality_proof (a m n p : ℝ) 
  (h1 : a * Real.log a = 1)
  (h2 : m = Real.exp (1/2 + a))
  (h3 : Real.exp n = 3^a)
  (h4 : a^p = 2^Real.exp 1) : 
  n < p ∧ p < m := by
  sorry

end inequality_proof_l2511_251191


namespace solution_product_l2511_251170

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 11) = r^2 - 14 * r + 48 →
  (s - 3) * (3 * s + 11) = s^2 - 14 * s + 48 →
  r ≠ s →
  (r + 4) * (s + 4) = -226 := by
sorry

end solution_product_l2511_251170


namespace quadratic_symmetry_l2511_251104

/-- A function representing quadratic variation of y with respect to x -/
def quadratic_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem quadratic_symmetry (k : ℝ) :
  quadratic_variation k 5 = 25 →
  quadratic_variation k (-5) = 25 := by
  sorry

#check quadratic_symmetry

end quadratic_symmetry_l2511_251104


namespace tank_capacity_after_adding_gas_l2511_251115

/-- 
Given a tank with a capacity of 48 gallons, initially filled to 3/4 of its capacity,
prove that after adding 8 gallons of gasoline, the tank will be filled to 11/12 of its capacity.
-/
theorem tank_capacity_after_adding_gas (tank_capacity : ℚ) (initial_fill_fraction : ℚ) 
  (added_gas : ℚ) (final_fill_fraction : ℚ) : 
  tank_capacity = 48 → 
  initial_fill_fraction = 3/4 → 
  added_gas = 8 → 
  final_fill_fraction = (initial_fill_fraction * tank_capacity + added_gas) / tank_capacity →
  final_fill_fraction = 11/12 := by
sorry

end tank_capacity_after_adding_gas_l2511_251115


namespace discount_order_difference_l2511_251180

-- Define the original price and discounts
def original_price : ℝ := 30
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.25

-- Define the two discount application orders
def discount_flat_then_percent : ℝ := (original_price - flat_discount) * (1 - percentage_discount)
def discount_percent_then_flat : ℝ := (original_price * (1 - percentage_discount)) - flat_discount

-- Theorem statement
theorem discount_order_difference :
  discount_flat_then_percent - discount_percent_then_flat = 1.25 := by
  sorry

end discount_order_difference_l2511_251180


namespace moose_population_canada_l2511_251165

theorem moose_population_canada :
  ∀ (moose beaver human : ℕ),
    beaver = 2 * moose →
    human = 19 * beaver →
    human = 38000000 →
    moose = 1000000 :=
by
  sorry

end moose_population_canada_l2511_251165


namespace arithmetic_geometric_inequality_l2511_251182

/-- Given a, b, s, t, u, v are real numbers satisfying the following conditions:
    - 0 < a < b
    - a, s, t, b form an arithmetic sequence
    - a, u, v, b form a geometric sequence
    Prove that s * t * (s + t) > u * v * (u + v)
-/
theorem arithmetic_geometric_inequality (a b s t u v : ℝ) 
  (h1 : 0 < a) (h2 : a < b)
  (h3 : s = (2*a + b)/3) (h4 : t = (a + 2*b)/3)  -- arithmetic sequence condition
  (h5 : u = (a^2 * b)^(1/3)) (h6 : v = (a * b^2)^(1/3))  -- geometric sequence condition
  : s * t * (s + t) > u * v * (u + v) := by
  sorry

end arithmetic_geometric_inequality_l2511_251182


namespace parallel_lines_intersection_l2511_251132

theorem parallel_lines_intersection (n : ℕ) : 
  (10 - 1) * (n - 1) = 1260 → n = 141 :=
by sorry

end parallel_lines_intersection_l2511_251132


namespace q_value_l2511_251186

-- Define the polynomial Q(x)
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- Define the property that the mean of zeros, twice the product of zeros, and sum of coefficients are equal
def property (p q d : ℝ) : Prop :=
  let sum_of_zeros := -p
  let product_of_zeros := -d
  let sum_of_coefficients := 1 + p + q + d
  (sum_of_zeros / 3 = 2 * product_of_zeros) ∧ (sum_of_zeros / 3 = sum_of_coefficients)

-- Theorem statement
theorem q_value (p q d : ℝ) :
  property p q d → Q p q d 0 = 4 → q = -37 := by
  sorry

end q_value_l2511_251186


namespace angel_envelopes_l2511_251187

/-- The number of large envelopes Angel used --/
def large_envelopes : ℕ := 11

/-- The number of medium envelopes Angel used --/
def medium_envelopes : ℕ := 2 * large_envelopes

/-- The number of letters in small envelopes --/
def small_letters : ℕ := 20

/-- The number of letters in each medium envelope --/
def letters_per_medium : ℕ := 3

/-- The number of letters in each large envelope --/
def letters_per_large : ℕ := 5

/-- The total number of letters --/
def total_letters : ℕ := 150

theorem angel_envelopes :
  small_letters +
  medium_envelopes * letters_per_medium +
  large_envelopes * letters_per_large = total_letters :=
by sorry

end angel_envelopes_l2511_251187


namespace mets_fans_count_l2511_251178

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def baseball_town (fans : FanCounts) : Prop :=
  fans.yankees * 2 = fans.mets * 3 ∧
  fans.mets * 5 = fans.redsox * 4 ∧
  fans.yankees + fans.mets + fans.redsox = 330

/-- The theorem to prove -/
theorem mets_fans_count (fans : FanCounts) :
  baseball_town fans → fans.mets = 88 := by
  sorry


end mets_fans_count_l2511_251178


namespace dodecagon_diagonals_doubled_l2511_251109

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The theorem stating that the number of diagonals in a dodecagon, when doubled, is 108 -/
theorem dodecagon_diagonals_doubled :
  2 * (num_diagonals dodecagon_sides) = 108 := by
  sorry

end dodecagon_diagonals_doubled_l2511_251109


namespace first_apartment_rent_l2511_251100

theorem first_apartment_rent (R : ℝ) : 
  R + 260 + (31 * 20 * 0.58) - (900 + 200 + (21 * 20 * 0.58)) = 76 → R = 800 := by
  sorry

end first_apartment_rent_l2511_251100


namespace ratio_problem_l2511_251105

theorem ratio_problem : ∃ x : ℚ, (150 : ℚ) / 1 = x / 2 ∧ x = 300 := by sorry

end ratio_problem_l2511_251105


namespace second_term_of_arithmetic_sequence_l2511_251137

/-- Given an arithmetic sequence where the sum of the first and third terms is 8,
    prove that the second term is 4. -/
theorem second_term_of_arithmetic_sequence (a d : ℝ) 
  (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_of_arithmetic_sequence_l2511_251137


namespace quadratic_solution_l2511_251142

theorem quadratic_solution (b : ℝ) : 
  ((-5 : ℝ)^2 + b * (-5) - 45 = 0) → b = -4 := by sorry

end quadratic_solution_l2511_251142


namespace product_negative_implies_zero_l2511_251121

theorem product_negative_implies_zero (a b : ℝ) (h : a * b < 0) :
  a^2 * abs b - b^2 * abs a + a * b * (abs a - abs b) = 0 := by
  sorry

end product_negative_implies_zero_l2511_251121


namespace diamond_calculation_l2511_251190

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29 / 132 := by sorry

end diamond_calculation_l2511_251190


namespace supermarket_spending_l2511_251158

theorem supermarket_spending (total : ℚ) : 
  (3/7 : ℚ) * total + (2/5 : ℚ) * total + (1/4 : ℚ) * total + (1/14 : ℚ) * total + 12 = total →
  total = 80 := by
  sorry

end supermarket_spending_l2511_251158


namespace equal_roots_quadratic_l2511_251172

/-- A quadratic equation x^2 - x + 2k = 0 has two equal real roots if and only if k = 1/8 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - x + 2*k = 0 ∧ (∀ y : ℝ, y^2 - y + 2*k = 0 → y = x)) ↔ k = 1/8 := by
sorry

end equal_roots_quadratic_l2511_251172


namespace total_price_is_23_l2511_251122

-- Define the price of cucumbers per kilogram
def cucumber_price : ℝ := 5

-- Define the price of tomatoes as 20% cheaper than cucumbers
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

-- Define the quantity of tomatoes and cucumbers
def tomato_quantity : ℝ := 2
def cucumber_quantity : ℝ := 3

-- Theorem statement
theorem total_price_is_23 :
  tomato_quantity * tomato_price + cucumber_quantity * cucumber_price = 23 := by
  sorry

end total_price_is_23_l2511_251122


namespace range_of_x_l2511_251129

theorem range_of_x (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) →
  x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) := by
  sorry

end range_of_x_l2511_251129


namespace perpendicular_lines_a_value_l2511_251166

/-- Given two perpendicular lines with direction vectors (4, -5) and (a, 2), prove that a = 5/2 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -5]
  let v2 : Fin 2 → ℝ := ![a, 2]
  (∀ i : Fin 2, (v1 i) * (v2 i) = 0) → a = 5/2 := by
sorry

end perpendicular_lines_a_value_l2511_251166


namespace jack_driving_years_l2511_251119

/-- Represents the number of miles Jack drives in four months -/
def miles_per_four_months : ℕ := 37000

/-- Represents the total number of miles Jack has driven -/
def total_miles_driven : ℕ := 999000

/-- Calculates the number of years Jack has been driving -/
def years_driving : ℚ :=
  total_miles_driven / (miles_per_four_months * 3)

/-- Theorem stating that Jack has been driving for 9 years -/
theorem jack_driving_years :
  years_driving = 9 := by sorry

end jack_driving_years_l2511_251119


namespace open_box_volume_l2511_251103

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length sheet_width cut_size : ℕ)
  (h1 : sheet_length = 40)
  (h2 : sheet_width = 30)
  (h3 : cut_size = 8)
  : (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 2688 := by
  sorry

#check open_box_volume

end open_box_volume_l2511_251103


namespace volume_of_specific_polyhedron_l2511_251154

/-- A rectangle in 3D space -/
structure Rectangle3D where
  ab : ℝ
  bc : ℝ

/-- A line segment in 3D space -/
structure Segment3D where
  length : ℝ
  distance_from_plane : ℝ

/-- The volume of a polyhedron formed by a rectangle and a parallel segment -/
def polyhedron_volume (rect : Rectangle3D) (seg : Segment3D) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron ABCDKM -/
theorem volume_of_specific_polyhedron :
  let rect := Rectangle3D.mk 2 3
  let seg := Segment3D.mk 5 1
  polyhedron_volume rect seg = 9/2 := by sorry

end volume_of_specific_polyhedron_l2511_251154


namespace quadratic_root_sum_cubes_equals_sum_l2511_251111

theorem quadratic_root_sum_cubes_equals_sum (k : ℚ) : 
  (∃ a b : ℚ, (4 * a^2 + 5 * a + k = 0) ∧ 
               (4 * b^2 + 5 * b + k = 0) ∧ 
               (a ≠ b) ∧
               (a^3 + b^3 = a + b)) ↔ 
  (k = 9/4) := by
sorry

end quadratic_root_sum_cubes_equals_sum_l2511_251111


namespace watch_cost_price_l2511_251181

theorem watch_cost_price (loss_price gain_price : ℝ) : 
  loss_price = 0.9 * 1500 →
  gain_price = 1.04 * 1500 →
  gain_price - loss_price = 210 →
  1500 = (210 : ℝ) / 0.14 := by
  sorry

end watch_cost_price_l2511_251181


namespace intersection_complement_equality_l2511_251157

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 2, 4}
def B : Set Nat := {1, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_complement_equality_l2511_251157


namespace fourth_power_inequality_l2511_251116

theorem fourth_power_inequality (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 := by
  sorry

end fourth_power_inequality_l2511_251116


namespace abs_inequality_l2511_251185

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end abs_inequality_l2511_251185


namespace circle_area_tripled_radius_l2511_251198

theorem circle_area_tripled_radius (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A :=
by sorry

end circle_area_tripled_radius_l2511_251198


namespace symmetric_function_implies_a_eq_neg_four_l2511_251146

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- Theorem: If f(2-x) = f(2+x) for all x, then a = -4 -/
theorem symmetric_function_implies_a_eq_neg_four (a : ℝ) :
  (∀ x, f a (2 - x) = f a (2 + x)) → a = -4 :=
by
  sorry

end symmetric_function_implies_a_eq_neg_four_l2511_251146


namespace earthquake_energy_ratio_l2511_251167

-- Define the Richter scale energy relation
def richter_energy_ratio (x : ℝ) : ℝ := 10

-- Define the frequency function type
def frequency := ℝ → ℝ

-- Theorem statement
theorem earthquake_energy_ratio 
  (f : frequency) 
  (x y : ℝ) 
  (h1 : y - x = 2) 
  (h2 : f y = 2 * f x) :
  (richter_energy_ratio ^ y) / (richter_energy_ratio ^ x) = 200 := by
  sorry

end earthquake_energy_ratio_l2511_251167


namespace ken_kept_pencils_l2511_251193

def pencil_distribution (total : ℕ) (manny nilo carlos tina rina : ℕ) : Prop :=
  total = 200 ∧
  manny = 20 ∧
  nilo = manny + 10 ∧
  carlos = nilo + 5 ∧
  tina = carlos + 15 ∧
  rina = tina + 5

theorem ken_kept_pencils (total manny nilo carlos tina rina : ℕ) :
  pencil_distribution total manny nilo carlos tina rina →
  total - (manny + nilo + carlos + tina + rina) = 10 := by
  sorry

end ken_kept_pencils_l2511_251193


namespace billy_horse_feeding_days_billy_horse_feeding_problem_l2511_251175

theorem billy_horse_feeding_days 
  (num_horses : ℕ) 
  (oats_per_feeding : ℕ) 
  (feedings_per_day : ℕ) 
  (total_oats : ℕ) : ℕ :=
  let daily_oats_per_horse := oats_per_feeding * feedings_per_day
  let total_daily_oats := daily_oats_per_horse * num_horses
  total_oats / total_daily_oats

theorem billy_horse_feeding_problem :
  billy_horse_feeding_days 4 4 2 96 = 3 := by
  sorry

end billy_horse_feeding_days_billy_horse_feeding_problem_l2511_251175


namespace simplify_expression_1_simplify_expression_2_l2511_251112

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  (x - 2)^2 - (x - 3) * (x + 3) = -4 * x + 13 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 + 2*x) / (x^2 - 1) / (x + 1 + (2*x + 1) / (x - 1)) = 1 / (x + 1) := by sorry

end simplify_expression_1_simplify_expression_2_l2511_251112


namespace integer_solutions_of_equation_l2511_251150

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * y^2 - 2 * x * y + x + 9 * y - 2 = 0 ↔
    ((x = 9 ∧ y = 1) ∨ (x = 2 ∧ y = 0) ∨ (x = 8 ∧ y = 2) ∨ (x = 3 ∧ y = -1)) :=
by sorry

end integer_solutions_of_equation_l2511_251150


namespace quadratic_roots_implies_d_l2511_251173

theorem quadratic_roots_implies_d (d : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + d = 0 ↔ x = (-8 + Real.sqrt 12) / 4 ∨ x = (-8 - Real.sqrt 12) / 4) → 
  d = 6.5 := by
sorry

end quadratic_roots_implies_d_l2511_251173


namespace car_speed_ratio_l2511_251149

theorem car_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > 0 → v₂ > 0 →
  (3 * v₂ / v₁ - 3 * v₁ / v₂ = 1.1) →
  v₂ / v₁ = 6 / 5 := by
sorry

end car_speed_ratio_l2511_251149


namespace inequality_and_equality_condition_l2511_251127

theorem inequality_and_equality_condition (x y z : ℝ) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 ∧
  (x^2 + y^4 + z^6 = x*y^2 + y^2*z^3 + x*z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end inequality_and_equality_condition_l2511_251127


namespace sum_range_l2511_251108

theorem sum_range : 
  let sum := (25/8 : ℚ) + (31/7 : ℚ) + (128/21 : ℚ)
  (27/2 : ℚ) < sum ∧ sum < 14 := by
  sorry

end sum_range_l2511_251108


namespace no_points_in_circle_l2511_251152

theorem no_points_in_circle (r : ℝ) (A B : ℝ × ℝ) : r = 1 → 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 →
  ¬∃ P : ℝ × ℝ, (P.1 - A.1)^2 + (P.2 - A.2)^2 < r^2 ∧ 
                (P.1 - B.1)^2 + (P.2 - B.2)^2 < r^2 ∧
                (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5 :=
by sorry

end no_points_in_circle_l2511_251152


namespace largest_c_value_l2511_251107

/-- The function f(x) = x^2 - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- 2 is in the range of f -/
def two_in_range (c : ℝ) : Prop := ∃ x, f c x = 2

theorem largest_c_value :
  (∃ c_max : ℝ, two_in_range c_max ∧ ∀ c > c_max, ¬(two_in_range c)) ∧
  (∀ c_max : ℝ, (two_in_range c_max ∧ ∀ c > c_max, ¬(two_in_range c)) → c_max = 11) :=
sorry

end largest_c_value_l2511_251107


namespace quadratic_vertex_form_l2511_251141

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x + 3/2)^2 + k := by
  sorry

end quadratic_vertex_form_l2511_251141


namespace smallest_n_for_inequality_l2511_251106

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 64 ∧ 
  (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧
  (∀ (m : ℕ), m < n → ∃ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 > m * (w^6 + x^6 + y^6 + z^6)) :=
by sorry

end smallest_n_for_inequality_l2511_251106


namespace average_of_four_data_points_l2511_251160

theorem average_of_four_data_points
  (n : ℕ)
  (total_average : ℚ)
  (one_data_point : ℚ)
  (h1 : n = 5)
  (h2 : total_average = 81)
  (h3 : one_data_point = 85) :
  (n : ℚ) * total_average - one_data_point = (n - 1 : ℚ) * 80 :=
by sorry

end average_of_four_data_points_l2511_251160


namespace school_population_l2511_251102

theorem school_population (b g t : ℕ) : 
  b = 4 * g → 
  g = 10 * t → 
  b + g + t = (51 * b) / 40 := by
sorry

end school_population_l2511_251102


namespace xyz_value_l2511_251197

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 3)
  (eq3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
sorry

end xyz_value_l2511_251197


namespace remaining_amount_with_taxes_remaining_amount_is_622_54_l2511_251138

/-- Calculates the remaining amount to be paid including taxes for a product purchase --/
theorem remaining_amount_with_taxes (deposit_percent : ℝ) (cash_deposit : ℝ) (reward_points : ℕ) 
  (point_value : ℝ) (tax_rate : ℝ) (luxury_tax_rate : ℝ) : ℝ :=
  let total_deposit := cash_deposit + (reward_points : ℝ) * point_value
  let total_price := total_deposit / deposit_percent
  let remaining_before_taxes := total_price - total_deposit
  let tax := remaining_before_taxes * tax_rate
  let luxury_tax := remaining_before_taxes * luxury_tax_rate
  remaining_before_taxes + tax + luxury_tax

/-- The remaining amount to be paid including taxes is $622.54 --/
theorem remaining_amount_is_622_54 :
  remaining_amount_with_taxes 0.30 150 800 0.10 0.12 0.04 = 622.54 := by
  sorry

end remaining_amount_with_taxes_remaining_amount_is_622_54_l2511_251138


namespace hands_closest_and_farthest_l2511_251145

/-- Represents a time between 6:30 and 6:35 -/
inductive ClockTime
  | t630
  | t631
  | t632
  | t633
  | t634
  | t635

/-- Calculates the angle between hour and minute hands for a given time -/
def angleBetweenHands (t : ClockTime) : ℝ :=
  match t with
  | ClockTime.t630 => 15
  | ClockTime.t631 => 9.5
  | ClockTime.t632 => 4
  | ClockTime.t633 => 1.5
  | ClockTime.t634 => 7
  | ClockTime.t635 => 12.5

theorem hands_closest_and_farthest :
  (∀ t : ClockTime, angleBetweenHands ClockTime.t633 ≤ angleBetweenHands t) ∧
  (∀ t : ClockTime, angleBetweenHands t ≤ angleBetweenHands ClockTime.t630) :=
by sorry


end hands_closest_and_farthest_l2511_251145


namespace piano_cost_solution_l2511_251148

def piano_cost_problem (total_lessons : ℕ) (lesson_cost : ℚ) (discount_percent : ℚ) (total_cost : ℚ) : Prop :=
  let original_lesson_cost := total_lessons * lesson_cost
  let discount_amount := discount_percent * original_lesson_cost
  let discounted_lesson_cost := original_lesson_cost - discount_amount
  let piano_cost := total_cost - discounted_lesson_cost
  piano_cost = 500

theorem piano_cost_solution :
  piano_cost_problem 20 40 0.25 1100 := by
  sorry

end piano_cost_solution_l2511_251148


namespace simple_interest_problem_l2511_251155

theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_R : R > 0) :
  (P * (R + 2) * 5) / 100 - (P * R * 5) / 100 = 250 →
  P = 2500 := by
sorry

end simple_interest_problem_l2511_251155


namespace dice_roll_probability_l2511_251153

-- Define the type for dice rolls
def DiceRoll := Fin 6

-- Define the condition for the angle being greater than 90°
def angleGreaterThan90 (m n : DiceRoll) : Prop :=
  (m.val : ℤ) - (n.val : ℤ) > 0

-- Define the probability space
def totalOutcomes : ℕ := 36

-- Define the number of favorable outcomes
def favorableOutcomes : ℕ := 15

-- State the theorem
theorem dice_roll_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 12 := by
  sorry

end dice_roll_probability_l2511_251153


namespace geometric_arithmetic_sequence_ratio_l2511_251113

theorem geometric_arithmetic_sequence_ratio 
  (x y z : ℝ) 
  (h_geometric : ∃ q : ℝ, y = x * q ∧ z = y * q) 
  (h_arithmetic : ∃ d : ℝ, y + z = (x + y) + d ∧ z + x = (y + z) + d) :
  ∃ q : ℝ, (y = x * q ∧ z = y * q) ∧ (q = -2 ∨ q = 1) :=
sorry

end geometric_arithmetic_sequence_ratio_l2511_251113


namespace cos_neg_thirty_degrees_l2511_251125

theorem cos_neg_thirty_degrees : Real.cos (-(30 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end cos_neg_thirty_degrees_l2511_251125


namespace exactly_one_correct_statement_l2511_251128

-- Define the type for geometric statements
inductive GeometricStatement
  | uniquePerpendicular
  | perpendicularIntersect
  | equalVertical
  | distanceDefinition
  | uniqueParallel

-- Function to check if a statement is correct
def isCorrect (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.perpendicularIntersect => True
  | _ => False

-- Theorem stating that exactly one statement is correct
theorem exactly_one_correct_statement :
  ∃! (s : GeometricStatement), isCorrect s :=
  sorry

end exactly_one_correct_statement_l2511_251128


namespace bucket_capacity_l2511_251183

/-- Represents the number of buckets needed to fill the bathtub to the top -/
def full_bathtub : ℕ := 14

/-- Represents the number of buckets removed to reach the bath level -/
def removed_buckets : ℕ := 3

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the total amount of water used in a week (in ounces) -/
def weekly_water_usage : ℕ := 9240

/-- Calculates the number of buckets used for each bath -/
def buckets_per_bath : ℕ := full_bathtub - removed_buckets

/-- Calculates the number of buckets used in a week -/
def weekly_buckets : ℕ := buckets_per_bath * days_per_week

/-- Theorem: The bucket holds 120 ounces of water -/
theorem bucket_capacity : weekly_water_usage / weekly_buckets = 120 := by
  sorry

end bucket_capacity_l2511_251183


namespace parabola_intersection_difference_l2511_251123

theorem parabola_intersection_difference : ∃ (a b c d : ℝ),
  (3 * a^2 - 6 * a + 5 = -2 * a^2 - 3 * a + 7) ∧
  (3 * c^2 - 6 * c + 5 = -2 * c^2 - 3 * c + 7) ∧
  c ≥ a ∧
  c - a = 7/5 := by sorry

end parabola_intersection_difference_l2511_251123


namespace unknown_rate_is_225_l2511_251194

/-- The unknown rate of two blankets given the following conditions:
    - 3 blankets at Rs. 100 each
    - 6 blankets at Rs. 150 each
    - 2 blankets at an unknown rate
    - The average price of all blankets is Rs. 150
-/
def unknown_rate : ℕ := by
  -- Define the known quantities
  let blankets_100 : ℕ := 3
  let price_100 : ℕ := 100
  let blankets_150 : ℕ := 6
  let price_150 : ℕ := 150
  let blankets_unknown : ℕ := 2
  let average_price : ℕ := 150

  -- Calculate the total number of blankets
  let total_blankets : ℕ := blankets_100 + blankets_150 + blankets_unknown

  -- Calculate the total cost of all blankets
  let total_cost : ℕ := average_price * total_blankets

  -- Calculate the cost of known blankets
  let cost_known : ℕ := blankets_100 * price_100 + blankets_150 * price_150

  -- Calculate the cost of unknown blankets
  let cost_unknown : ℕ := total_cost - cost_known

  -- Calculate the rate of each unknown blanket
  exact cost_unknown / blankets_unknown

theorem unknown_rate_is_225 : unknown_rate = 225 := by
  sorry

end unknown_rate_is_225_l2511_251194


namespace cost_price_of_article_l2511_251130

/-- Proves that the cost price of an article is 800, given the conditions from the problem. -/
theorem cost_price_of_article : ∃ (C : ℝ), 
  (C = 800) ∧ 
  (1.05 * C - (0.95 * C + 0.1 * (0.95 * C)) = 4) := by
  sorry

end cost_price_of_article_l2511_251130


namespace swimmers_meet_problem_l2511_251140

/-- Represents the number of times two swimmers meet in a pool -/
def swimmers_meet (pool_length : ℝ) (speed_a speed_b : ℝ) (time : ℝ) : ℕ :=
  sorry

theorem swimmers_meet_problem :
  swimmers_meet 90 3 2 (12 * 60) = 20 := by sorry

end swimmers_meet_problem_l2511_251140


namespace min_sequence_length_l2511_251199

def S : Finset Nat := {1, 2, 3, 4}

def ValidSequence (seq : List Nat) : Prop :=
  ∀ B : Finset Nat, B ⊆ S → B.Nonempty → 
    ∃ subseq : List Nat, subseq.length = B.card ∧ 
      subseq.toFinset = B ∧ seq.Sublist subseq

theorem min_sequence_length : 
  (∃ seq : List Nat, ValidSequence seq ∧ seq.length = 8) ∧
  (∀ seq : List Nat, ValidSequence seq → seq.length ≥ 8) :=
sorry

end min_sequence_length_l2511_251199


namespace marcus_pies_l2511_251126

/-- The number of pies Marcus can fit in his oven at once -/
def oven_capacity : ℕ := 5

/-- The number of pies Marcus dropped -/
def dropped_pies : ℕ := 8

/-- The number of pies Marcus has left -/
def remaining_pies : ℕ := 27

/-- The number of batches Marcus baked -/
def batches : ℕ := 7

theorem marcus_pies :
  oven_capacity * batches = remaining_pies + dropped_pies :=
sorry

end marcus_pies_l2511_251126


namespace ali_class_size_l2511_251162

/-- Calculates the total number of students in a class given a student's rank from top and bottom -/
def class_size (rank_from_top : ℕ) (rank_from_bottom : ℕ) : ℕ :=
  rank_from_top + rank_from_bottom - 1

/-- Theorem: In a class where a student ranks 40th from both the top and bottom, the total number of students is 79 -/
theorem ali_class_size :
  class_size 40 40 = 79 := by
  sorry

#eval class_size 40 40

end ali_class_size_l2511_251162


namespace product_of_sum_of_roots_l2511_251192

theorem product_of_sum_of_roots (x : ℝ) :
  (Real.sqrt (5 + x) + Real.sqrt (25 - x) = 8) →
  (5 + x) * (25 - x) = 289 := by
  sorry

end product_of_sum_of_roots_l2511_251192


namespace min_value_sine_l2511_251156

/-- Given that f(x) = 3sin(x) - cos(x) attains its minimum value when x = θ, prove that sin(θ) = -3√10/10 -/
theorem min_value_sine (θ : ℝ) (h : ∀ x, 3 * Real.sin x - Real.cos x ≥ 3 * Real.sin θ - Real.cos θ) : 
  Real.sin θ = -3 * Real.sqrt 10 / 10 := by
  sorry

end min_value_sine_l2511_251156


namespace coins_problem_l2511_251161

theorem coins_problem (x : ℚ) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (4 : ℚ) / 5 * lost
  let remaining := x - lost + found
  x - remaining = (2 : ℚ) / 15 * x :=
by sorry

end coins_problem_l2511_251161


namespace custom_operations_fraction_l2511_251174

-- Define the custom operations
def oplus (a b : ℝ) : ℝ := a * b + b^2
def otimes (a b : ℝ) : ℝ := a - b + a * b^2

-- State the theorem
theorem custom_operations_fraction :
  (oplus 8 3) / (otimes 8 3) = 33 / 77 := by sorry

end custom_operations_fraction_l2511_251174


namespace calculate_expression_l2511_251147

theorem calculate_expression : 2023^0 + (-1/3) = 2/3 := by
  sorry

end calculate_expression_l2511_251147


namespace rectangle_with_cut_corners_l2511_251131

/-- Given a rectangle ABCD with identical isosceles right triangles cut off from its corners,
    each having a leg of length a, and the total area cut off is 160 cm²,
    if the longer side of ABCD is 32√2 cm, then the length of PQ is 16√2 cm. -/
theorem rectangle_with_cut_corners (a : ℝ) (l : ℝ) (PQ : ℝ) :
  (4 * (1/2 * a^2) = 160) →  -- Total area cut off
  (l = 32 * Real.sqrt 2) →   -- Longer side of ABCD
  (PQ = l - 2*a) →           -- Definition of PQ
  (PQ = 16 * Real.sqrt 2) :=
by sorry

end rectangle_with_cut_corners_l2511_251131


namespace fourth_root_of_506250000_l2511_251110

theorem fourth_root_of_506250000 : (506250000 : ℝ) ^ (1/4 : ℝ) = 150 := by sorry

end fourth_root_of_506250000_l2511_251110


namespace magnitude_of_3_minus_i_l2511_251136

/-- Given a complex number z = 3 - i, prove that its magnitude |z| is equal to √10 -/
theorem magnitude_of_3_minus_i :
  let z : ℂ := 3 - I
  Complex.abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_3_minus_i_l2511_251136


namespace solution_in_interval_l2511_251120

def f (x : ℝ) := 4 * x^3 + x - 8

theorem solution_in_interval :
  (f 2 > 0) →
  (f 1.5 > 0) →
  (f 1 < 0) →
  ∃ x, x > 1 ∧ x < 1.5 ∧ f x = 0 :=
by sorry

end solution_in_interval_l2511_251120


namespace zeros_not_adjacent_probability_l2511_251184

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with three ones in a row -/
def prob_zeros_not_adjacent : ℚ := 3/5

theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 3/5 :=
sorry

end zeros_not_adjacent_probability_l2511_251184


namespace magic_star_sum_l2511_251134

/-- Represents a 6th-order magic star -/
structure MagicStar :=
  (numbers : Finset ℕ)
  (lines : Finset (Finset ℕ))
  (h_numbers : numbers = Finset.range 12)
  (h_lines_count : lines.card = 6)
  (h_line_size : ∀ l ∈ lines, l.card = 4)
  (h_numbers_in_lines : ∀ n ∈ numbers, (lines.filter (λ l => n ∈ l)).card = 2)
  (h_line_sum_equal : ∃ s, ∀ l ∈ lines, l.sum id = s)

/-- The magic sum of a 6th-order magic star is 26 -/
theorem magic_star_sum (ms : MagicStar) : 
  ∃ (s : ℕ), (∀ l ∈ ms.lines, l.sum id = s) ∧ s = 26 := by
  sorry

end magic_star_sum_l2511_251134


namespace inequality_proof_l2511_251189

theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : 0 ≤ x ∧ x ≤ π / 2) :
  a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a := by
  sorry

end inequality_proof_l2511_251189


namespace set_operations_l2511_251176

def A : Set ℤ := {1,2,3,4,5}
def B : Set ℤ := {-1,1,2,3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {1,2,3}) ∧
  (A ∪ B = {-1,1,2,3,4,5}) ∧
  ((U \ B) ∩ A = {4,5}) := by sorry

end set_operations_l2511_251176


namespace sum_of_fractions_value_of_m_l2511_251169

noncomputable section

variable (θ : Real)
variable (m : Real)

-- Define the equation and its roots
def equation (x : Real) := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

-- Conditions
axiom theta_range : 0 < θ ∧ θ < 2 * Real.pi
axiom roots : equation (Real.sin θ) = 0 ∧ equation (Real.cos θ) = 0

-- Theorems to prove
theorem sum_of_fractions :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem value_of_m : m = Real.sqrt 3 / 2 :=
sorry

end sum_of_fractions_value_of_m_l2511_251169


namespace number_with_specific_remainders_l2511_251196

theorem number_with_specific_remainders (x : ℤ) :
  x % 7 = 3 →
  x^2 % 49 = 44 →
  x^3 % 343 = 111 →
  x % 343 = 17 := by
sorry

end number_with_specific_remainders_l2511_251196


namespace quadratic_transformation_l2511_251164

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) →
  ∃ m k, ∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - 5)^2 + k :=
by sorry

end quadratic_transformation_l2511_251164


namespace rectangle_dimensions_area_l2511_251114

theorem rectangle_dimensions_area (x : ℝ) : 
  (2*x - 3 > 0) → 
  (3*x + 4 > 0) → 
  (2*x - 3) * (3*x + 4) = 14*x - 6 → 
  x = (5 + Real.sqrt 41) / 4 :=
by sorry

end rectangle_dimensions_area_l2511_251114


namespace imaginary_part_of_complex_fraction_l2511_251135

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 * I + 1) / (1 - I) → z.im = 2 := by
  sorry

end imaginary_part_of_complex_fraction_l2511_251135


namespace train_passing_time_l2511_251171

/-- Time taken for trains to pass each other -/
theorem train_passing_time (man_speed goods_speed : ℝ) (goods_length : ℝ) : 
  man_speed = 80 →
  goods_speed = 32 →
  goods_length = 280 →
  let relative_speed := (man_speed + goods_speed) * 1000 / 3600
  let time := goods_length / relative_speed
  ∃ ε > 0, |time - 8.993| < ε :=
by sorry

end train_passing_time_l2511_251171


namespace number_sequence_problem_l2511_251177

theorem number_sequence_problem :
  ∃ k : ℕ+,
    let a : ℕ+ → ℤ := λ n => (-2) ^ n.val
    let b : ℕ+ → ℤ := λ n => a n + 2
    let c : ℕ+ → ℚ := λ n => (1 / 2 : ℚ) * (a n)
    (a k + b k + c k = 642) ∧ (a k = 256) :=
by
  sorry

end number_sequence_problem_l2511_251177


namespace negation_existence_cube_plus_one_l2511_251144

theorem negation_existence_cube_plus_one (x : ℝ) :
  (¬ ∃ x, x^3 + 1 = 0) ↔ ∀ x, x^3 + 1 ≠ 0 :=
by sorry

end negation_existence_cube_plus_one_l2511_251144


namespace square_grid_division_l2511_251188

theorem square_grid_division (m n k : ℕ) (h : m * m = n * k) :
  ∃ (d : ℕ), d ∣ m ∧ d ∣ n ∧ (m / d) * d = k ∧ (n / d) * d = n :=
sorry

end square_grid_division_l2511_251188


namespace intersection_point_a_l2511_251179

/-- A linear function f(x) = 4x + b -/
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

/-- The inverse of f -/
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 4

theorem intersection_point_a (b : ℤ) (a : ℤ) :
  f b 4 = a ∧ f_inv b a = 4 → a = 4 := by sorry

end intersection_point_a_l2511_251179


namespace jiayuan_supermarket_fruit_weight_l2511_251124

theorem jiayuan_supermarket_fruit_weight :
  let apple_baskets : ℕ := 62
  let pear_baskets : ℕ := 38
  let weight_per_basket : ℕ := 25
  apple_baskets * weight_per_basket + pear_baskets * weight_per_basket = 2500 := by
  sorry

end jiayuan_supermarket_fruit_weight_l2511_251124


namespace remainder_problem_l2511_251195

theorem remainder_problem (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end remainder_problem_l2511_251195


namespace square_tiles_count_l2511_251151

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30)
  (h_total_edges : total_edges = 100) :
  ∃ (triangles squares pentagons : ℕ),
    triangles + squares + pentagons = total_tiles ∧
    3 * triangles + 4 * squares + 5 * pentagons = total_edges ∧
    squares = 10 := by
  sorry

end square_tiles_count_l2511_251151


namespace gcd_of_powers_of_two_minus_one_l2511_251163

theorem gcd_of_powers_of_two_minus_one :
  Nat.gcd (2^2100 - 1) (2^1950 - 1) = 2^150 - 1 := by
  sorry

end gcd_of_powers_of_two_minus_one_l2511_251163


namespace vikki_tax_percentage_l2511_251118

/-- Calculates the tax percentage given the working conditions and take-home pay --/
def calculate_tax_percentage (hours_worked : ℕ) (hourly_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) (take_home_pay : ℚ) : ℚ :=
  let gross_earnings := hours_worked * hourly_rate
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := gross_earnings - take_home_pay
  let tax_deduction := total_deductions - insurance_deduction - union_dues
  (tax_deduction / gross_earnings) * 100

/-- Theorem stating that the tax percentage is 20% given Vikki's working conditions --/
theorem vikki_tax_percentage :
  calculate_tax_percentage 42 10 (5/100) 5 310 = 20 := by
  sorry

end vikki_tax_percentage_l2511_251118


namespace larger_part_of_66_l2511_251168

theorem larger_part_of_66 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_sum : x + y = 66) (h_relation : 0.40 * x = 0.625 * y + 10) : 
  max x y = 50 := by
sorry

end larger_part_of_66_l2511_251168


namespace pencils_left_l2511_251143

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took out -/
def pencils_taken : ℕ := 22

/-- The number of pencils Dan returned -/
def pencils_returned : ℕ := 5

/-- Theorem: The number of pencils left in the drawer is 17 -/
theorem pencils_left : initial_pencils - (pencils_taken - pencils_returned) = 17 := by
  sorry

#eval initial_pencils - (pencils_taken - pencils_returned)

end pencils_left_l2511_251143


namespace sum_of_digits_up_to_1000_l2511_251117

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := (Finset.range n).sum sumOfDigits

theorem sum_of_digits_up_to_1000 : sumOfDigitsUpTo 1000 = 14446 := by sorry

end sum_of_digits_up_to_1000_l2511_251117


namespace race_probability_l2511_251133

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) 
  (h_total : total_cars = 12)
  (h_x : prob_x = 1 / 6)
  (h_y : prob_y = 1 / 10)
  (h_z : prob_z = 1 / 8)
  (h_no_tie : ∀ a b : ℕ, a ≠ b → a ≤ total_cars → b ≤ total_cars → 
    (∃ t : ℚ, t > 0 ∧ t < 1 ∧ prob_x + prob_y + prob_z + t = 1)) :
  prob_x + prob_y + prob_z = 47 / 120 :=
sorry

end race_probability_l2511_251133


namespace average_of_first_21_multiples_of_6_l2511_251159

/-- The average of the first n multiples of a number -/
def averageOfMultiples (n : ℕ) (x : ℕ) : ℚ :=
  (n * x * (n + 1)) / (2 * n)

/-- Theorem: The average of the first 21 multiples of 6 is 66 -/
theorem average_of_first_21_multiples_of_6 :
  averageOfMultiples 21 6 = 66 := by
  sorry

end average_of_first_21_multiples_of_6_l2511_251159


namespace village_income_growth_and_prediction_l2511_251101

/-- Represents the annual average growth rate calculation and prediction for a village's per capita income. -/
theorem village_income_growth_and_prediction 
  (initial_income : ℝ) 
  (final_income : ℝ) 
  (years : ℕ) 
  (growth_rate : ℝ) 
  (predicted_income : ℝ)
  (h1 : initial_income = 20000)
  (h2 : final_income = 24200)
  (h3 : years = 2) :
  (final_income = initial_income * (1 + growth_rate) ^ years ∧ 
   growth_rate = 0.1 ∧
   predicted_income = final_income * (1 + growth_rate)) := by
  sorry

#check village_income_growth_and_prediction

end village_income_growth_and_prediction_l2511_251101
