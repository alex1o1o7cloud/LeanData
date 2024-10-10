import Mathlib

namespace divisibility_implies_inequality_l3749_374906

theorem divisibility_implies_inequality (a k : ℕ+) :
  (a^2 + k : ℕ) ∣ ((a - 1) * a * (a + 1) : ℕ) → k ≥ a :=
by sorry

end divisibility_implies_inequality_l3749_374906


namespace solution_set_implies_a_value_l3749_374997

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 := by
  sorry

end solution_set_implies_a_value_l3749_374997


namespace factor_expression_l3749_374942

theorem factor_expression (x : ℝ) : 72 * x^4 - 252 * x^9 = 36 * x^4 * (2 - 7 * x^5) := by
  sorry

end factor_expression_l3749_374942


namespace reflect_2_5_across_x_axis_l3749_374945

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: Reflecting the point (2,5) across the x-axis results in (2,-5) -/
theorem reflect_2_5_across_x_axis :
  reflectAcrossXAxis { x := 2, y := 5 } = { x := 2, y := -5 } := by
  sorry

end reflect_2_5_across_x_axis_l3749_374945


namespace invisible_dots_count_l3749_374989

/-- The sum of numbers on a standard six-sided die -/
def dieSumOfFaces : ℕ := 21

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 4

/-- The list of visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [1, 1, 2, 3, 4, 4, 5, 6]

/-- The theorem stating that the number of invisible dots is 58 -/
theorem invisible_dots_count : 
  numberOfDice * dieSumOfFaces - visibleNumbers.sum = 58 := by
  sorry

end invisible_dots_count_l3749_374989


namespace ab_range_l3749_374988

-- Define the line equation
def line_equation (a b x y : ℝ) : Prop := a * x - b * y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the property of bisecting the circumference
def bisects_circle (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), line_equation a b x y ∧ circle_equation x y

-- Theorem statement
theorem ab_range (a b : ℝ) : 
  bisects_circle a b → ab ∈ Set.Iic (1/8) :=
sorry

end ab_range_l3749_374988


namespace odd_decreasing_function_range_l3749_374904

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem odd_decreasing_function_range 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_decreasing : is_decreasing f (-1) 1) 
  (h_condition : ∀ a, f (1 - a) + (f (1 - a))^2 > 0) :
  ∃ a, 1 < a ∧ a ≤ Real.sqrt 2 :=
sorry

end odd_decreasing_function_range_l3749_374904


namespace largest_value_l3749_374950

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 6 * (6 ^ (1 / 6)))
  (hb : b = 6 ^ (1 / 3))
  (hc : c = 6 ^ (1 / 4))
  (hd : d = 2 * (6 ^ (1 / 3)))
  (he : e = 3 * (4 ^ (1 / 3))) :
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e :=
sorry

end largest_value_l3749_374950


namespace ram_price_decrease_l3749_374999

theorem ram_price_decrease (initial_price increased_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : increased_price = initial_price * 1.3)
  (h3 : final_price = 52) :
  (increased_price - final_price) / increased_price * 100 = 20 := by
sorry

end ram_price_decrease_l3749_374999


namespace even_function_implies_f_3_equals_5_l3749_374985

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) * (x - a)

-- State the theorem
theorem even_function_implies_f_3_equals_5 :
  (∀ x : ℝ, f a x = f a (-x)) → f a 3 = 5 := by
  sorry

end even_function_implies_f_3_equals_5_l3749_374985


namespace runner_area_theorem_l3749_374911

/-- Given a table and three runners, calculates the total area of the runners -/
def total_runner_area (table_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ) : ℝ :=
  let covered_area := 0.8 * table_area
  let single_layer_area := covered_area - double_layer_area - triple_layer_area
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area

/-- Theorem stating that under the given conditions, the total area of the runners is 168 square inches -/
theorem runner_area_theorem (table_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ) 
  (h1 : table_area = 175)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 28) :
  total_runner_area table_area double_layer_area triple_layer_area = 168 := by
  sorry

#eval total_runner_area 175 24 28

end runner_area_theorem_l3749_374911


namespace line_through_points_l3749_374928

/-- Given a line y = ax + b passing through points (3, 7) and (6, 19), prove that a - b = 9 -/
theorem line_through_points (a b : ℝ) : 
  (7 : ℝ) = a * 3 + b ∧ (19 : ℝ) = a * 6 + b → a - b = 9 := by
  sorry

end line_through_points_l3749_374928


namespace solution_difference_l3749_374925

theorem solution_difference (p q : ℝ) : 
  ((p - 4) * (p + 4) = 24 * p - 96) →
  ((q - 4) * (q + 4) = 24 * q - 96) →
  p ≠ q →
  p > q →
  p - q = 16 := by
sorry

end solution_difference_l3749_374925


namespace cubic_expression_value_l3749_374971

theorem cubic_expression_value (x : ℝ) (h : x^2 - 2*x - 1 = 0) :
  x^3 - x^2 - 3*x + 2 = 3 := by
sorry

end cubic_expression_value_l3749_374971


namespace cosine_of_arithmetic_sequence_l3749_374969

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem cosine_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = Real.pi) : 
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end cosine_of_arithmetic_sequence_l3749_374969


namespace staplers_left_proof_l3749_374905

/-- Represents the number of staplers used per report by Stacie -/
def stacie_rate : ℚ := 1

/-- Calculates the number of reports from dozens -/
def dozen_to_reports (dozens : ℕ) : ℕ := dozens * 12

theorem staplers_left_proof (initial_staplers : ℕ) 
  (stacie_dozens jack_dozens : ℕ) (laura_reports : ℕ) : 
  initial_staplers = 450 →
  stacie_dozens = 8 →
  jack_dozens = 9 →
  laura_reports = 50 →
  initial_staplers - 
    (stacie_rate * dozen_to_reports stacie_dozens +
     stacie_rate / 2 * dozen_to_reports jack_dozens +
     stacie_rate * 2 * laura_reports) = 200 := by
  sorry

end staplers_left_proof_l3749_374905


namespace complete_square_factorization_quadratic_factorization_l3749_374930

/-- A quadratic expression ax^2 + bx + c can be factored using the complete square formula
    if and only if b = ±2√(ac) -/
theorem complete_square_factorization (a b c : ℝ) :
  (∃ (k : ℝ), b = 2 * k * Real.sqrt (a * c)) ∨ (∃ (k : ℝ), b = -2 * k * Real.sqrt (a * c)) ↔
  ∃ (p q : ℝ), a * x^2 + b * x + c = a * (x - p)^2 + q := sorry

/-- For the quadratic expression 4x^2 - (m+1)x + 9 to be factored using the complete square formula,
    m must equal 11 or -13 -/
theorem quadratic_factorization (m : ℝ) :
  (∃ (p q : ℝ), 4 * x^2 - (m + 1) * x + 9 = 4 * (x - p)^2 + q) ↔ (m = 11 ∨ m = -13) := by
  sorry

end complete_square_factorization_quadratic_factorization_l3749_374930


namespace two_point_eight_million_scientific_notation_l3749_374952

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_eight_million_scientific_notation :
  toScientificNotation 2800000 = ScientificNotation.mk 2.8 6 (by sorry) :=
sorry

end two_point_eight_million_scientific_notation_l3749_374952


namespace min_value_function_extremum_function_l3749_374955

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) + 6 ≥ 9 ∧
  (x + 4 / (x + 1) + 6 = 9 ↔ x = 1) :=
sorry

-- Part 2
theorem extremum_function (x : ℝ) (h : x > 1) :
  (x^2 + 8) / (x - 1) ≥ 8 ∧
  ((x^2 + 8) / (x - 1) = 8 ↔ x = 4) :=
sorry

end min_value_function_extremum_function_l3749_374955


namespace min_value_of_expression_l3749_374943

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  3 * x + 1 / (x^3) ≥ 4 ∧
  (3 * x + 1 / (x^3) = 4 ↔ x = 1) :=
by sorry

end min_value_of_expression_l3749_374943


namespace cos_equality_problem_l3749_374987

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end cos_equality_problem_l3749_374987


namespace centroid_tetrahedron_volume_centroid_tetrahedron_volume_54_l3749_374921

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- Represents the tetrahedron formed by the centroids of the faces of another tetrahedron -/
def centroid_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- The volume of the centroid tetrahedron is 1/27 of the original tetrahedron's volume -/
theorem centroid_tetrahedron_volume (t : RegularTetrahedron) :
  (centroid_tetrahedron t).volume = t.volume / 27 :=
sorry

/-- Given a regular tetrahedron with volume 54, the volume of the tetrahedron
    formed by the centroids of its four faces is 2 -/
theorem centroid_tetrahedron_volume_54 :
  let t : RegularTetrahedron := ⟨54⟩
  (centroid_tetrahedron t).volume = 2 :=
sorry

end centroid_tetrahedron_volume_centroid_tetrahedron_volume_54_l3749_374921


namespace mean_median_difference_l3749_374901

/-- Represents the score distribution of a math test -/
structure ScoreDistribution where
  score60 : Float
  score75 : Float
  score85 : Float
  score90 : Float
  score100 : Float
  sum_to_one : score60 + score75 + score85 + score90 + score100 = 1

/-- Calculates the mean score given a ScoreDistribution -/
def meanScore (d : ScoreDistribution) : Float :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 100 * d.score100

/-- Calculates the median score given a ScoreDistribution -/
def medianScore (d : ScoreDistribution) : Float :=
  if d.score60 + d.score75 > 0.5 then 75
  else if d.score60 + d.score75 + d.score85 > 0.5 then 85
  else 90

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.15)
  (h2 : d.score75 = 0.20)
  (h3 : d.score85 = 0.25)
  (h4 : d.score90 = 0.25) :
  medianScore d - meanScore d = 2.25 := by
  sorry


end mean_median_difference_l3749_374901


namespace log_relation_l3749_374938

theorem log_relation (y : ℝ) (k : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 625 / Real.log 2 = k * y) → 
  k = 12 := by sorry

end log_relation_l3749_374938


namespace volume_increase_rectangular_prism_l3749_374939

/-- Theorem: Increase in volume of a rectangular prism -/
theorem volume_increase_rectangular_prism 
  (L B H : ℝ) 
  (h_positive : L > 0 ∧ B > 0 ∧ H > 0) :
  let V_original := L * B * H
  let V_new := (L * 1.15) * (B * 1.30) * (H * 1.20)
  (V_new - V_original) / V_original = 0.794 := by
  sorry

end volume_increase_rectangular_prism_l3749_374939


namespace total_baked_goods_is_338_l3749_374965

/-- The total number of baked goods Diane makes -/
def total_baked_goods : ℕ :=
  let gingerbread_trays : ℕ := 4
  let gingerbread_per_tray : ℕ := 25
  let chocolate_chip_trays : ℕ := 3
  let chocolate_chip_per_tray : ℕ := 30
  let oatmeal_trays : ℕ := 2
  let oatmeal_per_tray : ℕ := 20
  let sugar_trays : ℕ := 6
  let sugar_per_tray : ℕ := 18
  gingerbread_trays * gingerbread_per_tray +
  chocolate_chip_trays * chocolate_chip_per_tray +
  oatmeal_trays * oatmeal_per_tray +
  sugar_trays * sugar_per_tray

theorem total_baked_goods_is_338 : total_baked_goods = 338 := by
  sorry

end total_baked_goods_is_338_l3749_374965


namespace BG_length_l3749_374992

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  (AB_length : B.1 - A.1 = 6)
  (BC_length : C.2 - B.2 = 4)

-- Define point E on BC
def E (rect : Rectangle) : ℝ × ℝ := (rect.B.1, rect.B.2 + 3)

-- Define point F on AE
def F (rect : Rectangle) : ℝ × ℝ := (4, 2)

-- Define point G as intersection of DF and BC
def G (rect : Rectangle) : ℝ × ℝ := (rect.B.1, 1)

-- Theorem statement
theorem BG_length (rect : Rectangle) : (G rect).2 - rect.B.2 = 1 := by
  sorry

end BG_length_l3749_374992


namespace divisibility_condition_l3749_374954

/-- s_n is the sum of all integers in [1,n] that are mutually prime to n -/
def s_n (n : ℕ) : ℕ := sorry

/-- t_n is the sum of the remaining integers in [1,n] -/
def t_n (n : ℕ) : ℕ := sorry

/-- Theorem: For all integers n ≥ 2, n divides (s_n - t_n) if and only if n is odd -/
theorem divisibility_condition (n : ℕ) (h : n ≥ 2) :
  n ∣ (s_n n - t_n n) ↔ Odd n := by
  sorry

end divisibility_condition_l3749_374954


namespace project_payment_main_project_payment_l3749_374920

/-- Represents the project details and calculates the total payment -/
structure Project where
  q_wage : ℝ  -- Hourly wage of candidate q
  p_hours : ℝ  -- Hours required by candidate p to complete the project
  total_payment : ℝ  -- Total payment for the project

/-- Theorem stating the total payment for the project is $540 -/
theorem project_payment (proj : Project) : proj.total_payment = 540 :=
  by
  have h1 : proj.q_wage + proj.q_wage / 2 = proj.q_wage + 9 := by sorry
  have h2 : (proj.q_wage + proj.q_wage / 2) * proj.p_hours = proj.q_wage * (proj.p_hours + 10) := by sorry
  have h3 : proj.total_payment = (proj.q_wage + proj.q_wage / 2) * proj.p_hours := by sorry
  sorry

/-- Main theorem proving the project payment is $540 -/
theorem main_project_payment : ∃ (proj : Project), proj.total_payment = 540 :=
  by
  sorry

end project_payment_main_project_payment_l3749_374920


namespace popped_kernel_probability_l3749_374957

theorem popped_kernel_probability (p_white p_yellow p_red : ℝ)
  (pop_white pop_yellow pop_red : ℝ) :
  p_white = 1/2 →
  p_yellow = 1/3 →
  p_red = 1/6 →
  pop_white = 1/2 →
  pop_yellow = 2/3 →
  pop_red = 1/3 →
  (p_white * pop_white) / (p_white * pop_white + p_yellow * pop_yellow + p_red * pop_red) = 9/19 := by
sorry

end popped_kernel_probability_l3749_374957


namespace farey_consecutive_fraction_l3749_374961

/-- Represents a fraction as a pair of integers -/
structure Fraction where
  numerator : ℤ
  denominator : ℤ
  den_nonzero : denominator ≠ 0

/-- Checks if three fractions are consecutive in a Farey sequence -/
def consecutive_in_farey (f1 f2 f3 : Fraction) : Prop :=
  f1.numerator * f2.denominator - f1.denominator * f2.numerator = 1 ∧
  f3.numerator * f2.denominator - f3.denominator * f2.numerator = 1

/-- The main theorem about three consecutive fractions in a Farey sequence -/
theorem farey_consecutive_fraction (a b c d x y : ℤ) 
  (hb : b ≠ 0) (hd : d ≠ 0) (hy : y ≠ 0)
  (h_order : (a : ℚ) / b < x / y ∧ x / y < c / d)
  (h_consecutive : consecutive_in_farey 
    ⟨a, b, hb⟩ 
    ⟨x, y, hy⟩ 
    ⟨c, d, hd⟩) :
  (x : ℚ) / y = (a + c) / (b + d) := by
  sorry

end farey_consecutive_fraction_l3749_374961


namespace smallest_K_for_inequality_l3749_374956

theorem smallest_K_for_inequality : 
  ∃ (K : ℝ), K = Real.sqrt 6 / 3 ∧ 
  (∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 → 
    K + (a + b + c) / 3 ≥ (K + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3)) ∧
  (∀ (K' : ℝ), K' > 0 ∧ K' < K → 
    ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧
      K' + (a + b + c) / 3 < (K' + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3)) :=
by sorry

end smallest_K_for_inequality_l3749_374956


namespace constant_for_max_n_l3749_374915

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, c * n^2 ≤ 6400 → n ≤ 7) ∧ 
  (∃ n : ℤ, c * n^2 ≤ 6400 ∧ n = 7) →
  c = 6400 / 49 :=
sorry

end constant_for_max_n_l3749_374915


namespace remainder_2023_div_73_l3749_374923

theorem remainder_2023_div_73 : 2023 % 73 = 52 := by
  sorry

end remainder_2023_div_73_l3749_374923


namespace total_length_of_items_l3749_374983

theorem total_length_of_items (rubber pen pencil : ℝ) 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) :
  rubber + pen + pencil = 29 := by
sorry

end total_length_of_items_l3749_374983


namespace max_value_theorem_l3749_374991

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 := by
  sorry

end max_value_theorem_l3749_374991


namespace product_with_9999_l3749_374994

theorem product_with_9999 : ∃ x : ℝ, x * 9999 = 4690910862 ∧ x = 469.1 := by
  sorry

end product_with_9999_l3749_374994


namespace tan_ratio_given_sin_equality_l3749_374908

theorem tan_ratio_given_sin_equality (α : ℝ) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (π / 180))) : 
  Real.tan (α + π / 180) / Real.tan (α - π / 180) = -3/2 := by
  sorry

end tan_ratio_given_sin_equality_l3749_374908


namespace balance_spheres_l3749_374980

/-- Represents the density of a material -/
structure Density where
  value : ℝ
  positive : value > 0

/-- Represents the volume of a sphere -/
structure Volume where
  value : ℝ
  positive : value > 0

/-- Represents the mass of a sphere -/
structure Mass where
  value : ℝ
  positive : value > 0

/-- Represents a sphere with its properties -/
structure Sphere where
  density : Density
  volume : Volume
  mass : Mass

/-- Theorem: Balance of two spheres in air -/
theorem balance_spheres (cast_iron wood : Sphere) (air_density : Density) : 
  cast_iron.density.value > wood.density.value →
  cast_iron.volume.value < wood.volume.value →
  cast_iron.mass.value < wood.mass.value →
  (cast_iron.density.value - air_density.value) * cast_iron.volume.value = 
  (wood.density.value - air_density.value) * wood.volume.value →
  ∃ (fulcrum_position : ℝ), 
    fulcrum_position > 0 ∧ 
    fulcrum_position < 1 ∧ 
    fulcrum_position * cast_iron.mass.value = (1 - fulcrum_position) * wood.mass.value :=
by
  sorry

end balance_spheres_l3749_374980


namespace two_number_problem_l3749_374929

def is_solution (x y : ℕ) : Prop :=
  (x + y = 667) ∧ 
  (Nat.lcm x y / Nat.gcd x y = 120)

theorem two_number_problem :
  ∀ x y : ℕ, is_solution x y → 
    ((x = 115 ∧ y = 552) ∨ (x = 552 ∧ y = 115) ∨ 
     (x = 232 ∧ y = 435) ∨ (x = 435 ∧ y = 232)) :=
by sorry

end two_number_problem_l3749_374929


namespace roses_in_vase_l3749_374951

/-- The total number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: The total number of roses is 22 when there were initially 6 roses and 16 were added -/
theorem roses_in_vase : total_roses 6 16 = 22 := by
  sorry

end roses_in_vase_l3749_374951


namespace dragon_tower_theorem_l3749_374924

/-- Represents the configuration of a dragon tethered to a cylindrical tower. -/
structure DragonTower where
  towerRadius : ℝ
  ropeLength : ℝ
  dragonHeight : ℝ
  ropeTowerDistance : ℝ

/-- Represents the parameters of the rope touching the tower. -/
structure RopeParameters where
  p : ℕ
  q : ℕ
  r : ℕ

/-- Theorem stating the relationship between the dragon-tower configuration
    and the rope parameters. -/
theorem dragon_tower_theorem (dt : DragonTower) (rp : RopeParameters) :
  dt.towerRadius = 10 ∧
  dt.ropeLength = 30 ∧
  dt.dragonHeight = 6 ∧
  dt.ropeTowerDistance = 6 ∧
  Nat.Prime rp.r ∧
  (rp.p - Real.sqrt rp.q) / rp.r = Real.sqrt ((dt.ropeLength - dt.ropeTowerDistance)^2 - dt.towerRadius^2) -
    (dt.ropeLength * Real.sqrt (dt.towerRadius^2 + dt.dragonHeight^2)) / dt.towerRadius +
    dt.dragonHeight * Real.sqrt (dt.towerRadius^2 + dt.dragonHeight^2) / dt.towerRadius →
  rp.p + rp.q + rp.r = 993 :=
by sorry

end dragon_tower_theorem_l3749_374924


namespace arithmetic_geometric_sequence_ratio_l3749_374976

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1)^2) : 
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13/16 := by
  sorry

end arithmetic_geometric_sequence_ratio_l3749_374976


namespace evaluate_expression_l3749_374967

theorem evaluate_expression : -(20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 := by
  sorry

end evaluate_expression_l3749_374967


namespace total_squares_is_86_l3749_374972

/-- The number of squares of a given size in a 6x6 grid -/
def count_squares (size : Nat) : Nat :=
  (7 - size) ^ 2

/-- The total number of squares of sizes 1x1, 2x2, 3x3, and 4x4 in a 6x6 grid -/
def total_squares : Nat :=
  count_squares 1 + count_squares 2 + count_squares 3 + count_squares 4

theorem total_squares_is_86 : total_squares = 86 := by
  sorry

end total_squares_is_86_l3749_374972


namespace distance_origin_to_line_l3749_374933

/-- The distance from the origin to a line passing through a given point with a given direction vector -/
theorem distance_origin_to_line (P : ℝ × ℝ) (n : ℝ × ℝ) : 
  P.1 = 2 ∧ P.2 = 0 ∧ n.1 = 1 ∧ n.2 = -1 →
  Real.sqrt ((P.1^2 + P.2^2) * (n.1^2 + n.2^2) - (P.1*n.1 + P.2*n.2)^2) / Real.sqrt (n.1^2 + n.2^2) = Real.sqrt 2 := by
sorry

end distance_origin_to_line_l3749_374933


namespace room_length_proof_l3749_374963

/-- Given the cost of carpeting, carpet width, cost per meter, and room breadth, 
    prove the length of the room. -/
theorem room_length_proof 
  (total_cost : ℝ) 
  (carpet_width : ℝ) 
  (cost_per_meter : ℝ) 
  (room_breadth : ℝ) 
  (h1 : total_cost = 36)
  (h2 : carpet_width = 0.75)
  (h3 : cost_per_meter = 0.30)
  (h4 : room_breadth = 6) :
  ∃ (room_length : ℝ), room_length = 15 := by
  sorry

end room_length_proof_l3749_374963


namespace unique_solution_l3749_374909

/-- The inequality condition for positive real numbers a, b, c, d, and real number x -/
def inequality_condition (a b c d x : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  ((a^3 / (a^3 + 15*b*c*d))^(1/2) : ℝ) ≥ (a^x / (a^x + b^x + c^x + d^x) : ℝ)

/-- The theorem stating that 15/8 is the only solution -/
theorem unique_solution :
  ∀ x : ℝ, (∀ a b c d : ℝ, inequality_condition a b c d x) ↔ x = 15/8 :=
sorry

end unique_solution_l3749_374909


namespace monic_quartic_value_l3749_374982

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value (p : ℝ → ℝ) :
  is_monic_quartic p →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 5 = 40 := by
sorry

end monic_quartic_value_l3749_374982


namespace smallest_valid_number_l3749_374977

def is_valid_number (n : ℕ) : Prop :=
  (n % 10 = 6) ∧ 
  (∃ m : ℕ, m > 0 ∧ 6 * 10^m + n / 10 = 4 * n)

theorem smallest_valid_number : 
  (is_valid_number 1538466) ∧ 
  (∀ k < 1538466, ¬(is_valid_number k)) := by sorry

end smallest_valid_number_l3749_374977


namespace cafeteria_earnings_l3749_374902

/-- Calculates the total earnings of a cafeteria from selling fruits --/
theorem cafeteria_earnings
  (initial_apples initial_oranges initial_bananas : ℕ)
  (remaining_apples remaining_oranges remaining_bananas : ℕ)
  (apple_cost orange_cost banana_cost : ℚ) :
  initial_apples = 80 →
  initial_oranges = 60 →
  initial_bananas = 40 →
  remaining_apples = 25 →
  remaining_oranges = 15 →
  remaining_bananas = 5 →
  apple_cost = 1.20 →
  orange_cost = 0.75 →
  banana_cost = 0.55 →
  (initial_apples - remaining_apples) * apple_cost +
  (initial_oranges - remaining_oranges) * orange_cost +
  (initial_bananas - remaining_bananas) * banana_cost = 119 :=
by sorry

end cafeteria_earnings_l3749_374902


namespace sugar_amount_in_new_recipe_l3749_374970

/-- Represents the ratio of ingredients in a recipe -/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio -/
def original_ratio : Ratio := ⟨7, 2, 1⟩

/-- The new recipe ratio -/
def new_ratio : Ratio :=
  let flour_water_doubled := original_ratio.flour / original_ratio.water * 2
  let flour_sugar_halved := original_ratio.flour / original_ratio.sugar / 2
  ⟨flour_water_doubled * original_ratio.water, original_ratio.water, flour_sugar_halved⟩

/-- The amount of water in the new recipe (in cups) -/
def new_water_amount : ℚ := 2

theorem sugar_amount_in_new_recipe :
  (new_water_amount * new_ratio.sugar / new_ratio.water) = 1 :=
sorry

end sugar_amount_in_new_recipe_l3749_374970


namespace newberg_total_landed_l3749_374981

/-- Represents the passenger data for an airport -/
structure AirportData where
  onTime : ℕ
  late : ℕ
  cancelled : ℕ

/-- Calculates the total number of landed passengers, excluding cancelled flights -/
def totalLanded (data : AirportData) : ℕ :=
  data.onTime + data.late

/-- Theorem: The total number of passengers who landed in Newberg last year is 28,690 -/
theorem newberg_total_landed :
  let airportA : AirportData := ⟨16507, 256, 198⟩
  let airportB : AirportData := ⟨11792, 135, 151⟩
  totalLanded airportA + totalLanded airportB = 28690 := by
  sorry


end newberg_total_landed_l3749_374981


namespace courier_distance_l3749_374944

/-- The total distance from A to B -/
def total_distance : ℝ := 412.5

/-- The additional distance traveled -/
def additional_distance : ℝ := 60

/-- The ratio of distance covered to remaining distance at the first point -/
def initial_ratio : ℚ := 2/3

/-- The ratio of distance covered to remaining distance after traveling the additional distance -/
def final_ratio : ℚ := 6/5

theorem courier_distance :
  ∃ (x : ℝ),
    (2 * x) / (3 * x) = initial_ratio ∧
    (2 * x + additional_distance) / (3 * x - additional_distance) = final_ratio ∧
    5 * x = total_distance :=
by sorry

end courier_distance_l3749_374944


namespace simplify_fraction_l3749_374978

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l3749_374978


namespace complex_ratio_l3749_374934

theorem complex_ratio (z : ℂ) (a b : ℝ) (h1 : z = Complex.mk a b) (h2 : z * (1 - Complex.I) = Complex.I) :
  a / b = -1 := by sorry

end complex_ratio_l3749_374934


namespace subset_implies_m_range_l3749_374935

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 3}

-- Define the range of m
def m_range : Set ℝ := {m | m < -4 ∨ m > 2}

-- Theorem statement
theorem subset_implies_m_range :
  ∀ m : ℝ, B m ⊆ A → m ∈ m_range :=
by sorry

end subset_implies_m_range_l3749_374935


namespace unique_fraction_decomposition_l3749_374931

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ 2 / p = 1 / n + 1 / m ∧ n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end unique_fraction_decomposition_l3749_374931


namespace marble_probability_l3749_374998

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  total = 84 →
  p_white = 1/4 →
  p_green = 1/7 →
  1 - (p_white + p_green) = 17/28 := by
  sorry

end marble_probability_l3749_374998


namespace tree_distance_l3749_374922

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, and a sign 30 feet beyond the last tree,
    the total distance between the first tree and the sign is 205 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (s : ℝ) : 
  n = 8 → d = 100 → s = 30 → 
  (n - 1) * (d / 4) + s = 205 :=
by sorry

end tree_distance_l3749_374922


namespace complement_of_A_l3749_374958

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 + 2*x ≥ 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end complement_of_A_l3749_374958


namespace hyperbola_equation_l3749_374990

/-- Given a hyperbola C and an ellipse with the following properties:
    1. C has the form x²/a² - y²/b² = 1 where a > 0 and b > 0
    2. C has an asymptote with equation y = (√5/2)x
    3. C shares a common focus with the ellipse x²/12 + y²/3 = 1
    Then the equation of C is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), y = (Real.sqrt 5 / 2) * x) ∧
  (∃ (c : ℝ), c^2 = 3^2 ∧ c^2 = a^2 + b^2) →
  a^2 = 4 ∧ b^2 = 5 :=
sorry

end hyperbola_equation_l3749_374990


namespace hyperbola_asymptote_l3749_374966

theorem hyperbola_asymptote (m : ℝ) :
  (∀ x y : ℝ, x^2 / |m| - y^2 / (|m| + 3) = 1) →
  (2 * Real.sqrt 5 = Real.sqrt (2 * |m| + 3)) →
  (∃ k : ℝ, k = 2 ∧ ∀ x : ℝ, k * x = 2 * x) :=
by sorry

end hyperbola_asymptote_l3749_374966


namespace equation_solution_l3749_374962

theorem equation_solution :
  let f (x : ℝ) := Real.sqrt (7*x - 3) + Real.sqrt (2*x - 2)
  ∃ (x : ℝ), (f x = 3 ↔ (x = 2 ∨ x = 172/25)) :=
by sorry

end equation_solution_l3749_374962


namespace line_direction_vector_c_l3749_374926

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (c : ℝ) : Prop :=
  let direction := (p2.1 - p1.1, p2.2 - p1.2)
  direction.1 = 3 ∧ direction.2 = c

/-- Theorem stating that for a line passing through (-6, 1) and (-3, 4) with direction vector (3, c), c must equal 3 -/
theorem line_direction_vector_c (c : ℝ) :
  Line (-6, 1) (-3, 4) c → c = 3 := by
  sorry


end line_direction_vector_c_l3749_374926


namespace smallest_k_no_real_roots_l3749_374986

theorem smallest_k_no_real_roots :
  ∃ (k : ℤ),
    (∀ (j : ℤ), j < k → ∃ (x : ℝ), 3 * x * (j * x - 5) - 2 * x^2 + 9 = 0) ∧
    (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 9 ≠ 0) ∧
    k = 3 := by
  sorry

end smallest_k_no_real_roots_l3749_374986


namespace manoj_lending_problem_l3749_374968

/-- Calculates simple interest -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem manoj_lending_problem (borrowed : ℚ) (borrowRate : ℚ) (lendRate : ℚ) (time : ℚ) (totalGain : ℚ)
  (h1 : borrowed = 3900)
  (h2 : borrowRate = 6)
  (h3 : lendRate = 9)
  (h4 : time = 3)
  (h5 : totalGain = 824.85)
  : ∃ (lentSum : ℚ), 
    lentSum = 5655 ∧ 
    simpleInterest lentSum lendRate time - simpleInterest borrowed borrowRate time = totalGain :=
sorry

end manoj_lending_problem_l3749_374968


namespace greatest_prime_factor_f_24_l3749_374919

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2)) (fun i => 2 * (i + 1))

theorem greatest_prime_factor_f_24 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ f 24 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ f 24 → q ≤ p :=
by sorry

end greatest_prime_factor_f_24_l3749_374919


namespace tangent_equation_solution_l3749_374940

open Real

theorem tangent_equation_solution (x : ℝ) :
  tan x + tan (50 * π / 180) + tan (70 * π / 180) = tan x * tan (50 * π / 180) * tan (70 * π / 180) →
  ∃ n : ℤ, x = (60 + 180 * n) * π / 180 :=
by sorry

end tangent_equation_solution_l3749_374940


namespace odd_sequence_sum_l3749_374917

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n / 2 * (a₁ + aₙ)

theorem odd_sequence_sum :
  ∃ (n : ℕ), 
    let a₁ := 1
    let aₙ := 79
    let sum := arithmetic_sum a₁ aₙ n
    n > 0 ∧ aₙ = a₁ + 2 * (n - 1) ∧ 3 * sum = 4800 := by
  sorry

end odd_sequence_sum_l3749_374917


namespace rectangle_formations_6_7_l3749_374936

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of ways to form a rectangle given h horizontal lines and v vertical lines -/
def rectangle_formations (h v : ℕ) : ℕ := choose_2 h * choose_2 v

/-- Theorem stating that with 6 horizontal and 7 vertical lines, there are 315 ways to form a rectangle -/
theorem rectangle_formations_6_7 : rectangle_formations 6 7 = 315 := by sorry

end rectangle_formations_6_7_l3749_374936


namespace painted_cube_theorem_l3749_374949

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : Fin 6 → Bool

/-- Counts the number of unit cubes with at least two painted faces in a painted cube -/
def count_multi_painted_cubes (c : Cube 4) : ℕ :=
  sorry

/-- The theorem stating that a 4x4x4 painted cube has 56 unit cubes with at least two painted faces -/
theorem painted_cube_theorem (c : Cube 4) 
  (h : ∀ (f : Fin 6), c.painted_faces f = true) : 
  count_multi_painted_cubes c = 56 := by
  sorry

end painted_cube_theorem_l3749_374949


namespace infinitely_many_invalid_d_l3749_374903

/-- The perimeter difference between the triangle and rectangle -/
def perimeter_difference : ℕ := 504

/-- The length of the shorter side of the rectangle -/
def rectangle_short_side : ℕ := 7

/-- Represents the relationship between the triangle side length, rectangle long side, and d -/
def triangle_rectangle_relation (triangle_side : ℝ) (rectangle_long_side : ℝ) (d : ℝ) : Prop :=
  triangle_side = rectangle_long_side + d

/-- Represents the perimeter relationship between the triangle and rectangle -/
def perimeter_relation (triangle_side : ℝ) (rectangle_long_side : ℝ) : Prop :=
  3 * triangle_side - 2 * (rectangle_long_side + rectangle_short_side) = perimeter_difference

/-- The main theorem stating that there are infinitely many positive integers
    that cannot be valid values for d -/
theorem infinitely_many_invalid_d : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (d : ℕ), d ∈ S →
    ¬∃ (triangle_side rectangle_long_side : ℝ),
      triangle_rectangle_relation triangle_side rectangle_long_side d ∧
      perimeter_relation triangle_side rectangle_long_side :=
sorry

end infinitely_many_invalid_d_l3749_374903


namespace sequences_properties_l3749_374947

def a (n : ℕ) : ℤ := (-3) ^ n
def b (n : ℕ) : ℤ := (-3) ^ n - 3
def c (n : ℕ) : ℤ := -(-3) ^ n - 1

def m (n : ℕ) : ℤ := a n + b n + c n

theorem sequences_properties :
  (a 5 = -243 ∧ b 5 = -246 ∧ c 5 = 242) ∧
  (∃ k : ℕ, a k + a (k + 1) + a (k + 2) = -1701) ∧
  (∀ n : ℕ,
    (n % 2 = 1 → max (a n) (max (b n) (c n)) - min (a n) (min (b n) (c n)) = -2 * m n - 6) ∧
    (n % 2 = 0 → max (a n) (max (b n) (c n)) - min (a n) (min (b n) (c n)) = 2 * m n + 9)) :=
by sorry

end sequences_properties_l3749_374947


namespace remaining_black_portion_l3749_374996

/-- The fraction of black area remaining after one transformation -/
def black_fraction : ℚ := 3 / 4

/-- The number of transformations applied -/
def num_transformations : ℕ := 5

/-- The theorem stating the remaining black portion after transformations -/
theorem remaining_black_portion :
  black_fraction ^ num_transformations = 243 / 1024 := by
  sorry

end remaining_black_portion_l3749_374996


namespace star_properties_l3749_374948

/-- Custom binary operation ※ -/
def star (x y : ℚ) : ℚ := x * y + 1

/-- Theorem stating the properties of the ※ operation -/
theorem star_properties :
  (star 2 4 = 9) ∧
  (star (star 1 4) (-2) = -9) ∧
  (∀ a b c : ℚ, star a (b + c) + 1 = star a b + star a c) :=
by sorry

end star_properties_l3749_374948


namespace inequality_solution_set_k_value_range_l3749_374910

-- Problem 1
theorem inequality_solution_set (x : ℝ) : 
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3/2 :=
sorry

-- Problem 2
theorem k_value_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) ↔ (k > Real.sqrt 2 ∨ k < -Real.sqrt 2) :=
sorry

end inequality_solution_set_k_value_range_l3749_374910


namespace surface_area_of_problem_solid_l3749_374913

/-- Represents an L-shaped solid formed by unit cubes -/
structure LShapedSolid where
  base_layer : ℕ
  top_layer : ℕ
  top_layer_start : ℕ

/-- Calculates the surface area of an L-shaped solid -/
def surface_area (solid : LShapedSolid) : ℕ :=
  let base_exposed := solid.base_layer - (solid.top_layer - (solid.top_layer_start - 1))
  let top_exposed := solid.top_layer
  let front_back := 2 * (solid.base_layer + solid.top_layer)
  let sides := 2 * 2
  let top_bottom := base_exposed + top_exposed + (solid.top_layer_start - 1)
  front_back + sides + top_bottom

/-- The specific L-shaped solid described in the problem -/
def problem_solid : LShapedSolid :=
  { base_layer := 8
  , top_layer := 6
  , top_layer_start := 5 }

theorem surface_area_of_problem_solid :
  surface_area problem_solid = 44 :=
sorry

end surface_area_of_problem_solid_l3749_374913


namespace basketball_league_games_l3749_374979

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team, 
    the total number of games is 180 -/
theorem basketball_league_games : total_games 10 4 = 180 := by
  sorry

end basketball_league_games_l3749_374979


namespace no_divisible_by_seven_l3749_374916

theorem no_divisible_by_seven : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2009 → ¬(7 ∣ (4 * n^6 + n^3 + 5)) := by
  sorry

end no_divisible_by_seven_l3749_374916


namespace complex_number_in_second_quadrant_l3749_374941

/-- Given a complex number z = (10-5ai)/(1-2i) where a is a real number,
    and the sum of its real and imaginary parts is 4,
    prove that its real part is negative and its imaginary part is positive. -/
theorem complex_number_in_second_quadrant (a : ℝ) :
  let z : ℂ := (10 - 5*a*Complex.I) / (1 - 2*Complex.I)
  (z.re + z.im = 4) →
  (z.re < 0 ∧ z.im > 0) :=
by sorry


end complex_number_in_second_quadrant_l3749_374941


namespace even_odd_function_sum_l3749_374995

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_function_sum (f g : ℝ → ℝ) 
  (hf : is_even_function f) (hg : is_odd_function g) 
  (h : ∀ x, f x + g x = Real.exp x) : 
  ∀ x, g x = Real.exp x - Real.exp (-x) := by
  sorry

end even_odd_function_sum_l3749_374995


namespace composite_prime_calculation_l3749_374946

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]
def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem composite_prime_calculation :
  (((first_six_composites.prod : ℚ) / (next_six_composites.prod : ℚ)) * (first_five_primes.prod : ℚ)) = 377.55102040816324 := by
  sorry

end composite_prime_calculation_l3749_374946


namespace geometric_sequence_sum_l3749_374914

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ+, a n > 0) →
  a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81 →
  a 4 + a 6 = 9 := by
sorry

end geometric_sequence_sum_l3749_374914


namespace group_frequency_l3749_374975

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) :
  sample_capacity = 80 →
  group_frequency = 0.125 →
  (sample_capacity : ℚ) * group_frequency = 10 := by
  sorry

end group_frequency_l3749_374975


namespace units_digit_17_39_l3749_374964

theorem units_digit_17_39 : (17^39) % 10 = 3 := by
  sorry

end units_digit_17_39_l3749_374964


namespace ninth_grade_class_problem_l3749_374900

theorem ninth_grade_class_problem (total : ℕ) (science : ℕ) (arts : ℕ) 
  (h_total : total = 120)
  (h_science : science = 85)
  (h_arts : arts = 65)
  (h_covers_all : total ≤ science + arts) :
  science - (science + arts - total) = 55 := by
  sorry

end ninth_grade_class_problem_l3749_374900


namespace marks_vaccine_wait_l3749_374993

/-- Theorem: Mark's wait for first vaccine appointment
Given:
- The total waiting time is 38 days
- There's a 20-day wait between appointments
- There's a 14-day wait for full effectiveness after the second appointment
Prove: The wait for the first appointment is 4 days
-/
theorem marks_vaccine_wait (total_wait : ℕ) (between_appointments : ℕ) (full_effectiveness : ℕ) :
  total_wait = 38 →
  between_appointments = 20 →
  full_effectiveness = 14 →
  total_wait = between_appointments + full_effectiveness + 4 :=
by sorry

end marks_vaccine_wait_l3749_374993


namespace cos_A_eq_11_15_l3749_374953

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_A_eq_C (q : Quadrilateral) : Prop :=
  sorry

def side_AB_eq_150 (q : Quadrilateral) : Prop :=
  sorry

def side_CD_eq_150 (q : Quadrilateral) : Prop :=
  sorry

def side_AD_ne_BC (q : Quadrilateral) : Prop :=
  sorry

def perimeter_eq_520 (q : Quadrilateral) : Prop :=
  sorry

-- Define cos A
def cos_A (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem cos_A_eq_11_15 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_A_eq_C q)
  (h_AB : side_AB_eq_150 q)
  (h_CD : side_CD_eq_150 q)
  (h_AD_ne_BC : side_AD_ne_BC q)
  (h_perimeter : perimeter_eq_520 q) :
  cos_A q = 11/15 := by sorry

end cos_A_eq_11_15_l3749_374953


namespace largest_common_term_of_arithmetic_progressions_l3749_374973

theorem largest_common_term_of_arithmetic_progressions :
  let seq1 (n : ℕ) := 4 + 5 * n
  let seq2 (m : ℕ) := 3 + 7 * m
  ∃ (n m : ℕ), seq1 n = seq2 m ∧ seq1 n = 299 ∧
  ∀ (k l : ℕ), seq1 k = seq2 l → seq1 k ≤ 299 :=
by sorry

end largest_common_term_of_arithmetic_progressions_l3749_374973


namespace triangle_max_perimeter_l3749_374927

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 4*x →
  x + 4*x > 20 →
  4*x + 20 > x →
  x + 4*x + 20 ≤ 50 :=
by sorry

end triangle_max_perimeter_l3749_374927


namespace inequality_system_sum_l3749_374937

theorem inequality_system_sum (a b : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x + 2*a > 4 ∧ 2*x < b)) → 
  a + b = 6 := by
  sorry

end inequality_system_sum_l3749_374937


namespace intersection_points_l3749_374918

theorem intersection_points (x : ℝ) : 
  (∃ y : ℝ, y = 10 / (x^2 + 1) ∧ x^2 + y = 3) ↔ 
  (x = Real.sqrt (1 + 2 * Real.sqrt 2) ∨ x = -Real.sqrt (1 + 2 * Real.sqrt 2)) :=
by sorry

end intersection_points_l3749_374918


namespace toothpicks_10th_stage_l3749_374984

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 5
  else toothpicks (n - 1) + 3 * n

/-- The theorem stating that the 10th stage has 167 toothpicks -/
theorem toothpicks_10th_stage : toothpicks 10 = 167 := by
  sorry

end toothpicks_10th_stage_l3749_374984


namespace unique_positive_solution_l3749_374959

theorem unique_positive_solution : ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The unique positive solution is 5/3
  use 5/3
  constructor
  · -- Prove that 5/3 satisfies the conditions
    constructor
    · -- Prove 5/3 > 0
      sorry
    · -- Prove 3 * (5/3)^2 + 7 * (5/3) - 20 = 0
      sorry
  · -- Prove uniqueness
    sorry

end unique_positive_solution_l3749_374959


namespace root_difference_quadratic_equation_l3749_374912

theorem root_difference_quadratic_equation : 
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 5.5 := by
sorry

end root_difference_quadratic_equation_l3749_374912


namespace nonnegative_solutions_count_l3749_374960

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end nonnegative_solutions_count_l3749_374960


namespace mothers_day_discount_l3749_374907

def original_price : ℝ := 125
def mother_discount : ℝ := 0.1
def additional_discount : ℝ := 0.04
def children_count : ℕ := 4

theorem mothers_day_discount (price : ℝ) (md : ℝ) (ad : ℝ) (cc : ℕ) :
  price > 0 →
  md > 0 →
  ad > 0 →
  cc ≥ 3 →
  price * (1 - md) * (1 - ad) = 108 := by
sorry

end mothers_day_discount_l3749_374907


namespace natalia_albums_count_l3749_374974

/-- Represents the number of items in Natalia's library --/
structure LibraryItems where
  novels : Nat
  comics : Nat
  documentaries : Nat
  albums : Nat

/-- Represents the crate information --/
structure CrateInfo where
  capacity : Nat
  count : Nat

/-- Theorem: Given the library items and crate information, prove that Natalia has 209 albums --/
theorem natalia_albums_count
  (items : LibraryItems)
  (crates : CrateInfo)
  (h1 : items.novels = 145)
  (h2 : items.comics = 271)
  (h3 : items.documentaries = 419)
  (h4 : crates.capacity = 9)
  (h5 : crates.count = 116)
  (h6 : items.novels + items.comics + items.documentaries + items.albums = crates.capacity * crates.count) :
  items.albums = 209 := by
  sorry


end natalia_albums_count_l3749_374974


namespace solve_for_k_l3749_374932

theorem solve_for_k (x y k : ℝ) : 
  x = 2 → 
  y = 1 → 
  k * x - y = 3 → 
  k = 2 := by
sorry

end solve_for_k_l3749_374932
