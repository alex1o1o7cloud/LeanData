import Mathlib

namespace pure_imaginary_m_l2717_271712

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_m (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 1) (m^2 + 2*m - 3)
  is_pure_imaginary z → m = -1 := by sorry

end pure_imaginary_m_l2717_271712


namespace school_supplies_cost_l2717_271754

/-- The cost of all pencils and pens given their individual prices and quantities -/
def total_cost (pencil_price pen_price : ℚ) (num_pencils num_pens : ℕ) : ℚ :=
  pencil_price * num_pencils + pen_price * num_pens

/-- Theorem stating the total cost of 38 pencils at $2.50 each and 56 pens at $3.50 each is $291.00 -/
theorem school_supplies_cost :
  total_cost (5/2) (7/2) 38 56 = 291 := by
  sorry

end school_supplies_cost_l2717_271754


namespace walking_speed_problem_l2717_271738

/-- Proves that given the conditions of the problem, A's walking speed is 10 kmph -/
theorem walking_speed_problem (v : ℝ) : 
  v > 0 → -- A's walking speed is positive
  v * (200 / v) = 20 * (200 / v - 10) → -- Distance equation
  v = 10 := by
  sorry

end walking_speed_problem_l2717_271738


namespace inequalities_given_m_gt_neg_one_l2717_271718

theorem inequalities_given_m_gt_neg_one (m : ℝ) (h : m > -1) :
  (4*m > -4) ∧ (-5*m < -5) ∧ (m+1 > 0) ∧ (1-m < 2) := by
  sorry

end inequalities_given_m_gt_neg_one_l2717_271718


namespace expression_simplification_l2717_271761

theorem expression_simplification (a : ℝ) (h : a = 3) : 
  (a^2 / (a + 1)) - (1 / (a + 1)) = 2 := by
  sorry

end expression_simplification_l2717_271761


namespace expansion_equality_l2717_271729

-- Define the left-hand side of the equation
def lhs (x : ℝ) : ℝ := (5 * x^2 + 3 * x - 7) * 4 * x^3

-- Define the right-hand side of the equation
def rhs (x : ℝ) : ℝ := 20 * x^5 + 12 * x^4 - 28 * x^3

-- State the theorem
theorem expansion_equality : ∀ x : ℝ, lhs x = rhs x := by sorry

end expansion_equality_l2717_271729


namespace class_size_l2717_271755

theorem class_size (total : ℕ) (girls : ℕ) (boys : ℕ) :
  girls = total * 52 / 100 →
  girls = boys + 1 →
  total = girls + boys →
  total = 25 :=
by sorry

end class_size_l2717_271755


namespace beast_of_war_runtime_l2717_271737

/-- The running time of Millennium in hours -/
def millennium_runtime : ℝ := 2

/-- The difference in minutes between Millennium and Alpha Epsilon runtimes -/
def alpha_epsilon_diff : ℝ := 30

/-- The difference in minutes between Beast of War and Alpha Epsilon runtimes -/
def beast_of_war_diff : ℝ := 10

/-- Conversion factor from hours to minutes -/
def hours_to_minutes : ℝ := 60

/-- Theorem stating the runtime of Beast of War: Armoured Command -/
theorem beast_of_war_runtime : 
  millennium_runtime * hours_to_minutes - alpha_epsilon_diff + beast_of_war_diff = 100 := by
sorry

end beast_of_war_runtime_l2717_271737


namespace stamp_difference_l2717_271767

theorem stamp_difference (p q : ℕ) (h1 : p * 4 = q * 7) 
  (h2 : (p - 8) * 5 = (q + 8) * 6) : p - q = 8 := by
  sorry

end stamp_difference_l2717_271767


namespace phone_plan_monthly_fee_l2717_271777

theorem phone_plan_monthly_fee :
  let first_plan_per_minute : ℚ := 13/100
  let second_plan_monthly_fee : ℚ := 8
  let second_plan_per_minute : ℚ := 18/100
  let equal_minutes : ℕ := 280
  ∃ (F : ℚ),
    F + first_plan_per_minute * equal_minutes = 
    second_plan_monthly_fee + second_plan_per_minute * equal_minutes ∧
    F = 22 := by
  sorry

end phone_plan_monthly_fee_l2717_271777


namespace shirt_cost_to_marked_price_ratio_l2717_271742

/-- Given a shop with shirts on sale, this theorem proves the ratio of cost to marked price. -/
theorem shirt_cost_to_marked_price_ratio :
  ∀ (marked_price : ℝ), marked_price > 0 →
  let discount_rate : ℝ := 0.25
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 0.60
  let cost_price : ℝ := selling_price * cost_rate
  cost_price / marked_price = 0.45 := by
sorry


end shirt_cost_to_marked_price_ratio_l2717_271742


namespace x_squared_minus_y_squared_l2717_271726

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 22) : 
  x^2 - y^2 = -120 := by
  sorry

end x_squared_minus_y_squared_l2717_271726


namespace ice_cream_stacking_problem_l2717_271765

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The problem statement -/
theorem ice_cream_stacking_problem :
  permutations 5 = 120 := by sorry

end ice_cream_stacking_problem_l2717_271765


namespace f_has_two_zeros_l2717_271745

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - abs x - 6

-- Theorem stating that f has exactly two zeros
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end f_has_two_zeros_l2717_271745


namespace arithmetic_geometric_sequence_l2717_271746

theorem arithmetic_geometric_sequence (x : ℝ) : 
  (∃ y : ℝ, 
    -- y is between 3 and x
    3 < y ∧ y < x ∧
    -- arithmetic sequence condition
    (y - 3 = x - y) ∧
    -- geometric sequence condition after subtracting 6 from the middle term
    ((y - 6) / 3 = x / (y - 6))) →
  (x = 3 ∨ x = 27) :=
by sorry

end arithmetic_geometric_sequence_l2717_271746


namespace tan_domain_theorem_l2717_271724

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - π / 4)

def domain_set : Set ℝ := ⋃ k : ℤ, Ioo ((k : ℝ) * π / 2 - π / 8) ((k : ℝ) * π / 2 + 3 * π / 8)

theorem tan_domain_theorem :
  {x : ℝ | ∃ y, f x = y} = domain_set :=
sorry

end tan_domain_theorem_l2717_271724


namespace distribute_6_3_max2_l2717_271751

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) (max_per_box : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes
    with at most 2 balls per box -/
theorem distribute_6_3_max2 : distribute 6 3 2 = 100 := by sorry

end distribute_6_3_max2_l2717_271751


namespace inequality_system_solution_set_l2717_271769

theorem inequality_system_solution_set :
  ∀ x : ℝ, (3/2 * x + 5 ≤ -1 ∧ x + 3 < 0) ↔ x ≤ -4 := by
sorry

end inequality_system_solution_set_l2717_271769


namespace ellipse_eccentricity_l2717_271706

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  P : ℝ × ℝ
  h_on_ellipse : (P.1^2 / E.a^2) + (P.2^2 / E.b^2) = 1

/-- The foci of the ellipse -/
def foci (E : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (E : Ellipse) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity (E : Ellipse) (P : PointOnEllipse E) :
  let (F₁, F₂) := foci E
  dot_product (P.P.1 - F₁.1, P.P.2 - F₁.2) (P.P.1 - F₂.1, P.P.2 - F₂.2) = 0 →
  Real.tan (angle (P.P.1 - F₁.1, P.P.2 - F₁.2) (F₂.1 - F₁.1, F₂.2 - F₁.2)) = 1/2 →
  eccentricity E = Real.sqrt 5 / 3 := by
  sorry

end ellipse_eccentricity_l2717_271706


namespace negation_of_implication_l2717_271725

theorem negation_of_implication (a b : ℝ) :
  ¬(a = 0 → a * b = 0) ↔ (a ≠ 0 → a * b ≠ 0) := by sorry

end negation_of_implication_l2717_271725


namespace jack_email_difference_l2717_271796

theorem jack_email_difference : 
  let morning_emails : ℕ := 6
  let afternoon_emails : ℕ := 2
  morning_emails - afternoon_emails = 4 :=
by sorry

end jack_email_difference_l2717_271796


namespace lcm_gcd_sum_implies_divisibility_l2717_271752

theorem lcm_gcd_sum_implies_divisibility (m n : ℕ) :
  Nat.lcm m n + Nat.gcd m n = m + n → m ∣ n ∨ n ∣ m := by
  sorry

end lcm_gcd_sum_implies_divisibility_l2717_271752


namespace triangle_height_l2717_271748

theorem triangle_height (a b c : ℝ) (h_sum : a + c = 11) 
  (h_angle : Real.cos (π / 3) = 1 / 2) 
  (h_radius : (a * b * Real.sin (π / 3)) / (a + b + c) = 2 / Real.sqrt 3) 
  (h_longer : a > c) : 
  c * Real.sin (π / 3) = 4 * Real.sqrt 3 := by
sorry

end triangle_height_l2717_271748


namespace ratio_problem_l2717_271793

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) :
  second_part = 20 →
  percent = 25 →
  first_part / second_part = percent / 100 →
  first_part = 5 := by
sorry

end ratio_problem_l2717_271793


namespace geometric_sequence_special_ratio_l2717_271735

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the problem statement
theorem geometric_sequence_special_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a q)
  (h_arith : a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) :
  q = (Real.sqrt 5 + 1) / 2 :=
sorry

end geometric_sequence_special_ratio_l2717_271735


namespace final_score_for_five_hours_l2717_271739

/-- Represents a student's test performance -/
structure TestPerformance where
  maxPoints : ℝ
  preparationTime : ℝ
  score : ℝ
  effortBonus : ℝ

/-- Calculates the final score given a TestPerformance -/
def finalScore (tp : TestPerformance) : ℝ :=
  tp.score * (1 + tp.effortBonus)

/-- Theorem stating the final score for 5 hours of preparation -/
theorem final_score_for_five_hours 
  (tp : TestPerformance)
  (h1 : tp.maxPoints = 150)
  (h2 : tp.preparationTime = 5)
  (h3 : tp.effortBonus = 0.1)
  (h4 : ∃ (t : TestPerformance), t.preparationTime = 2 ∧ t.score = 90 ∧ 
        tp.score / tp.preparationTime = t.score / t.preparationTime) :
  finalScore tp = 247.5 := by
sorry


end final_score_for_five_hours_l2717_271739


namespace inscribed_polygon_perimeter_l2717_271790

/-- 
Given two regular polygons with perimeters a and b circumscribed around a circle, 
and a third regular polygon inscribed in the same circle, where the second and 
third polygons have twice as many sides as the first, the perimeter of the third 
polygon is equal to b√(a / (2a - b)).
-/
theorem inscribed_polygon_perimeter 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_circumscribed : ∃ (r : ℝ) (n : ℕ), a = 2 * n * r * Real.tan (2 * π / n) ∧ 
                                        b = 4 * n * r * Real.tan (π / n))
  (h_inscribed : ∃ (x : ℝ), x = b * Real.cos (π / n)) :
  ∃ (x : ℝ), x = b * Real.sqrt (a / (2 * a - b)) :=
sorry

end inscribed_polygon_perimeter_l2717_271790


namespace square_root_inequalities_l2717_271773

theorem square_root_inequalities : 
  (∃ (x y : ℝ), x = Real.sqrt 7 ∧ y = Real.sqrt 3 ∧ x + y ≠ Real.sqrt 10) ∧
  (Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15) ∧
  (Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2) ∧
  ((-Real.sqrt 3)^2 = 3) := by
  sorry


end square_root_inequalities_l2717_271773


namespace symmetric_point_y_axis_l2717_271730

def point_symmetry_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_y_axis :
  let P : ℝ × ℝ := (2, 3)
  point_symmetry_y_axis P = (-2, 3) := by sorry

end symmetric_point_y_axis_l2717_271730


namespace knights_in_company_l2717_271794

def is_knight (person : Nat) : Prop := sorry

def statement (n : Nat) : Prop :=
  ∃ k, k ∣ n ∧ (∀ p, is_knight p ↔ p ≤ k)

theorem knights_in_company :
  ∀ k : Nat, k ≤ 39 →
  (∀ n : Nat, n ≤ 39 → (∃ p, is_knight p ↔ statement n)) →
  (k = 0 ∨ k = 6) :=
sorry

end knights_in_company_l2717_271794


namespace lexie_age_l2717_271791

/-- Represents the ages of Lexie, her brother, and her sister -/
structure Family where
  lexie : ℕ
  brother : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (f : Family) : Prop :=
  f.lexie = f.brother + 6 ∧
  f.sister = 2 * f.lexie ∧
  f.sister - f.brother = 14

/-- Theorem stating that if a family satisfies the given conditions, Lexie's age is 8 -/
theorem lexie_age (f : Family) (h : satisfiesConditions f) : f.lexie = 8 := by
  sorry

#check lexie_age

end lexie_age_l2717_271791


namespace hyperbola_eccentricity_l2717_271719

/-- Given a hyperbola and a circle satisfying certain conditions, prove that the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  ((c - a)^2 = c^2 / 16) →  -- Circle passes through right focus F(c, 0)
  (∃ k : ℝ, ∀ x y : ℝ, 
    (x - a)^2 + y^2 = c^2 / 16 →  -- Circle equation
    (y = k * x ∨ y = -k * x) →  -- Asymptote equations
    ∃ m : ℝ, m * k = -1 ∧ 
      ∃ x₀ y₀ : ℝ, (x₀ - a)^2 + y₀^2 = c^2 / 16 ∧ 
        y₀ - 0 = m * (x₀ - c)) →  -- Tangent line perpendicular to asymptote
  c / a = 2  -- Eccentricity is 2
:= by sorry

end hyperbola_eccentricity_l2717_271719


namespace common_solution_l2717_271763

def m_values : List ℤ := [-5, -4, -3, -1, 0, 1, 3, 23, 124, 1000]

def equation (m x y : ℤ) : Prop :=
  (2 * m + 1) * x + (2 - 3 * m) * y + 1 - 5 * m = 0

theorem common_solution :
  ∀ m ∈ m_values, equation m 1 (-1) :=
by sorry

end common_solution_l2717_271763


namespace water_bucket_problem_l2717_271711

theorem water_bucket_problem (total_grams : ℕ) : 
  (total_grams % 900 = 200) ∧ 
  (total_grams / 900 = 7) → 
  (total_grams : ℚ) / 1000 = 6.5 := by
sorry

end water_bucket_problem_l2717_271711


namespace sequence_inequality_l2717_271778

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

/-- The main theorem -/
theorem sequence_inequality (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (heq : a 11 = b 10) : 
  a 13 + a 9 ≤ b 14 + b 6 :=
sorry

end sequence_inequality_l2717_271778


namespace maintenance_check_interval_l2717_271736

theorem maintenance_check_interval (original_interval : ℝ) : 
  (original_interval * 1.2 = 60) → original_interval = 50 := by
  sorry

end maintenance_check_interval_l2717_271736


namespace sleep_increase_l2717_271783

theorem sleep_increase (initial_sleep : ℝ) (increase_factor : ℝ) (final_sleep : ℝ) :
  initial_sleep = 6 →
  increase_factor = 1/3 →
  final_sleep = initial_sleep + increase_factor * initial_sleep →
  final_sleep = 8 := by
sorry

end sleep_increase_l2717_271783


namespace min_value_ab_l2717_271710

theorem min_value_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b)
  (h4 : ∀ x y, 0 < x → x < y → a^x + b^x < a^y + b^y) :
  1 ≤ a * b :=
sorry

end min_value_ab_l2717_271710


namespace angle_phi_value_l2717_271740

theorem angle_phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : Real.sqrt 3 * Real.sin (20 * Real.pi / 180) = Real.cos φ - Real.sin φ) : 
  φ = 25 * Real.pi / 180 := by
sorry

end angle_phi_value_l2717_271740


namespace inequality_proof_l2717_271705

theorem inequality_proof (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_one : a + b + c + d = 1) :
  a * b * c + b * c * d + c * a * d + d * a * b ≤ 1 / 27 + (176 / 27) * a * b * c * d :=
by sorry

end inequality_proof_l2717_271705


namespace last_hour_probability_l2717_271734

/-- The number of attractions available -/
def num_attractions : ℕ := 6

/-- The number of attractions each person chooses -/
def num_chosen : ℕ := 4

/-- The probability of two people being at the same attraction during their last hour -/
def same_attraction_probability : ℚ := 1 / 6

theorem last_hour_probability :
  (num_attractions : ℚ) / (num_attractions * num_attractions) = same_attraction_probability :=
sorry

end last_hour_probability_l2717_271734


namespace photo_lineup_arrangements_l2717_271758

/-- The number of boys in the lineup -/
def num_boys : ℕ := 4

/-- The number of girls in the lineup -/
def num_girls : ℕ := 3

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of arrangements when Boy A must stand at either end -/
def arrangements_boy_a_at_end : ℕ := 1440

/-- The number of arrangements when Girl B cannot stand to the left of Girl C -/
def arrangements_girl_b_not_left_of_c : ℕ := 2520

/-- The number of arrangements when Girl B does not stand at either end, and Girl C does not stand in the middle -/
def arrangements_girl_b_not_end_c_not_middle : ℕ := 3120

theorem photo_lineup_arrangements :
  (arrangements_boy_a_at_end = 1440) ∧
  (arrangements_girl_b_not_left_of_c = 2520) ∧
  (arrangements_girl_b_not_end_c_not_middle = 3120) := by
  sorry

end photo_lineup_arrangements_l2717_271758


namespace other_intersection_point_l2717_271704

def f (x k : ℝ) : ℝ := 3 * (x - 4)^2 + k

theorem other_intersection_point (k : ℝ) :
  f 2 k = 0 → f 6 k = 0 := by
  sorry

end other_intersection_point_l2717_271704


namespace f_equality_l2717_271728

noncomputable def f (x : ℝ) : ℝ := Real.arctan ((2 * x) / (1 - x^2))

theorem f_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : 3 - 4 * x^2 ≠ 0) :
  f ((x - 4 * x^3) / (3 - 4 * x^2)) = f x :=
by sorry

end f_equality_l2717_271728


namespace modulus_v_is_five_l2717_271759

/-- Given two complex numbers u and v, prove that |v| = 5 when uv = 15 - 20i and |u| = 5 -/
theorem modulus_v_is_five (u v : ℂ) (h1 : u * v = 15 - 20 * I) (h2 : Complex.abs u = 5) : 
  Complex.abs v = 5 := by
sorry

end modulus_v_is_five_l2717_271759


namespace seashells_sum_l2717_271701

/-- The number of seashells Mary found -/
def mary_shells : ℕ := 18

/-- The number of seashells Jessica found -/
def jessica_shells : ℕ := 41

/-- The total number of seashells found by Mary and Jessica -/
def total_shells : ℕ := mary_shells + jessica_shells

theorem seashells_sum : total_shells = 59 := by
  sorry

end seashells_sum_l2717_271701


namespace coin_equation_solution_l2717_271727

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of quarters on the left side of the equation -/
def left_quarters : ℕ := 30

/-- The number of dimes on the left side of the equation -/
def left_dimes : ℕ := 20

/-- The number of quarters on the right side of the equation -/
def right_quarters : ℕ := 5

theorem coin_equation_solution :
  ∃ n : ℕ, 
    left_quarters * quarter_value + left_dimes * dime_value = 
    right_quarters * quarter_value + n * dime_value ∧
    n = 83 := by
  sorry

end coin_equation_solution_l2717_271727


namespace inequality_proof_l2717_271715

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end inequality_proof_l2717_271715


namespace stratified_sampling_female_students_l2717_271797

/-- Calculates the number of female students selected in a stratified sampling -/
def female_students_selected (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * female_students) / total_students

/-- Theorem: In a school with 2000 total students and 800 female students,
    a stratified sampling of 50 students will select 20 female students -/
theorem stratified_sampling_female_students :
  female_students_selected 2000 800 50 = 20 := by
  sorry

end stratified_sampling_female_students_l2717_271797


namespace equation_may_not_hold_l2717_271708

theorem equation_may_not_hold (a b c : ℝ) : 
  a = b → ¬(∀ c, a / c = b / c) :=
by
  sorry

end equation_may_not_hold_l2717_271708


namespace football_lineup_count_l2717_271747

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose a starting lineup for the given football team -/
theorem football_lineup_count :
  choose_lineup 15 5 = 109200 := by
  sorry

end football_lineup_count_l2717_271747


namespace hyperbola_equation_l2717_271782

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Theorem statement
theorem hyperbola_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ hyperbola x₀ y₀ a b) →
  (∃ (x y : ℝ), asymptote x y) →
  ∀ (x y : ℝ), hyperbola x y a b ↔ x^2 - y^2/3 = 1 :=
sorry

end hyperbola_equation_l2717_271782


namespace number_division_theorem_l2717_271753

theorem number_division_theorem (x : ℚ) : 
  x / 6 = 1 / 10 → x / (3 / 25) = 5 := by
  sorry

end number_division_theorem_l2717_271753


namespace set_equality_implies_a_squared_minus_b_zero_l2717_271709

theorem set_equality_implies_a_squared_minus_b_zero (a b : ℝ) (h : a ≠ 0) :
  ({1, a + b, a} : Set ℝ) = {0, b / a, b} →
  a^2 - b = 0 := by
sorry

end set_equality_implies_a_squared_minus_b_zero_l2717_271709


namespace board_sum_l2717_271795

theorem board_sum : ∀ (numbers : List ℕ),
  (numbers.length = 9) →
  (∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 5) →
  (numbers.filter (λ n => n ≥ 2)).length ≥ 7 →
  (numbers.filter (λ n => n > 2)).length ≥ 6 →
  (numbers.filter (λ n => n ≥ 4)).length ≥ 3 →
  (numbers.filter (λ n => n ≥ 5)).length ≥ 1 →
  numbers.sum = 26 := by
sorry

end board_sum_l2717_271795


namespace quadratic_roots_l2717_271774

theorem quadratic_roots (x : ℝ) : x^2 - 3*x + 2 = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end quadratic_roots_l2717_271774


namespace largest_n_satisfying_conditions_l2717_271785

theorem largest_n_satisfying_conditions : 
  ∃ (n : ℤ), n = 313 ∧ 
  (∀ (x : ℤ), x > n → 
    (¬∃ (m : ℤ), x^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℤ), 2*x + 103 = k^2)) ∧
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧
  (∃ (k : ℤ), 2*n + 103 = k^2) := by
sorry

end largest_n_satisfying_conditions_l2717_271785


namespace alyssa_puppies_l2717_271792

/-- The number of puppies Alyssa has after breeding and giving some away -/
def remaining_puppies (initial : ℕ) (puppies_per_puppy : ℕ) (given_away : ℕ) : ℕ :=
  initial + initial * puppies_per_puppy - given_away

/-- Theorem stating that Alyssa has 20 puppies left -/
theorem alyssa_puppies : remaining_puppies 7 4 15 = 20 := by
  sorry

end alyssa_puppies_l2717_271792


namespace binomial_expansion_result_l2717_271713

theorem binomial_expansion_result (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end binomial_expansion_result_l2717_271713


namespace third_month_sale_proof_l2717_271772

/-- Calculates the sale in the third month given the sales for other months and the average --/
def third_month_sale (first_month : ℕ) (second_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  6 * average - (first_month + second_month + fourth_month + fifth_month + sixth_month)

theorem third_month_sale_proof :
  third_month_sale 5266 5744 6122 6588 4916 5750 = 5864 := by
  sorry

end third_month_sale_proof_l2717_271772


namespace ice_skating_skiing_probability_l2717_271780

theorem ice_skating_skiing_probability (P_ice_skating P_skiing P_either : ℝ)
  (h1 : P_ice_skating = 0.6)
  (h2 : P_skiing = 0.5)
  (h3 : P_either = 0.7)
  (h4 : 0 ≤ P_ice_skating ∧ P_ice_skating ≤ 1)
  (h5 : 0 ≤ P_skiing ∧ P_skiing ≤ 1)
  (h6 : 0 ≤ P_either ∧ P_either ≤ 1) :
  (P_ice_skating + P_skiing - P_either) / P_skiing = 0.8 :=
by sorry

end ice_skating_skiing_probability_l2717_271780


namespace max_min_f_on_interval_l2717_271764

-- Define the function f(x) = -3x + 1
def f (x : ℝ) : ℝ := -3 * x + 1

-- State the theorem
theorem max_min_f_on_interval :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 1, f x = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f x = -2) :=
sorry

end max_min_f_on_interval_l2717_271764


namespace complex_distance_bounds_l2717_271749

theorem complex_distance_bounds (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  (∃ (w : ℂ), Complex.abs (z - 2 - 2*I) = 3 ∧ 
    ∀ (u : ℂ), Complex.abs (u + 2 - 2*I) = 1 → Complex.abs (u - 2 - 2*I) ≥ 3) ∧
  (∃ (w : ℂ), Complex.abs (z - 2 - 2*I) = 5 ∧ 
    ∀ (u : ℂ), Complex.abs (u + 2 - 2*I) = 1 → Complex.abs (u - 2 - 2*I) ≤ 5) :=
sorry

end complex_distance_bounds_l2717_271749


namespace negation_of_existence_inequality_l2717_271771

theorem negation_of_existence_inequality (p : Prop) :
  (p ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) →
  (¬p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existence_inequality_l2717_271771


namespace range_of_a_theorem_l2717_271723

/-- Proposition p: For all x ∈ [1, 2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: The equation x^2 + 2ax + 2 - a = 0 has real roots -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The proposition "¬p ∨ ¬q" is false -/
def not_p_or_not_q_is_false (a : ℝ) : Prop :=
  ¬(¬(prop_p a) ∨ ¬(prop_q a))

/-- The range of the real number a is a ≤ -2 or a = 1 -/
def range_of_a (a : ℝ) : Prop :=
  a ≤ -2 ∨ a = 1

theorem range_of_a_theorem (a : ℝ) :
  prop_p a ∧ prop_q a ∧ not_p_or_not_q_is_false a → range_of_a a :=
by
  sorry

end range_of_a_theorem_l2717_271723


namespace part1_part2_l2717_271776

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -2)
def C : ℝ × ℝ := (4, 1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Part 1
theorem part1 : ∀ D : ℝ × ℝ, AB = (D.1 - C.1, D.2 - C.2) → D = (5, -4) := by sorry

-- Part 2
theorem part2 : ∀ k : ℝ, (∃ t : ℝ, t ≠ 0 ∧ (k * AB.1 - BC.1, k * AB.2 - BC.2) = (t * (AB.1 + 3 * BC.1), t * (AB.2 + 3 * BC.2))) → k = -1/3 := by sorry

end part1_part2_l2717_271776


namespace area_fraction_above_line_l2717_271762

/-- The fraction of the area of a square above a line -/
def fraction_above_line (square_vertices : Fin 4 → ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ) : ℚ :=
  sorry

/-- The theorem statement -/
theorem area_fraction_above_line :
  let square_vertices : Fin 4 → ℝ × ℝ := ![
    (2, 1), (5, 1), (5, 4), (2, 4)
  ]
  let line_point1 : ℝ × ℝ := (2, 3)
  let line_point2 : ℝ × ℝ := (5, 1)
  fraction_above_line square_vertices line_point1 line_point2 = 2/3 := by
  sorry

end area_fraction_above_line_l2717_271762


namespace square_equation_solution_l2717_271741

theorem square_equation_solution (x y : ℕ) : 
  x^2 = y^2 + 7*y + 6 → x = 6 ∧ y = 3 := by
  sorry

end square_equation_solution_l2717_271741


namespace compound_molecular_weight_l2717_271703

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- The number of Aluminum atoms in the compound -/
def Al_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- The number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := Al_weight * Al_count + O_weight * O_count + H_weight * H_count

theorem compound_molecular_weight : molecular_weight = 78.01 := by
  sorry

end compound_molecular_weight_l2717_271703


namespace problem_proof_l2717_271702

theorem problem_proof (m n : ℕ) (h1 : m + 9 < n) 
  (h2 : (m + (m + 3) + (m + 9) + n + (n + 1) + (2*n - 1)) / 6 = n - 1) 
  (h3 : (m + 9 + n) / 2 = n - 1) : m + n = 21 := by
  sorry

end problem_proof_l2717_271702


namespace pam_has_1200_apples_l2717_271733

/-- The number of apples Pam has in total -/
def pams_total_apples (pams_bags : ℕ) (geralds_apples_per_bag : ℕ) : ℕ :=
  pams_bags * (3 * geralds_apples_per_bag)

/-- Theorem stating that Pam has 1200 apples given the conditions -/
theorem pam_has_1200_apples :
  pams_total_apples 10 40 = 1200 := by
  sorry

#eval pams_total_apples 10 40

end pam_has_1200_apples_l2717_271733


namespace captain_times_proof_l2717_271775

-- Define the points and captain times for each boy
def points_A : ℕ := sorry
def points_E : ℕ := sorry
def points_B : ℕ := sorry
def captain_time_A : ℕ := sorry
def captain_time_E : ℕ := sorry
def captain_time_B : ℕ := sorry

-- Define the total travel time
def total_time : ℕ := sorry

-- State the theorem
theorem captain_times_proof :
  -- Conditions
  (points_A = points_B + 3) →
  (points_E + points_B = 15) →
  (total_time / 10 = points_A + points_E + points_B + 25) →
  (captain_time_B = 160) →
  -- Proportionality condition
  (∃ (k : ℚ), 
    captain_time_A = k * points_A ∧
    captain_time_E = k * points_E ∧
    captain_time_B = k * points_B) →
  -- Conclusion
  (captain_time_A = 200 ∧ captain_time_B = 140) :=
by sorry

end captain_times_proof_l2717_271775


namespace school_population_l2717_271798

theorem school_population (b g t : ℕ) : 
  b = 3 * g → g = 9 * t → b + g + t = (37 * b) / 27 := by
  sorry

end school_population_l2717_271798


namespace cylinder_height_l2717_271787

/-- A cylinder with given lateral area and volume has height 3 -/
theorem cylinder_height (r h : ℝ) (h_positive : h > 0) (r_positive : r > 0) 
  (lateral_area : 2 * π * r * h = 12 * π) 
  (volume : π * r^2 * h = 12 * π) : h = 3 := by
  sorry

end cylinder_height_l2717_271787


namespace second_group_size_l2717_271784

/-- Represents a tour group with a number of people -/
structure TourGroup where
  people : ℕ

/-- Represents a day's tour schedule -/
structure TourSchedule where
  group1 : TourGroup
  group2 : TourGroup
  group3 : TourGroup
  group4 : TourGroup

def questions_per_tourist : ℕ := 2

def total_questions : ℕ := 68

theorem second_group_size (schedule : TourSchedule) : 
  schedule.group1.people = 6 ∧ 
  schedule.group3.people = 8 ∧ 
  schedule.group4.people = 7 ∧
  questions_per_tourist * (schedule.group1.people + schedule.group2.people + schedule.group3.people + schedule.group4.people) = total_questions →
  schedule.group2.people = 13 := by
  sorry

end second_group_size_l2717_271784


namespace wire_length_l2717_271756

/-- The length of a wire cut into two pieces, where one piece is 2/3 of the other --/
theorem wire_length (shorter_piece : ℝ) (h : shorter_piece = 27.999999999999993) : 
  ∃ (longer_piece total_length : ℝ),
    longer_piece = (2/3) * shorter_piece ∧
    total_length = shorter_piece + longer_piece ∧
    total_length = 46.66666666666666 := by
  sorry

end wire_length_l2717_271756


namespace greatest_integer_c_l2717_271789

-- Define the numerator and denominator of the expression
def numerator (x : ℝ) : ℝ := 16 * x^3 + 5 * x^2 + 28 * x + 12
def denominator (c x : ℝ) : ℝ := x^2 + c * x + 12

-- Define the condition for the expression to have a domain of all real numbers
def has_full_domain (c : ℝ) : Prop :=
  ∀ x : ℝ, denominator c x ≠ 0

-- State the theorem
theorem greatest_integer_c :
  (∃ c : ℤ, has_full_domain (c : ℝ) ∧ 
   ∀ d : ℤ, d > c → ¬has_full_domain (d : ℝ)) ∧
  (∃ c : ℤ, c = 6 ∧ has_full_domain (c : ℝ) ∧ 
   ∀ d : ℤ, d > c → ¬has_full_domain (d : ℝ)) :=
by sorry

end greatest_integer_c_l2717_271789


namespace cube_root_simplification_l2717_271716

theorem cube_root_simplification :
  (40^3 + 50^3 + 60^3 : ℝ)^(1/3) = 10 * 405^(1/3) := by
  sorry

end cube_root_simplification_l2717_271716


namespace best_strategy_is_red_l2717_271770

/-- Represents the color of a disk side -/
inductive Color
| Red
| Blue

/-- Represents a disk with two sides -/
structure Disk where
  side1 : Color
  side2 : Color

/-- The set of all disks in the hat -/
def diskSet : Finset Disk := sorry

/-- The total number of disks -/
def totalDisks : ℕ := 10

/-- The number of disks with both sides red -/
def redDisks : ℕ := 3

/-- The number of disks with both sides blue -/
def blueDisks : ℕ := 2

/-- The number of disks with one side red and one side blue -/
def mixedDisks : ℕ := 5

/-- The probability of observing a red side -/
def probRedSide : ℚ := 11 / 20

/-- The probability of observing a blue side -/
def probBlueSide : ℚ := 9 / 20

/-- The probability that the other side is red, given that a red side is observed -/
def probRedGivenRed : ℚ := 6 / 11

/-- The probability that the other side is red, given that a blue side is observed -/
def probRedGivenBlue : ℚ := 5 / 9

theorem best_strategy_is_red :
  probRedGivenRed > 1 / 2 ∧ probRedGivenBlue > 1 / 2 := by sorry

end best_strategy_is_red_l2717_271770


namespace total_lives_after_joining_l2717_271717

theorem total_lives_after_joining (initial_players : Nat) (joined_players : Nat) (lives_per_player : Nat) : 
  initial_players = 8 → joined_players = 2 → lives_per_player = 6 → 
  (initial_players + joined_players) * lives_per_player = 60 := by
  sorry

end total_lives_after_joining_l2717_271717


namespace isosceles_triangle_k_values_l2717_271768

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  isIsosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ (side1 = side3 ∧ side2 ≠ side1) ∨ (side2 = side3 ∧ side1 ≠ side2)

-- Define the quadratic equation
def quadraticRoots (k : ℝ) : Set ℝ :=
  {x : ℝ | x^2 - 4*x + k = 0}

-- Theorem statement
theorem isosceles_triangle_k_values :
  ∀ (t : IsoscelesTriangle) (k : ℝ),
    (t.side1 = 3 ∨ t.side2 = 3 ∨ t.side3 = 3) →
    (∃ (x y : ℝ), x ∈ quadraticRoots k ∧ y ∈ quadraticRoots k ∧ 
      ((t.side1 = x ∧ t.side2 = y) ∨ (t.side1 = x ∧ t.side3 = y) ∨ (t.side2 = x ∧ t.side3 = y))) →
    (k = 3 ∨ k = 4) :=
by sorry


end isosceles_triangle_k_values_l2717_271768


namespace fruits_given_to_jane_l2717_271714

def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def initial_apples : ℕ := 21
def fruits_left : ℕ := 15

def total_initial_fruits : ℕ := initial_plums + initial_guavas + initial_apples

theorem fruits_given_to_jane : 
  total_initial_fruits - fruits_left = 40 := by sorry

end fruits_given_to_jane_l2717_271714


namespace john_test_scores_l2717_271707

theorem john_test_scores (total_tests : ℕ) (target_percentage : ℚ) 
  (tests_taken : ℕ) (tests_at_target : ℕ) : 
  total_tests = 60 →
  target_percentage = 85 / 100 →
  tests_taken = 40 →
  tests_at_target = 28 →
  (total_tests - tests_taken : ℕ) - 
    (↑total_tests * target_percentage - tests_at_target : ℚ).floor = 0 :=
by sorry

end john_test_scores_l2717_271707


namespace radius_q3_is_one_point_five_l2717_271721

/-- A triangle with an inscribed circle and two additional tangent circles -/
structure TripleCircleTriangle where
  /-- Side length AB of the triangle -/
  ab : ℝ
  /-- Side length BC of the triangle -/
  bc : ℝ
  /-- Side length AC of the triangle -/
  ac : ℝ
  /-- Radius of the inscribed circle Q1 -/
  r1 : ℝ
  /-- Radius of circle Q2, tangent to Q1 and sides AB and BC -/
  r2 : ℝ
  /-- Radius of circle Q3, tangent to Q2 and sides AB and BC -/
  r3 : ℝ
  /-- AB equals BC -/
  ab_eq_bc : ab = bc
  /-- AB equals 80 -/
  ab_eq_80 : ab = 80
  /-- AC equals 96 -/
  ac_eq_96 : ac = 96
  /-- Q1 is inscribed in the triangle -/
  q1_inscribed : r1 = (ab + bc + ac) / 2 - ab
  /-- Q2 is tangent to Q1 and sides AB and BC -/
  q2_tangent : r2 = r1 / 4
  /-- Q3 is tangent to Q2 and sides AB and BC -/
  q3_tangent : r3 = r2 / 4

/-- The radius of Q3 is 1.5 in the given triangle configuration -/
theorem radius_q3_is_one_point_five (t : TripleCircleTriangle) : t.r3 = 1.5 := by
  sorry

end radius_q3_is_one_point_five_l2717_271721


namespace point_coordinates_l2717_271760

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    SecondQuadrant p →
    DistToXAxis p = 3 →
    DistToYAxis p = 7 →
    p.x = -7 ∧ p.y = 3 :=
by sorry

end point_coordinates_l2717_271760


namespace not_equivalent_expression_l2717_271744

theorem not_equivalent_expression (x : ℝ) : 
  (3 * (x + 2) = 3 * x + 6) ∧
  ((-9 * x - 18) / (-3) = 3 * x + 6) ∧
  ((1/3) * (9 * x + 18) = 3 * x + 6) ∧
  ((1/3) * (3 * x) + (2/3) * 9 ≠ 3 * x + 6) :=
by sorry

end not_equivalent_expression_l2717_271744


namespace last_digit_difference_l2717_271732

theorem last_digit_difference (p q : ℕ) : 
  p > q → 
  p % 10 ≠ 0 → 
  q % 10 ≠ 0 → 
  ∃ k : ℕ, p * q = 10^k → 
  (p - q) % 10 ≠ 5 :=
by sorry

end last_digit_difference_l2717_271732


namespace monkey_banana_distribution_l2717_271757

/-- Calculates the number of bananas each monkey receives when a family of monkeys divides a collection of bananas equally -/
def bananas_per_monkey (num_monkeys : ℕ) (num_piles_type1 : ℕ) (hands_per_pile_type1 : ℕ) (bananas_per_hand_type1 : ℕ)
                       (num_piles_type2 : ℕ) (hands_per_pile_type2 : ℕ) (bananas_per_hand_type2 : ℕ) : ℕ :=
  let total_bananas := num_piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
                       num_piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2
  total_bananas / num_monkeys

/-- Theorem stating that under the given conditions, each monkey receives 99 bananas -/
theorem monkey_banana_distribution :
  bananas_per_monkey 12 6 9 14 4 12 9 = 99 := by
  sorry

end monkey_banana_distribution_l2717_271757


namespace chocolate_bar_cost_l2717_271786

-- Define the total number of chocolate bars
def total_bars : ℕ := 9

-- Define the number of unsold bars
def unsold_bars : ℕ := 3

-- Define the total amount made from the sale
def total_amount : ℕ := 18

-- Theorem to prove
theorem chocolate_bar_cost :
  ∃ (cost : ℚ), cost * (total_bars - unsold_bars) = total_amount ∧ cost = 3 := by
  sorry

end chocolate_bar_cost_l2717_271786


namespace expand_expression_l2717_271731

theorem expand_expression (x y z : ℝ) : 
  (2*x - 3) * (4*y + 5 - 2*z) = 8*x*y + 10*x - 4*x*z - 12*y + 6*z - 15 := by
sorry

end expand_expression_l2717_271731


namespace product_greater_than_sum_minus_one_l2717_271779

theorem product_greater_than_sum_minus_one {a₁ a₂ : ℝ} 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ > a₁ + a₂ - 1 := by
  sorry

end product_greater_than_sum_minus_one_l2717_271779


namespace phone_plan_fee_proof_l2717_271750

/-- The monthly fee for the first plan -/
def first_plan_fee : ℝ := 22

/-- The per-minute rate for the first plan -/
def first_plan_rate : ℝ := 0.13

/-- The per-minute rate for the second plan -/
def second_plan_rate : ℝ := 0.18

/-- The number of minutes at which the plans cost the same -/
def equal_cost_minutes : ℝ := 280

/-- The monthly fee for the second plan -/
def second_plan_fee : ℝ := 8

theorem phone_plan_fee_proof :
  first_plan_fee + first_plan_rate * equal_cost_minutes =
  second_plan_fee + second_plan_rate * equal_cost_minutes :=
by sorry

end phone_plan_fee_proof_l2717_271750


namespace parabola_shift_l2717_271781

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The shifted parabola function -/
def g (x : ℝ) : ℝ := (x + 1)^2 - 4*(x + 1) + 3 + 2

/-- Theorem stating that the shifted parabola is equivalent to x^2 - 2x + 2 -/
theorem parabola_shift :
  ∀ x : ℝ, g x = x^2 - 2*x + 2 :=
by sorry

end parabola_shift_l2717_271781


namespace no_integer_solution_l2717_271799

/-- The equation x^3 - 3xy^2 + y^3 = 2891 has no integer solutions -/
theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2891 := by
  sorry

end no_integer_solution_l2717_271799


namespace supermarket_flour_import_l2717_271722

theorem supermarket_flour_import (long_grain : ℚ) (glutinous : ℚ) (flour : ℚ) : 
  long_grain = 9/20 →
  glutinous = 7/20 →
  flour = long_grain + glutinous - 3/20 →
  flour = 13/20 := by
sorry

end supermarket_flour_import_l2717_271722


namespace st_plus_tu_equals_ten_l2717_271720

/-- Represents a polygon PQRSTU -/
structure Polygon where
  area : ℝ
  pq : ℝ
  qr : ℝ
  up : ℝ
  st : ℝ
  tu : ℝ

/-- Theorem stating the sum of ST and TU in the given polygon -/
theorem st_plus_tu_equals_ten (poly : Polygon) 
  (h_area : poly.area = 64)
  (h_pq : poly.pq = 10)
  (h_qr : poly.qr = 10)
  (h_up : poly.up = 6) :
  poly.st + poly.tu = 10 := by
  sorry

end st_plus_tu_equals_ten_l2717_271720


namespace caiden_roofing_cost_l2717_271766

def metal_roofing_cost (total_feet : ℕ) (free_feet : ℕ) (cost_per_foot : ℚ) 
  (discount_rate : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let paid_feet := total_feet - free_feet
  let initial_cost := paid_feet * cost_per_foot
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_before_tax := discounted_cost
  let total_cost := total_before_tax * (1 + sales_tax_rate)
  total_cost

theorem caiden_roofing_cost :
  metal_roofing_cost 300 250 8 (15/100) (5/100) = 357 := by
  sorry

end caiden_roofing_cost_l2717_271766


namespace inscribed_circle_radius_l2717_271700

theorem inscribed_circle_radius (AB AC BC : ℝ) (h_AB : AB = 8) (h_AC : AC = 10) (h_BC : BC = 12) :
  let s := (AB + AC + BC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  area / s = Real.sqrt 7 := by
  sorry

end inscribed_circle_radius_l2717_271700


namespace correct_transformation_l2717_271788

theorem correct_transformation (x : ℝ) : (x / 2 - x / 3 = 1) ↔ (3 * x - 2 * x = 6) := by
  sorry

end correct_transformation_l2717_271788


namespace smallest_stairs_fifty_three_satisfies_stairs_solution_l2717_271743

theorem smallest_stairs (n : ℕ) : 
  (n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4) → n ≥ 53 :=
by sorry

theorem fifty_three_satisfies : 
  53 > 20 ∧ 53 % 6 = 5 ∧ 53 % 7 = 4 :=
by sorry

theorem stairs_solution : 
  ∃ (n : ℕ), n = 53 ∧ n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4 ∧
  ∀ (m : ℕ), (m > 20 ∧ m % 6 = 5 ∧ m % 7 = 4) → m ≥ n :=
by sorry

end smallest_stairs_fifty_three_satisfies_stairs_solution_l2717_271743
