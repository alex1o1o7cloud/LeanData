import Mathlib

namespace point_to_polar_coordinates_l716_716989

noncomputable def convert_to_polar_coordinates (x y : ℝ) (r θ : ℝ) : Prop :=
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x)

theorem point_to_polar_coordinates :
  convert_to_polar_coordinates 8 (2 * Real.sqrt 6) 
    (2 * Real.sqrt 22) (Real.arctan (Real.sqrt 6 / 4)) :=
sorry

end point_to_polar_coordinates_l716_716989


namespace z_pow6_eq_neg1_l716_716812

noncomputable def z : ℂ := (real.sqrt 3 - complex.i) / 2

theorem z_pow6_eq_neg1 : z^6 = -1 := 
by sorry

end z_pow6_eq_neg1_l716_716812


namespace probability_convex_number_l716_716232

-- define conditions
def digits := {1, 2, 3, 4}
structure IsDifferent (a b c : ℕ) : Prop :=
  (a_different_b : a ≠ b)
  (b_different_c : b ≠ c)
  (a_different_c : a ≠ c)

def is_convex (a b c : ℕ) := a < b ∧ b > c

-- main theorem statement
theorem probability_convex_number :
  let all_numbers := {n // ∃ a b c, n = a * 100 + b * 10 + c ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ IsDifferent a b c}
      convex_numbers := {n ∈ all_numbers | ∃ a b c, n = a * 100 + b * 10 + c ∧ is_convex a b c} in
  (convex_numbers.finite.to_finset.card : ℚ) / (all_numbers.finite.to_finset.card : ℚ) = 1 / 3 :=
by
  -- skipping the proof for brevity
  sorry

end probability_convex_number_l716_716232


namespace B_3_2_eq_4_l716_716990

def B (m n : ℕ) : ℕ :=
  if m = 0 then n + 1
  else if n = 0 then B (m - 1) 2
  else B (m - 1) (B m (n - 1))

theorem B_3_2_eq_4 : B 3 2 = 4 :=
  sorry

end B_3_2_eq_4_l716_716990


namespace sample_analysis_l716_716943

-- Definitions for the given problem
def x_values : List ℝ := [5, 6, 8, 9, 12]
def y_values : List ℝ := [17, 20, 25, 28, 35]
def regression_slope : ℝ := 2.6
def regression_intercept : ℝ := 4.2

-- Compute the average (center point) of the sample
def average (xs : List ℝ) : ℝ := xs.reduce (/ .size)

-- Center point computation
def center_x := average x_values
def center_y := average y_values

-- Function for regression prediction
def regression_pred (x : ℝ) : ℝ := regression_slope * x + regression_intercept

-- Residual computation
def residual (x y : ℝ) := y - regression_pred x

-- Lean statement signifies the proof problem
theorem sample_analysis :
  center_x = 8 ∧ center_y = 25 ∧ regression_intercept = 4.2 ∧ residual 5 17 = -0.2 :=
by
  sorry

end sample_analysis_l716_716943


namespace regular_polygon_sides_l716_716163

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716163


namespace find_t_l716_716744

theorem find_t (t : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - t| + |5 - x|) (h2 : ∃ x, f x = 3) : t = 2 ∨ t = 8 :=
by
  sorry

end find_t_l716_716744


namespace regular_polygon_sides_l716_716139

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716139


namespace find_a_l716_716348

def z1 : ℂ := -2 + complex.i

def z2 (a : ℝ) : ℂ := a + 2 * complex.i

theorem find_a (a : ℝ) (h : (z1 * z2 a).im = 0) : a = 4 :=
by
  sorry

end find_a_l716_716348


namespace adam_initial_boxes_l716_716620

theorem adam_initial_boxes (given_pieces : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) (pieces_given : ℕ) (initial_boxes : ℕ) :
  pieces_given = 7 → pieces_per_box = 6 → remaining_pieces = 36 → initial_boxes = ((pieces_given + remaining_pieces) / pieces_per_box).to_nat → initial_boxes = 7 :=
by
  intros
  cases h₃ : ((pieces_given + remaining_pieces) / pieces_per_box).to_nat
  . sorry

end adam_initial_boxes_l716_716620


namespace train_speed_l716_716908

theorem train_speed (L : ℝ) (T : ℝ) (V_m : ℝ) (V_t : ℝ) : (L = 500) → (T = 29.997600191984642) → (V_m = 5 / 6) → (V_t = (L / T) + V_m) → (V_t * 3.6 = 63) :=
by
  intros hL hT hVm hVt
  simp at hL hT hVm hVt
  sorry

end train_speed_l716_716908


namespace range_of_a_l716_716717

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a = x^2 - x - 1) ↔ -1 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l716_716717


namespace businessmen_neither_coffee_nor_tea_l716_716629

/-- Definitions of conditions -/
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 6

/-- Statement of the problem -/
theorem businessmen_neither_coffee_nor_tea : 
  (total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end businessmen_neither_coffee_nor_tea_l716_716629


namespace average_mark_of_second_class_l716_716874

/-- 
There is a class of 30 students with an average mark of 40. 
Another class has 50 students with an unknown average mark. 
The average marks of all students combined is 65. 
Prove that the average mark of the second class is 80.
-/
theorem average_mark_of_second_class (x : ℝ) (h1 : 30 * 40 + 50 * x = 65 * (30 + 50)) : x = 80 := 
sorry

end average_mark_of_second_class_l716_716874


namespace correct_number_of_possible_values_l716_716925

def initial_set : Set ℕ := {3, 6, 9, 10}

def is_median (s : List ℕ) (n : ℕ) :=
  let sorted_list := s.sorted
  sorted_list.getI (s.length / 2) = n

def is_mean (s : List ℕ) (n : ℕ) :=
  (s.sum / s.length) = n

def num_possible_values : ℕ :=
  (Finset.range 101).filter (λ n =>
    let new_set := [3, 6, 9, 10, n]
    is_median new_set n ∧ is_mean (initial_set.to_list :+ n) n  
  ).card

theorem correct_number_of_possible_values (h : ∀ n, n ∈ initial_set → n ∈ Finset.range 101) :
  num_possible_values = 3 := by
  sorry

end correct_number_of_possible_values_l716_716925


namespace smallest_positive_period_pi_monotonically_increasing_l716_716237

theorem smallest_positive_period_pi_monotonically_increasing :
  ∃ f, (f = (λ x, abs (sin x)) ∨ f = (λ x, tan x) ∨ f = (λ x, abs (tan x))) ∧ 
  (∀ x, 0 < x → x < π → f x = f (x + π)) ∧ 
  (∀ x y, 0 < x → x < y → y < π/2 → f x < f y) :=
sorry

end smallest_positive_period_pi_monotonically_increasing_l716_716237


namespace quadratic_inequality_l716_716448

variables {x y x₁ x₂ y₁ y₂ : ℝ}
variables {a b c : ℝ}

theorem quadratic_inequality 
  (h₀ : ∀ x y : ℝ, a * x^2 + 2 * b * x * y + c * y^2 ≥ 0)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (√((a * x₁^2 + 2 * b * x₁ * y₁ + c * y₁^2) * (a * x₂^2 + 2 * b * x₂ * y₂ + c * y₂^2))) *
  (a * (x₁ - x₂)^2 + 2 * b * (x₁ - x₂) * (y₁ - y₂) + c * (y₁ - y₂)^2) ≥
  (a * c - b^2) * (x₁ * y₂ - x₂ * y₁)^2 :=
sorry

end quadratic_inequality_l716_716448


namespace find_f_1_minus_a_l716_716296

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

theorem find_f_1_minus_a 
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_period : periodic_function f 2)
  (h_value : ∃ a : ℝ, f (1 + a) = 1) :
  ∃ a : ℝ, f (1 - a) = -1 :=
by
  sorry

end find_f_1_minus_a_l716_716296


namespace regular_polygon_sides_l716_716146

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716146


namespace total_animal_count_l716_716524

theorem total_animal_count (initial_hippos : ℕ) (initial_elephants : ℕ) 
  (female_hippo_factor : ℚ) (newborn_per_female_hippo : ℕ) 
  (extra_newborn_elephants : ℕ) 
  (h_initial_hippos : initial_hippos = 35)
  (h_initial_elephants : initial_elephants = 20)
  (h_female_hippo_factor : female_hippo_factor = 5 / 7)
  (h_newborn_per_female_hippo : newborn_per_female_hippo = 5)
  (h_extra_newborn_elephants : extra_newborn_elephants = 10) : 
  (initial_elephants + initial_hippos + 
  (initial_hippos * female_hippo_factor).to_nat * newborn_per_female_hippo + 
  (initial_hippos * female_hippo_factor).to_nat * newborn_per_female_hippo + 
  extra_newborn_elephants) = 315 :=
by
  sorry

end total_animal_count_l716_716524


namespace regular_polygon_sides_l716_716200

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716200


namespace system_of_equations_solution_l716_716488

theorem system_of_equations_solution (x y : ℝ) : 
  x^2 + y^2 = 34 ∧ x - y + sqrt ((x - y) / (x + y)) = 20 / (x + y) ↔ 
  (x, y) = (5, -3) ∨ (x, y) = (5, 3) ∨ (x, y) = (-sqrt 118 / 2, 3 * sqrt 2 / 2) ∨ 
  (x, y) = (-sqrt 118 / 2, -3 * sqrt 2 / 2) :=
by 
  sorry

end system_of_equations_solution_l716_716488


namespace cost_of_camel_proof_l716_716051

noncomputable def cost_of_camel (C H O E : ℕ) : ℕ :=
  if 10 * C = 24 * H ∧ 16 * H = 4 * O ∧ 6 * O = 4 * E ∧ 10 * E = 120000 then 4800 else 0

theorem cost_of_camel_proof (C H O E : ℕ) 
  (h1 : 10 * C = 24 * H) (h2 : 16 * H = 4 * O) (h3 : 6 * O = 4 * E) (h4 : 10 * E = 120000) :
  cost_of_camel C H O E = 4800 :=
by
  sorry

end cost_of_camel_proof_l716_716051


namespace regular_polygon_sides_l716_716099

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716099


namespace positional_relationship_l716_716735

-- Definitions of the required predicates.

def skew (a b : Line) : Prop := 
  -- Definition of skew lines
  sorry

def parallel_to_plane (a : Line) (α : Plane) : Prop := 
  -- Definition of a line parallel to a plane
  sorry

def intersect_plane (b : Line) (α : Plane) : Prop := 
  -- Definition of a line intersecting a plane
  sorry

def in_plane (b : Line) (α : Plane) : Prop :=
  -- Definition of a line being in a plane
  sorry

-- Theorem statement
theorem positional_relationship (a b : Line) (α : Plane) :
  skew a b → parallel_to_plane a α → 
  (parallel_to_plane b α ∨ intersect_plane b α ∨ in_plane b α) :=
begin
  sorry
end

end positional_relationship_l716_716735


namespace max_students_in_exam_l716_716965

theorem max_students_in_exam : 
  let seat_count := fun (i : ℕ) => 8 + 2 * (i - 1)
  let usable_seats_per_row := fun (i : ℕ) => seat_count i - 1
  let students_per_row := fun (i : ℕ) => (usable_seats_per_row i) / 2
  ((∑ i in finset.range 15, students_per_row (i + 1)) = 150) :=
by
  sorry

end max_students_in_exam_l716_716965


namespace sales_price_relationship_l716_716921

theorem sales_price_relationship (x : ℕ) (hx : 1 ≤ x) :
  total_sales_price (x : ℕ) = 40.5 * (x : Real) := by
  sorry

end sales_price_relationship_l716_716921


namespace mean_and_variance_unchanged_with_replacement_l716_716414

def S : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

structure Replacement :=
  (a b c : ℤ)
  (is_in_S : a ∈ S)
  (b_plus_c_eq_a : b + c = a)
  (b_sq_plus_c_sq_minus_a_sq_eq_ten : b^2 + c^2 - a^2 = 10)

def replacements : List Replacement :=
  [ { a := 4, b := -1, c := 5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num },
    { a := -4, b := 1, c := -5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num } ]

theorem mean_and_variance_unchanged_with_replacement (r : Replacement) :
  -- assuming r comes from the list of replacements
  ∃ (r ∈ replacements), r = r :=
begin
  sorry
end

end mean_and_variance_unchanged_with_replacement_l716_716414


namespace smallest_number_with_ten_divisors_l716_716864

/-- 
  Theorem: The smallest natural number n that has exactly 10 positive divisors is 48.
--/
theorem smallest_number_with_ten_divisors : 
  ∃ (n : ℕ), (∀ (p1 p2 p3 p4 p5 : ℕ) (a1 a2 a3 a4 a5 : ℕ), 
    n = p1^a1 * p2^a2 * p3^a3 * p4^a4 * p5^a5 → 
    n.factors.count = 10) 
    ∧ n = 48 := sorry

end smallest_number_with_ten_divisors_l716_716864


namespace regular_polygon_sides_l716_716072

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716072


namespace sum_of_coordinates_l716_716326

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 12 = 8) : 
  let y := (1 / 3) * f (12 / 3) + 1 in -- follows from f(3*4) and the transformed equation
  (4 : ℝ) + y = (53 / 9 : ℝ) :=
  by
  let y := (17 / 9 : ℝ)
  sorry


end sum_of_coordinates_l716_716326


namespace max_marked_points_l716_716685

theorem max_marked_points (n : ℕ) (h : n > 2) (h_collinear : ∀ (P Q R : ℝ × ℝ), 
  P ∈ some_points → Q ∈ some_points → R ∈ some_points → (P ≠ Q ∧ P ≠ R ∧ Q ≠ R) → 
  ¬collinear P Q R) (h_unique_closest_point : ∀ (P Q : ℝ × ℝ), 
  P ∈ some_points → Q ∈ some_points → (P ≠ Q) → 
  ∃! R, R ∈ some_points ∧ R ≠ P ∧ R ≠ Q ∧ is_closest R (line_through P Q)) : 
  ∃ m, m = 3 ∧ max_marked_some_points some_points m
:= sorry

end max_marked_points_l716_716685


namespace regular_polygon_sides_l716_716145

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716145


namespace regular_polygon_sides_l716_716184

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716184


namespace regular_polygon_sides_l716_716084

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716084


namespace regular_polygon_sides_l716_716176

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716176


namespace number_of_customers_l716_716579

theorem number_of_customers (offices_sandwiches : Nat)
                            (group_per_person_sandwiches : Nat)
                            (total_sandwiches : Nat)
                            (half_group : Nat) :
  (offices_sandwiches = 3 * 10) →
  (total_sandwiches = 54) →
  (half_group * group_per_person_sandwiches = total_sandwiches - offices_sandwiches) →
  (2 * half_group = 12) := 
by
  sorry

end number_of_customers_l716_716579


namespace remainder_when_divided_by_8_l716_716548

theorem remainder_when_divided_by_8:
  ∀ (n : ℕ), (∃ (q : ℕ), n = 7 * q + 5) → n % 8 = 1 :=
by
  intro n h
  rcases h with ⟨q, hq⟩
  sorry

end remainder_when_divided_by_8_l716_716548


namespace Belchonok_arrange_bags_l716_716633

theorem Belchonok_arrange_bags :
  ∃ (shelf1 shelf2 : List (ℕ × ℕ)),
    (∀ (s1 s2 : List (ℕ × ℕ)), 
      (s1 = [(2, 2), (2, 2)] ++ [(3, 3), (3, 3), (3, 3)] ++ [(4, 4), (4, 4), (4, 4), (4, 4)] ++[(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]) ∧
      sorted_by s1 s2 ∧
      (s2 = [(5, 5), (5, 5), (5, 5)] ++ [(4, 4), (4, 4)] ++ [(2, 2), (2, 2)]) ∧
      (shelf1.sum = [(2, 2)].sum + [(2, 2)].sum + [(3, 3)].sum + [(3, 3)].sum + [(4, 4)].sum + [(4, 4)].sum + [(5, 5)].sum + [(5, 5)].sum ) ∧
      (shelf2.sum = [(5, 5)].sum + [(4, 4)].sum + [(4, 4)].sum + [(3, 3)].sum + [(3, 3)].sum ) ∧
      (shelf1.length = 7) ∧ 
      (shelf2.length = 7) ∧
      (∃ n m, 
        (shelf1.sum = n + m) ∧ 
        (shelf2.sum = n + m)) ∧
      (∃ x, 
        x = shelf1 ++ shelf2 ∧
        (x.nuts % 2 = 0)) :=
begin
  sorry
end

end Belchonok_arrange_bags_l716_716633


namespace sum_of_coefficients_eq_neg36_l716_716996

theorem sum_of_coefficients_eq_neg36 (c : ℝ) : 
  let expr := -(5 - c) * (c - 2 * (c - 5))
  let expanded_expr := -c^2 + 15*c - 50 in
  (finset.sum (finset.image (λ i, expanded_expr.coeff i) (expanded_expr.support)) (λ i, expanded_expr.coeff i)) = -36 := 
by
  -- sorry for skipping the proof
  sorry

end sum_of_coefficients_eq_neg36_l716_716996


namespace area_of_trapezoid_EFGH_l716_716379

-- Definitions of vertices of the trapezoid
def E : ℝ × ℝ := (2, -3)
def F : ℝ × ℝ := (2, 2)
def G : ℝ × ℝ := (6, 8)
def H : ℝ × ℝ := (6, 0)

-- Function to calculate the area of a trapezoid given the coordinates of its vertices
def trapezoid_area (A B C D : ℝ × ℝ) : ℝ := 
  let height := (D.1 - A.1).abs
  let base1 := (B.2 - A.2).abs
  let base2 := (C.2 - D.2).abs
  (1 / 2) * (base1 + base2) * height

-- The theorem to be proven
theorem area_of_trapezoid_EFGH : trapezoid_area E F G H = 26 := by
  sorry

end area_of_trapezoid_EFGH_l716_716379


namespace regular_polygon_sides_l716_716124

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716124


namespace stoner_inequality_l716_716431

def positive_reals := {x : ℝ // x > 0}

theorem stoner_inequality
  {a b c : positive_reals}
  (h : a^2014 + b^2014 + c^2014 + a * b * c = 4) :
  (a^2013 + b^2013 - c) / c^2013 + 
  (b^2013 + c^2013 - a) / a^2013 + 
  (c^2013 + a^2013 - b) / b^2013 ≥ 
  a^2012 + b^2012 + c^2012 :=
sorry

end stoner_inequality_l716_716431


namespace problem_253_base2_l716_716980

def decimal_to_binary (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec f (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc.reverse
      else f (n / 2) ((n % 2) :: acc)
    in f n []

def count_digits (lst : list ℕ) : ℕ × ℕ :=
  lst.foldr (λ d (acc : ℕ × ℕ), if d = 0 then (acc.1 + 1, acc.2) else (acc.1, acc.2 + 1)) (0, 0)

theorem problem_253_base2 :
  let bits := decimal_to_binary 253
  let (x, y) := count_digits bits
  y - x = 6 :=
by
  sorry

end problem_253_base2_l716_716980


namespace number_of_ones_minus_zeros_in_253_binary_l716_716985

def binary_representation (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | value => (binary_representation (value / 2)) * 10 + (value % 2)

-- The binary representation of 253 is 11111101
theorem number_of_ones_minus_zeros_in_253_binary : 
  let x := 1
  let y := 7
  y - x = 6 :=
by
  -- Definitions for x, y
  let x := 1
  let y := 7
  -- Required proof
  sorry

end number_of_ones_minus_zeros_in_253_binary_l716_716985


namespace cone_base_circumference_l716_716225

theorem cone_base_circumference (V h : ℝ) (π : ℝ) (π_pos : 0 < π)
  (volume_eq : V = 16 * π) (height_eq : h = 6) :
  ∃ C, C = 4 * Real.sqrt 2 * π :=
by
  let r := 2 * Real.sqrt 2
  let C := 2 * π * r
  have volume : V = (1/3) * π * r^2 * h := by
    unfold r
    unfold h
    linarith
  exact ⟨4 * Real.sqrt 2 * π, rfl⟩

end cone_base_circumference_l716_716225


namespace days_provisions_initially_meant_l716_716599

theorem days_provisions_initially_meant (x : ℕ) (h1 : 250 * x = 200 * 50) : x = 40 :=
by sorry

end days_provisions_initially_meant_l716_716599


namespace total_palm_trees_l716_716594

theorem total_palm_trees (forest_palm_trees : ℕ) (h_forest : forest_palm_trees = 5000) 
    (h_ratio : 3 / 5) : forest_palm_trees + (forest_palm_trees - (h_ratio * forest_palm_trees).to_nat) = 7000 :=
by
  sorry

end total_palm_trees_l716_716594


namespace shift_parabola_two_units_right_l716_716953

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end shift_parabola_two_units_right_l716_716953


namespace increasing_function_interval_l716_716623

open Real

-- Define the interval
def interval := Set.Ioo (Real.pi / 2) Real.pi

-- Define the functions
def f1 (x : ℝ) : ℝ := Real.sin x
def f2 (x : ℝ) : ℝ := Real.cos x
def f3 (x : ℝ) : ℝ := Real.tan x
def f4 (x : ℝ) : ℝ := -Real.tan x

-- The theorem stating that f3 is increasing in the interval
theorem increasing_function_interval (h : x ∈ interval) : 
  (∀ x y, x < y → f3 x < f3 y) :=
sorry

end increasing_function_interval_l716_716623


namespace total_animals_l716_716523

theorem total_animals (initial_elephants initial_hippos : ℕ) 
  (ratio_female_hippos : ℚ)
  (births_per_female_hippo : ℕ)
  (newborn_elephants_diff : ℕ)
  (he : initial_elephants = 20)
  (hh : initial_hippos = 35)
  (rfh : ratio_female_hippos = 5 / 7)
  (bpfh : births_per_female_hippo = 5)
  (ned : newborn_elephants_diff = 10) :
  ∃ (total_animals : ℕ), total_animals = 315 :=
by sorry

end total_animals_l716_716523


namespace sum_of_odd_multiples_of_5_between_200_and_400_l716_716893

theorem sum_of_odd_multiples_of_5_between_200_and_400 :
  (∑ k in finset.filter (λ k, k % 2 = 1 ∧ k % 5 = 0) (finset.Ico 201 400), k) = 6000 :=
by
  sorry

end sum_of_odd_multiples_of_5_between_200_and_400_l716_716893


namespace sequence_divisible_103_count_l716_716356

theorem sequence_divisible_103_count (n : ℕ) (h : 1 ≤ n ∧ n ≤ 2023) :
  (∑ k in finset.range(n+1), if 103 ∣ (10^k - 1) then 1 else 0) = 404 :=
sorry

end sequence_divisible_103_count_l716_716356


namespace cosine_value_lambda_value_l716_716727

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (4, 3)
def vec_b : ℝ × ℝ := (1, -1)

-- Function to calculate the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to calculate the norm of a vector
def norm (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 ^ 2 + u.2 ^ 2)

-- Define the cosine of the angle between vectors a and b
def cos_theta (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (norm u * norm v)

-- The first statement: proving the cosine value
theorem cosine_value : cos_theta vec_a vec_b = real.sqrt 2 / 10 :=
by sorry

-- Define the vectors based on the given conditions for the second part of the problem
def vector1 : ℝ × ℝ := (12, 9) + (4, -4)
def vector2 (lam : ℝ) : ℝ × ℝ := (4 * lam - 1, 3 * lam + 1)

-- The second statement: proving the value of λ
theorem lambda_value (lam : ℝ) : vector1 = vector2 lam → lam = -3 / 4 :=
by sorry

end cosine_value_lambda_value_l716_716727


namespace find_f_neg_a_l716_716453

def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

theorem find_f_neg_a (h : f a = 11) : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l716_716453


namespace g_five_eq_one_l716_716499

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x y : ℝ) : g (x * y) = g x * g y
axiom g_zero_ne_zero : g 0 ≠ 0

theorem g_five_eq_one : g 5 = 1 := by
  sorry

end g_five_eq_one_l716_716499


namespace find_ABC_l716_716793

theorem find_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (hA : A < 5) (hB : B < 5) (hC : C < 5) (h_nonzeroA : A ≠ 0) (h_nonzeroB : B ≠ 0) (h_nonzeroC : C ≠ 0)
  (h4 : B + C = 5) (h5 : A + 1 = C) (h6 : A + B = C) : A = 3 ∧ B = 1 ∧ C = 4 := 
by
  sorry

end find_ABC_l716_716793


namespace sqrt_expression_simplification_l716_716246

-- Define the problem and the conditions needed to prove the statement
theorem sqrt_expression_simplification :
  real.sqrt 14 * real.sqrt 7 - real.sqrt 2 = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_simplification_l716_716246


namespace vectors_parallel_l716_716728

def vec_a (e1 e2 : ℝ) : ℝ × ℝ := (3 * e1, -4 * e2)
def vec_b (e1 e2 n : ℝ) : ℝ × ℝ := ((1 - n) * e1, 3 * n * e2)

theorem vectors_parallel (n : ℝ) (e1 e2 : ℝ) :
  let a := vec_a e1 e2 in
  let b := vec_b e1 e2 n in
  (e1 = e2) ∨ (a = (3 / (1 - n), -4 / (3 * n)) ∨ n = -4 / 5) :=
sorry

end vectors_parallel_l716_716728


namespace regular_polygon_sides_l716_716172

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716172


namespace regular_polygon_sides_l716_716090

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716090


namespace regular_polygon_sides_l716_716113

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716113


namespace determine_bases_l716_716372

noncomputable section

def satisfies_base_a_conditions (a : ℕ) (F G : ℚ) : Prop :=
  (F = (0.373737... : ℚ)) ∧ (G = (0.737373... : ℚ))

def satisfies_base_b_conditions (b : ℕ) (F G : ℚ) : Prop :=
  (F = (0.252525... : ℚ)) ∧ (G = (0.525252... : ℚ))

theorem determine_bases (F G : ℚ) (a b : ℕ) :
  satisfies_base_a_conditions a F G →
  satisfies_base_b_conditions b F G →
  a = 11 ∧ b = 8 := by
  sorry

end determine_bases_l716_716372


namespace gcf_48_160_120_l716_716023

theorem gcf_48_160_120 : Nat.gcd (Nat.gcd 48 160) 120 = 8 := by
  sorry

end gcf_48_160_120_l716_716023


namespace part_I_sin_cos_tan_part_II_sin_cos_half_angle_l716_716422

namespace Trigonometry

variables (α : ℝ)
noncomputable def x : ℝ := 4 / 5
noncomputable def y : ℝ := 3 / 5
noncomputable def r : ℝ := 1

theorem part_I_sin_cos_tan (hP : (x α)^2 + (y α)^2 = r^2) :
  sin α = y α ∧ cos α = x α ∧ tan α = y α / x α :=
by sorry

theorem part_II_sin_cos_half_angle :
  (sin (α / 2) - cos (α / 2))^2 = 2 / 5 :=
by sorry

end Trigonometry

end part_I_sin_cos_tan_part_II_sin_cos_half_angle_l716_716422


namespace intersection_A_B_intersection_A_complement_B_C_l716_716724

open Set

variable {U : Set ℝ} {A B C : Set ℝ}

def U : Set ℝ := univ
def A : Set ℝ := {x | x < -3 ∨ x > 3}
def B : Set ℝ := {x | x < -1 ∨ x > 7}
def C : Set ℝ := {x | -2 < x ∧ x < 6}

-- Prove that A ∩ B = {x | x < -3 ∨ x > 7}
theorem intersection_A_B : A ∩ B = {x | x < -3 ∨ x > 7} :=
by sorry

-- Prove that A ∩ (U \ (B ∩ C)) = {x | x < -3 ∨ x > 3}
theorem intersection_A_complement_B_C : A ∩ (U \ (B ∩ C)) = {x | x < -3 ∨ x > 3} :=
by sorry

end intersection_A_B_intersection_A_complement_B_C_l716_716724


namespace max_of_f_on_interval_l716_716506

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x

theorem max_of_f_on_interval :
  ∀ x ∈ set.Icc (-1 : ℝ) 2, f x ≤ 13 :=
begin
  intros x hx,
  sorry
end

end max_of_f_on_interval_l716_716506


namespace remainder_of_large_number_l716_716251

noncomputable def X (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 32
  | 6 => 64
  | 7 => 128
  | 8 => 256
  | 9 => 512
  | 10 => 1024
  | 11 => 2048
  | 12 => 4096
  | 13 => 8192
  | _ => 0

noncomputable def concatenate_X (k : ℕ) : ℕ :=
  if k = 5 then 
    100020004000800160032
  else if k = 11 then 
    100020004000800160032006401280256051210242048
  else if k = 13 then 
    10002000400080016003200640128025605121024204840968192
  else 
    0

theorem remainder_of_large_number :
  (concatenate_X 13) % (concatenate_X 5) = 40968192 :=
by
  sorry

end remainder_of_large_number_l716_716251


namespace q_investment_time_l716_716565

-- Definitions from the conditions
def investment_ratio_p_q : ℚ := 7 / 5
def profit_ratio_p_q : ℚ := 7 / 13
def time_p : ℕ := 5

-- Problem statement
theorem q_investment_time
  (investment_ratio_p_q : ℚ)
  (profit_ratio_p_q : ℚ)
  (time_p : ℕ)
  (hpq_inv : investment_ratio_p_q = 7 / 5)
  (hpq_profit : profit_ratio_p_q = 7 / 13)
  (ht_p : time_p = 5) : 
  ∃ t_q : ℕ, 35 * t_q = 455 :=
sorry

end q_investment_time_l716_716565


namespace total_dogs_equation_l716_716255

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end total_dogs_equation_l716_716255


namespace travel_distance_l716_716066

variables (speed time : ℕ) (distance : ℕ)

theorem travel_distance (hspeed : speed = 75) (htime : time = 4) : distance = speed * time → distance = 300 :=
by
  intros hdist
  rw [hspeed, htime] at hdist
  simp at hdist
  assumption

end travel_distance_l716_716066


namespace miles_per_tankful_city_l716_716917

theorem miles_per_tankful_city (miles_highway_tankful : ℕ) (diff_mpg : ℕ) (mpg_city : ℕ) :
  miles_highway_tankful = 462 →
  diff_mpg = 18 →
  mpg_city = 48 →
  let mpg_highway := mpg_city + diff_mpg in
  let tank_size := miles_highway_tankful / mpg_highway in
  mpg_city * tank_size = 336 :=
by
  intros h₁ h₂ h₃
  let mpg_highway := 48 + 18
  let tank_size := 462 / mpg_highway
  simp [h₁, h₂, h₃, mpg_highway, tank_size]
  sorry

end miles_per_tankful_city_l716_716917


namespace b_is_geometric_and_general_formulas_l716_716766

noncomputable def a : ℕ → ℤ
| 1       := 1
| (n + 1) := 2 * a n + 1

def b (n : ℕ) : ℤ := a n + 1

theorem b_is_geometric_and_general_formulas :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 2 * b n) ∧ (∀ n : ℕ, a n = 2^n - 1) ∧ (∀ n : ℕ, b n = 2^n) :=
by
  sorry

end b_is_geometric_and_general_formulas_l716_716766


namespace basketball_game_points_half_l716_716380

theorem basketball_game_points_half (a d b r : ℕ) (h_arith_seq : a + (a + d) + (a + 2 * d) + (a + 3 * d) ≤ 100)
    (h_geo_seq : b + b * r + b * r^2 + b * r^3 ≤ 100)
    (h_win_by_two : 4 * a + 6 * d = b * (1 + r + r^2 + r^3) + 2) :
    (a + (a + d)) + (b + b * r) = 14 :=
sorry

end basketball_game_points_half_l716_716380


namespace interest_rate_for_second_investment_l716_716270

variables (total_investment desired_interest first_principal second_principal first_rate second_rate : ℝ)
          (time : ℝ)
          (interest_from_first second_interest_needed : ℝ)

-- Conditions
def total_investment := 10000
def desired_interest := 980
def first_principal := 6000
def first_rate := 0.09
def time := 1

def interest_from_first := first_principal * first_rate * time
def second_interest_needed := desired_interest - interest_from_first

def second_principal := total_investment - first_principal
def second_rate := second_interest_needed / (second_principal * time)

theorem interest_rate_for_second_investment :
  second_rate = 0.11 :=
by
  unfold second_rate second_interest_needed interest_from_first time first_rate first_principal desired_interest total_investment
  sorry

end interest_rate_for_second_investment_l716_716270


namespace rival_awards_l716_716781

theorem rival_awards (jessie_multiple : ℕ) (scott_awards : ℕ) (rival_multiple : ℕ) 
  (h1 : jessie_multiple = 3) 
  (h2 : scott_awards = 4) 
  (h3 : rival_multiple = 2) 
  : (rival_multiple * (jessie_multiple * scott_awards) = 24) :=
by 
  sorry

end rival_awards_l716_716781


namespace delta_epsilon_time_l716_716885

-- The definitions extracted from the given conditions
def time_taken_together (D E : ℝ) : ℝ := (1 / D + 1 / E)⁻¹
def time_delta_epsilon (D E : ℝ) : ℝ := D - 4  -- (condition 1)
def time_delta_epsilon_gamma (D E : ℝ) : ℝ := (D - 4) / 2  -- (condition 3)

-- Mathematically equivalent proof problem in Lean 4 statement
theorem delta_epsilon_time (D E : ℝ) (h₁ : time_delta_epsilon D E = D - 4) (h₂ : time_delta_epsilon D E = E - 3)
  (h₃ : time_delta_epsilon_gamma D E = (D - 4) / 2) :
  time_taken_together D E = 42 / 13 := sorry

end delta_epsilon_time_l716_716885


namespace area_of_ABCD_l716_716756

-- Definitions of the conditions
structure Quadrilateral (α : Type) :=
(A B C D : α)
(AB BC CD DA : ℝ)
(angle_A : ℝ)
(angle_D : ℝ)

-- Given quadrilateral ABCD with specific side lengths and angles
def quadrilateral_ABCD : Quadrilateral ℝ :=
{ A := 0, B := 1, C := 2, D := 3,
  AB := 5, BC := 7, CD := 3, DA := 4,
  angle_A := 120, 
  angle_D := 120 }

-- Statement of the math proof problem
theorem area_of_ABCD : 
  let ang_rad := (120 : ℝ) * Real.pi / 180 in
  let sin_120 := Real.sin ang_rad in
  (1 / 2 * quadrilateral_ABCD.AB * quadrilateral_ABCD.BC * sin_120 +
  1 / 2 * quadrilateral_ABCD.DA * quadrilateral_ABCD.CD * sin_120) = 
  (47 * Real.sqrt 3 / 4) :=
by
  sorry

end area_of_ABCD_l716_716756


namespace ellipse_equation_and_range_OP_l716_716697

theorem ellipse_equation_and_range_OP
  {a b : ℝ} (h_ab_pos : 0 < a ∧ a > b ∧ b > 0)
  (h_area : 2 * a * b = 4 * sqrt 3)
  (h_min_dist : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) → sqrt ((x - 1)^2 + y^2) ≥ 1)
  : (a = 2 ∧ b = sqrt(3) ∧ 
      (∀ (k m : ℝ), 0 ≤ k ∧ k ≤ sqrt 3 ∧ ∀ x y, 
      (y = k * x + m) → (x^2 / 4 + y^2 / 3 = 1) → 
      sqrt (x^2 + y^2) ∈ set.Icc (sqrt 3) (sqrt (95) / 5))) :=
sorry

end ellipse_equation_and_range_OP_l716_716697


namespace volume_of_pyramid_l716_716834

-- Define the regular octagon and the pyramid
structure Octagon := (side_length : ℝ)

structure Pyramid := (base : Octagon) (apex : ℝ)

-- Given conditions
def octagon := Octagon.mk (10 / Real.sqrt 2)
def pyramid := Pyramid.mk octagon (5 * Real.sqrt 3)

-- Theorem statement
theorem volume_of_pyramid (h_base : Octagon) (h_apex : ℝ) (h_equilateral : ∀ PAC, PAC = EquilateralTriangle 10) :
  1 / 3 * (8 * (5 * Real.sqrt 2) ^ 2 * Real.sqrt 3 / 4) * (5 * Real.sqrt 3) = 1500 :=
by
  sorry

end volume_of_pyramid_l716_716834


namespace buckets_fill_drum_l716_716035

noncomputable def bucket_turns (cq cp : ℕ) (hp : cp = 3 * cq) (turns_p : ℕ) (drum_capacity : ℕ) (hp_fill : 60 * cp = drum_capacity)
    (turns_q : ℕ) (combined_capacity : ℕ) (hp_combined : combined_capacity = cp + cq) : Prop :=
  turns_q * combined_capacity = drum_capacity

theorem buckets_fill_drum :
  ∀ (cq cp : ℕ), cp = 3 * cq →
  ∀ (turns_p : ℕ) (drum_capacity : ℕ), 60 * cp = drum_capacity →
  ∀ (turns_q : ℕ) (combined_capacity : ℕ), combined_capacity = cp + cq →
  bucket_turns cq cp (rfl) turns_p drum_capacity (rfl) turns_q combined_capacity (rfl) :=
by
  sorry

end buckets_fill_drum_l716_716035


namespace regular_polygon_sides_l716_716193

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716193


namespace solve_for_a_when_diamond_eq_6_l716_716258

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem solve_for_a_when_diamond_eq_6 (a : ℝ) : diamond a 3 = 6 → a = 8 :=
by
  intros h
  simp [diamond] at h
  sorry

end solve_for_a_when_diamond_eq_6_l716_716258


namespace regular_polygon_sides_l716_716074

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716074


namespace rational_fraction_l716_716694

open Set

-- Definitions based on given conditions
def S : Set ℝ := { x : ℝ | 0 < x }

axiom S_nonempty : S.nonempty

axiom cond : ∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → (a^3 + b^3 + c^3 - 3*a*b*c) ∈ ℚ

-- Statement to prove
theorem rational_fraction (a b : ℝ) (ha : a ∈ S) (hb : b ∈ S) : (a - b) / (a + b) ∈ ℚ :=
sorry

end rational_fraction_l716_716694


namespace last_digit_crossed_out_l716_716757

-- Definition of the 100-digit sequence
def sequence := "1234567890".replicate 10

-- Statement of the theorem
theorem last_digit_crossed_out : 
  (final_digit_removed sequence = "4") :=
sorry

end last_digit_crossed_out_l716_716757


namespace triangle_area_l716_716019

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end triangle_area_l716_716019


namespace pair_C_does_not_yield_roots_l716_716267

theorem pair_C_does_not_yield_roots :
  ¬∃ x, (x - 1 = 0 ∨ x - 3 = 0) ∧ (x = x - 1 ∧ x = x - 3) := 
begin
  sorry
end

end pair_C_does_not_yield_roots_l716_716267


namespace mod_inverse_identity_l716_716635

theorem mod_inverse_identity : 
  (1 / 5 + 1 / 5^2) % 31 = 26 :=
by
  sorry

end mod_inverse_identity_l716_716635


namespace correct_chip_arrangement_l716_716627

-- Definitions based on the conditions from the problem
def flat_plane : Prop := true -- Assume a perfectly flat plane

def tangent (chip1 chip2 : Type) : Prop := 
  -- Assuming a generic definition of tangency between two chips.
  sorry 

def arrangement (chips : list Type) (centerpiece : Type) : Prop :=
  (∀ chip ∈ chips, ∃ chip_indications : (α : Type), 
    tangent chip centerpiece ∧ 
    ∃ (c1 c2 : Type), 
    c1 ≠ chip ∧ c2 ≠ chip ∧ c1 ∈ chips ∧ c2 ∈ chips ∧ tangent chip c1 ∧ tangent chip c2)

-- The theorem statement using the given conditions and answer
theorem correct_chip_arrangement :
  ∀ (chips : list Type) (centerpiece : Type),
  flat_plane →
  arrangement chips centerpiece →
  chips.length = 12 := 
  sorry

end correct_chip_arrangement_l716_716627


namespace part1_part2_part3_l716_716815

noncomputable def P (k : ℕ) : ℝ :=
if h : k ∈ {1, 2, 3, 4, 5} then 1 / 15 * k else 0

theorem part1 : ∀ k ∈ {1, 2, 3, 4, 5}, P k = 1 / 15 * k := by
  intros k hk
  simp [P, hk]
  sorry

theorem part2 : P (3) + P (4) + P (5) = 4 / 5 := by
  simp [P]
  linarith
  sorry

theorem part3 : P 1 + P 2 + P 3 = 2 / 5 := by
  simp [P]
  linarith
  sorry

end part1_part2_part3_l716_716815


namespace walking_time_l716_716928

theorem walking_time (distance_walking_rate : ℕ) 
                     (distance : ℕ)
                     (rest_distance : ℕ) 
                     (rest_time : ℕ) 
                     (total_walking_time : ℕ) : 
  distance_walking_rate = 10 → 
  rest_distance = 10 → 
  rest_time = 7 → 
  distance = 50 → 
  total_walking_time = 328 → 
  total_walking_time = (distance / distance_walking_rate) * 60 + ((distance / rest_distance) - 1) * rest_time :=
by
  sorry

end walking_time_l716_716928


namespace range_of_a_l716_716686

theorem range_of_a (a : ℝ) :
  (¬ (∀ x : ℝ, ax^2 - x + a / 36 > 0) ∨ ¬ (∀ x : ℝ, 2^x - 4^x < 2a - 3/4)) → a ≤ 3 :=
sorry

end range_of_a_l716_716686


namespace find_d_l716_716707

theorem find_d (d : ℝ) (h1 : 0 < d) (h2 : d < 90) (h3 : Real.cos 16 = Real.sin 14 + Real.sin d) : d = 46 :=
by
  sorry

end find_d_l716_716707


namespace smallest_range_of_sample_l716_716936

theorem smallest_range_of_sample : 
  ∀ (x : list ℝ), 
  x.length = 5 ∧ list.sum x = 75 ∧ x.nthLe 2 (by simp) = 18 → 
  ∃ y z, y ∈ x ∧ z ∈ x ∧ (∀ w, w ∈ x → y ≤ w ∧ w ≤ z) ∧ z - y = 3 :=
by
  sorry

end smallest_range_of_sample_l716_716936


namespace correct_options_l716_716894

section

  -- Define the domain and the function for each option

  -- Option A
  def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

  -- Option B
  def h (x : ℝ) (a : ℝ) : ℝ := a ^ (x - 1) - 2
  def point1 : (ℝ × ℝ) := (1, -1)

  -- Option C
  def f_c (x : ℝ) : ℝ := Real.sqrt (1 - x^2)
  def g (x : ℝ) : ℝ := Real.sqrt (1 - x) * Real.sqrt (1 + x)

  -- Option D
  def y (x : ℝ) : ℝ := 2 * x - Real.sqrt (1 + x)

  -- Proving the correct options
  theorem correct_options (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
    (h 1 a = -1) ∧ (∀ x, -1 ≤ x ∧ x ≤ 1 → f_c x = g x) :=
  by
    sorry

end

end correct_options_l716_716894


namespace katerina_weight_correct_l716_716961

-- We define the conditions
def total_weight : ℕ := 95
def alexa_weight : ℕ := 46

-- Define the proposition to prove: Katerina's weight is the total weight minus Alexa's weight, which should be 49.
theorem katerina_weight_correct : (total_weight - alexa_weight = 49) :=
by
  -- We use sorry to skip the proof.
  sorry

end katerina_weight_correct_l716_716961


namespace businessmen_neither_coffee_nor_tea_l716_716631

theorem businessmen_neither_coffee_nor_tea :
  ∀ (total_count coffee tea both neither : ℕ),
    total_count = 30 →
    coffee = 15 →
    tea = 13 →
    both = 6 →
    neither = total_count - (coffee + tea - both) →
    neither = 8 := 
by
  intros total_count coffee tea both neither ht hc ht2 hb hn
  rw [ht, hc, ht2, hb] at hn
  simp at hn
  exact hn

end businessmen_neither_coffee_nor_tea_l716_716631


namespace josh_can_buy_donuts_in_15_ways_l716_716573

-- Define the conditions
def total_donuts : ℕ := 8
def types : ℕ := 5
def min_per_type_i : Fin types → ℕ := 
  λ i, if i = 0 then 2 else 1
def additional_donuts_needed : ℕ :=
  total_donuts - (min_per_type_i 0 + min_per_type_i 1 + min_per_type_i 2 +
                  min_per_type_i 3 + min_per_type_i 4)

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.choose (n+k-1) (k-1)

-- State the main theorem
theorem josh_can_buy_donuts_in_15_ways :
  combinations additional_donuts_needed types = 15 := by
  sorry

end josh_can_buy_donuts_in_15_ways_l716_716573


namespace product_multiple_of_105_probability_l716_716375

theorem product_multiple_of_105_probability :
  let s := { 3, 5, 15, 21, 35, 42, 70 }
  ∃ (a b : ℕ) (h1 : a ≠ b) (h2 : a ∈ s) (h3 : b ∈ s),
  (a * b) % 105 = 0 ∧ 
  let total_pairs := 21 in
  total_pairs = finset.card (finset.ssubsets s 2) ∧
  let successful_pairs := 8 in
  (successful_pairs : ℚ) / (total_pairs : ℚ) = 8 / 21 :=
sorry

end product_multiple_of_105_probability_l716_716375


namespace regular_pentagon_cannot_tessellate_l716_716238

-- Definitions of polygons
def is_regular_triangle (angle : ℝ) : Prop := angle = 60
def is_square (angle : ℝ) : Prop := angle = 90
def is_regular_pentagon (angle : ℝ) : Prop := angle = 108
def is_hexagon (angle : ℝ) : Prop := angle = 120

-- Tessellation condition
def divides_evenly (a b : ℝ) : Prop := ∃ k : ℕ, b = k * a

-- The main statement
theorem regular_pentagon_cannot_tessellate :
  ¬ divides_evenly 108 360 :=
sorry

end regular_pentagon_cannot_tessellate_l716_716238


namespace tom_reads_pages_l716_716654

-- Definition of conditions
def initial_speed : ℕ := 12   -- pages per hour
def speed_factor : ℕ := 3
def time_period : ℕ := 2     -- hours

-- Calculated speeds
def increased_speed (initial_speed speed_factor : ℕ) : ℕ := initial_speed * speed_factor
def total_pages (increased_speed time_period : ℕ) : ℕ := increased_speed * time_period

-- Theorem statement
theorem tom_reads_pages :
  total_pages (increased_speed initial_speed speed_factor) time_period = 72 :=
by
  -- Omitting proof as only theorem statement is required
  sorry

end tom_reads_pages_l716_716654


namespace similar_triangle_shortest_side_l716_716606

theorem similar_triangle_shortest_side 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) (c₂ : ℝ) (k : ℝ)
  (h₁ : a₁ = 15) 
  (h₂ : c₁ = 39) 
  (h₃ : c₂ = 117) 
  (h₄ : k = c₂ / c₁) 
  (h₅ : k = 3) 
  (h₆ : a₂ = a₁ * k) :
  a₂ = 45 := 
by {
  sorry -- proof is not required
}

end similar_triangle_shortest_side_l716_716606


namespace rival_awards_eq_24_l716_716783

-- Definitions
def Scott_awards : ℕ := 4
def Jessie_awards (scott: ℕ) : ℕ := 3 * scott
def Rival_awards (jessie: ℕ) : ℕ := 2 * jessie

-- Theorem to prove
theorem rival_awards_eq_24 : Rival_awards (Jessie_awards Scott_awards) = 24 := by
  sorry

end rival_awards_eq_24_l716_716783


namespace find_x_l716_716992

-- Define the lengths of the edges in the pentagon
variables (a b c d x : ℝ)
-- Define the value of θ as 72 degrees in radians
def θ : ℝ := 2 * real.pi / 5
-- Define the cosine of 72 degrees
def cos_θ : ℝ := real.cos θ

-- Define the Law of Cosines equation
def law_of_cosines := a^2 + d^2 - 2 * a * d * cos_θ

-- Conditions (lengths of the sides of the pentagon):
axiom ha : a = 3
axiom hb : b = 7
axiom hc : c = 9
axiom hd : d = 6
axiom hx : x = sqrt (a^2 + d^2 - 2 * a * d * cos_θ)

-- The Prop we want to prove
theorem find_x : 
  x = 6 := 
by
  -- we leave the proof to be filled out
  sorry

end find_x_l716_716992


namespace number_of_acute_triangles_l716_716263

theorem number_of_acute_triangles : 
  ∃ (n : ℕ), 
    (∀ (x : ℕ), 23 ≤ x ∧ x < 30 → ∃ (y : ℕ), 4 * x + y = 180 ∧ y < 90) ∧
    n = (finset.range' 23 (30-23)).filter (λ x, ∃ y, 4 * x + y = 180 ∧ y < 90).card :=
sorry

end number_of_acute_triangles_l716_716263


namespace probability_two_i_adjacent_l716_716558

theorem probability_two_i_adjacent :
  let letters := ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'i'],
      total_codes := 10!,
      favorable_codes := 9! * 2 :=
  (favorable_codes / total_codes) = (1 / 5) :=
by sorry

end probability_two_i_adjacent_l716_716558


namespace problem_statement_l716_716320

noncomputable theory

open Real

theorem problem_statement (α β : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : sqrt 3 * cos (α / 2)^2 + sqrt 2 * sin (β /2)^2 = sqrt 2 / 2 + sqrt 3 / 2)
  (h4 : sin (2017 * π - α) = sqrt 2 * cos (5 * π / 2 - β)) 
  : α + β = 5 * π / 12 := 
  sorry

end problem_statement_l716_716320


namespace exists_pairwise_coprime_product_of_two_consecutive_integers_l716_716481

theorem exists_pairwise_coprime_product_of_two_consecutive_integers (n : ℕ) (h : 0 < n) :
  ∃ (a : Fin n → ℕ), (∀ i, 2 ≤ a i) ∧ (Pairwise (IsCoprime on fun i => a i)) ∧ (∃ k : ℕ, (Finset.univ.prod a) - 1 = k * (k + 1)) := 
sorry

end exists_pairwise_coprime_product_of_two_consecutive_integers_l716_716481


namespace number_of_rescue_team_arrangements_l716_716975

theorem number_of_rescue_team_arrangements :
  ∃ (A B C : Finset ℕ), 
    (∀ x ∈ range 6, x ∈ A ∨ x ∈ B ∨ x ∈ C) ∧               -- Each rescue team goes to exactly one site
    (1 ≤ A.card) ∧ (1 ≤ B.card) ∧ (1 ≤ C.card) ∧            -- Each site has at least one team
    (2 ≤ A.card) ∧                                            -- Site A has at least 2 teams
    A.disjoint B ∧ (0 ∉ A ∩ B)                                 -- Teams A and B cannot be at the same site
  → A.card + B.card + C.card = 6 →                                       -- Total of 6 rescue teams
    ∑ x in (A.ssubset ∘ B.ssubset ∘ C.ssubset), 1 = 266 :=            -- Number of different valid arrangements is 266
sorry

end number_of_rescue_team_arrangements_l716_716975


namespace width_of_track_l716_716935

theorem width_of_track (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 15 * Real.pi) : r1 - r2 = 7.5 := 
by
  sorry

end width_of_track_l716_716935


namespace mean_and_variance_unchanged_with_replacement_l716_716410

def S : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

structure Replacement :=
  (a b c : ℤ)
  (is_in_S : a ∈ S)
  (b_plus_c_eq_a : b + c = a)
  (b_sq_plus_c_sq_minus_a_sq_eq_ten : b^2 + c^2 - a^2 = 10)

def replacements : List Replacement :=
  [ { a := 4, b := -1, c := 5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num },
    { a := -4, b := 1, c := -5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num } ]

theorem mean_and_variance_unchanged_with_replacement (r : Replacement) :
  -- assuming r comes from the list of replacements
  ∃ (r ∈ replacements), r = r :=
begin
  sorry
end

end mean_and_variance_unchanged_with_replacement_l716_716410


namespace functional_equation_solution_l716_716791

noncomputable def quadratic_polynomial (P : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x + c

theorem functional_equation_solution (P : ℝ → ℝ) (f : ℝ → ℝ)
  (h_poly : quadratic_polynomial P)
  (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)
  (h_preserves_poly : ∀ x : ℝ, f (P x) = f x) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_solution_l716_716791


namespace find_values_for_cond_l716_716266

-- Define the main conditions in terms of functions and equality
def condition_eq (f g h : ℚ) (x : ℚ) :=
  (D * x - 20) / (x^2 - 10 * x + 21) = (C / (x - 3)) + (5 / (x - 7))

noncomputable def D := 40 / 7
noncomputable def C := 5 / 7

def proof_problem : Prop :=
  D = 40 / 7 ∧ C = 5 / 7 ∧ C + D = 45 / 7

-- State the main problem
theorem find_values_for_cond : ∃ D C, proof_problem :=
  by sorry

end find_values_for_cond_l716_716266


namespace regular_polygon_sides_l716_716095

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716095


namespace usual_time_is_42_l716_716012

noncomputable def usual_time_to_school (R T : ℝ) := T * R
noncomputable def improved_time_to_school (R T : ℝ) := ((7/6) * R) * (T - 6)

theorem usual_time_is_42 (R T : ℝ) :
  (usual_time_to_school R T) = (improved_time_to_school R T) → T = 42 :=
by
  sorry

end usual_time_is_42_l716_716012


namespace gcd_of_differences_is_10_l716_716561

theorem gcd_of_differences_is_10 (a b c : ℕ) (h1 : b > a) (h2 : c > b) (h3 : c > a)
  (h4 : b - a = 20) (h5 : c - b = 50) (h6 : c - a = 70) : Int.gcd (b - a) (Int.gcd (c - b) (c - a)) = 10 := 
sorry

end gcd_of_differences_is_10_l716_716561


namespace derivative_at_zero_l716_716684

def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem derivative_at_zero : deriv f 0 = 2 := 
by 
  sorry

end derivative_at_zero_l716_716684


namespace Bella_increase_success_rate_l716_716243

-- Definitions based on conditions
def initial_attempts : ℕ := 8
def initial_successes : ℕ := 3
def next_attempts : ℕ := 28
def next_success_rate : ℚ := 3 / 4

-- The Lean statement to prove the corresponding proof problem
theorem Bella_increase_success_rate : 
  let total_attempts := initial_attempts + next_attempts,
  let successful_initial := initial_successes,
  let successful_next := next_success_rate * next_attempts,
  let total_successful := successful_initial + successful_next,
  let initial_rate := (initial_successes : ℚ) / initial_attempts,
  let new_rate := total_successful / total_attempts,
  let rate_increase := new_rate - initial_rate
in (rate_increase * 100).natAbs = 29 := 
by sorry

end Bella_increase_success_rate_l716_716243


namespace unique_root_value_l716_716288

theorem unique_root_value {x n : ℝ} (h : (15 - n) = 15 - (35 / 4)) :
  (x + 5) * (x + 3) = n + 3 * x → n = 35 / 4 :=
sorry

end unique_root_value_l716_716288


namespace angle_between_slanted_line_and_plane_l716_716377

theorem angle_between_slanted_line_and_plane (L l : ℝ) (h : ℝ) 
  (hL : L = 2 * l) (h_triangle : L^2 = l^2 + h^2) : 
  ∃ θ : ℝ, θ = 60 ∧ tan θ = sqrt 3 := 
by
  sorry

end angle_between_slanted_line_and_plane_l716_716377


namespace regular_polygon_sides_l716_716121

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716121


namespace squirrel_rise_per_circuit_l716_716230

theorem squirrel_rise_per_circuit
    (height_post : ℝ) 
    (circumference_post : ℝ)
    (distance_traveled : ℝ)
    (height_post_eq : height_post = 16)
    (circumference_post_eq : circumference_post = 2)
    (distance_traveled_eq : distance_traveled = 8) : 
  (height_post / (distance_traveled / circumference_post) = 4) :=
begin
  sorry
end

end squirrel_rise_per_circuit_l716_716230


namespace negation_of_p_l716_716451

noncomputable def proposition_p : Prop :=
  ∃ x0 : ℝ, x0 > 0 ∧ cos x0 + sin x0 > 1

theorem negation_of_p :
  ¬ proposition_p ↔ ∀ x : ℝ, x > 0 → cos x + sin x ≤ 1 :=
by
  sorry

end negation_of_p_l716_716451


namespace shifted_parabola_l716_716956

theorem shifted_parabola (x : ℝ) : 
  (let y := x^2 in (let x := x + 2 in y)) = (x - 2)^2 := sorry

end shifted_parabola_l716_716956


namespace trajectory_of_midpoint_l716_716698

noncomputable section

open Real

-- Define the points and lines
def C : ℝ × ℝ := (-2, -2)
def A (x : ℝ) : ℝ × ℝ := (x, 0)
def B (y : ℝ) : ℝ × ℝ := (0, y)
def M (x y : ℝ) : ℝ × ℝ := ((x + 0) / 2, (0 + y) / 2)

theorem trajectory_of_midpoint (CA_dot_CB : (C.1 * (A 0).1 + (C.2 - (A 0).2)) * (C.1 * (B 0).1 + (C.2 - (B 0).2)) = 0) :
  ∀ (M : ℝ × ℝ), (M.1 = (A 0).1 / 2) ∧ (M.2 = (B 0).2 / 2) → (M.1 + M.2 + 2 = 0) :=
by
  -- here's where the proof would go
  sorry

end trajectory_of_midpoint_l716_716698


namespace cos_theta_of_triangle_median_l716_716614

theorem cos_theta_of_triangle_median
  (A : ℝ) (a : ℝ) (m : ℝ) (theta : ℝ)
  (area_eq : A = 24)
  (side_eq : a = 12)
  (median_eq : m = 5)
  (area_formula : A = (1/2) * a * m * Real.sin theta) :
  Real.cos theta = 3 / 5 := 
by 
  sorry

end cos_theta_of_triangle_median_l716_716614


namespace parallelogram_ratio_l716_716386

noncomputable def is_parallelogram (E F G H : Type) :=
  ∃ (P Q : E = F ∧ G = H ∧ E ≠ G ∧ F ≠ H), true

variable (E F G H Q R S : Type)

variable [is_parallelogram E F G H]
variable [on_line EQ EF (23/1005)]
variable [on_line ER EH (23/2011)]
variable [is_intersection S (EG) (QR)]

theorem parallelogram_ratio (E F G H Q R S : Type) [is_parallelogram E F G H] :
  ∀ (EG ES : ℝ), EG > 0 → ES > 0 → EG / ES = 131 :=
by
  sorry

end parallelogram_ratio_l716_716386


namespace parallelogram_area_l716_716036

theorem parallelogram_area (base height : ℝ) (h_base : base = 22) (h_height : height = 14) :
  base * height = 308 := by
  sorry

end parallelogram_area_l716_716036


namespace maximize_vector_sum_length_l716_716559

-- Definitions and conditions
def O : Point := sorry
def A : ℕ → Point := sorry
def n : ℕ := 25

-- Conditions for the regular polygon and vectors
def is_regular_25_sided_polygon (A : ℕ → Point) (O : Point) : Prop :=
  ∀ i j, (i ≠ j ∧ i < 25 ∧ j < 25) → 
    dist O (A i) = dist O (A j) ∧ 
    angle (A i - O) (A j - O) = 2 * π / 25

def unit_vectors (A : ℕ → Point) (O : Point) : Prop :=
  ∀ i, i < 25 → dist O (A i) = 1

-- The theorem
theorem maximize_vector_sum_length
  (O : Point) (A : ℕ → Point)
  (h1 : is_regular_25_sided_polygon A O)
  (h2 : unit_vectors A O) :
  ∃ S : Finset ℕ, (S.card = 12 ∨ S.card = 13) ∧ 
    ∑ i in S, (A i - O) = 7.97 := sorry

end maximize_vector_sum_length_l716_716559


namespace smallest_percent_increase_l716_716763

-- Define the values of each question.
def value (n : ℕ) : ℕ :=
  match n with
  | 1  => 150
  | 2  => 300
  | 3  => 450
  | 4  => 600
  | 5  => 800
  | 6  => 1500
  | 7  => 3000
  | 8  => 6000
  | 9  => 12000
  | 10 => 24000
  | 11 => 48000
  | 12 => 96000
  | 13 => 192000
  | 14 => 384000
  | 15 => 768000
  | _ => 0

-- Define the percent increase between two values.
def percent_increase (v1 v2 : ℕ) : ℚ :=
  ((v2 - v1 : ℕ) : ℚ) / v1 * 100 

-- Prove that the smallest percent increase is between question 4 and 5.
theorem smallest_percent_increase :
  percent_increase (value 4) (value 5) = 33.33 := 
by
  sorry

end smallest_percent_increase_l716_716763


namespace regular_polygon_sides_l716_716166

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716166


namespace magic_trick_constant_l716_716518

theorem magic_trick_constant (a : ℚ) : ((2 * a + 8) / 4 - a / 2) = 2 :=
by
  sorry

end magic_trick_constant_l716_716518


namespace triangle_area_ab_l716_716368

theorem triangle_area_ab (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : let A := 12 in let B := (1/2) * (A/a) * (A/b) in B = 12) : a * b = 6 :=
sorry

end triangle_area_ab_l716_716368


namespace regular_polygon_sides_l716_716117

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716117


namespace regular_polygon_sides_l716_716135

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716135


namespace ratio_areas_l716_716794

noncomputable def side_length_ABCD := 6
noncomputable def area_ABCD := (side_length_ABCD ^ 2)
noncomputable def area_EFGH := 27 + 18 * Real.sqrt 2

theorem ratio_areas (s : ℝ)
  (side_length_ABCD = s)
  (area_ABCD = s ^ 2)
  (area_EFGH = 27 + 18 * Real.sqrt 2)
  : (area_EFGH / area_ABCD) = (3 / 4 + Real.sqrt 2 / 2) := by
  sorry

end ratio_areas_l716_716794


namespace inclination_angle_of_line_l716_716341

theorem inclination_angle_of_line (θ : ℝ) : (∃ θ, l = sqrt 3 * x - y - 10 = 0) ∧ (0 ≤ θ ∧ θ < real.pi) → θ = real.pi / 3 :=
by
  sorry

end inclination_angle_of_line_l716_716341


namespace number_of_perfect_square_multiples_21_below_2000_l716_716668

/-- Define n and the condition 21n being a perfect square --/
def is_perfect_square (k : ℕ) : Prop :=
  ∃ m : ℕ, m * m = k

def count_perfect_squares_upto (N : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | n ≤ N ∧ ∃ a : ℕ, n = k * a ∧ is_perfect_square (k * n) }

theorem number_of_perfect_square_multiples_21_below_2000 : 
  count_perfect_squares_upto 2000 21 21 = 9 :=
sorry

end number_of_perfect_square_multiples_21_below_2000_l716_716668


namespace largest_integer_square_two_digits_l716_716795

theorem largest_integer_square_two_digits : 
  ∃ M : ℤ, (M * M ≥ 10 ∧ M * M < 100) ∧ (∀ x : ℤ, (x * x ≥ 10 ∧ x * x < 100) → x ≤ M) ∧ M = 9 := 
by
  sorry

end largest_integer_square_two_digits_l716_716795


namespace b_days_with_decreased_efficiency_l716_716555

noncomputable theory

variables (W : ℝ)

-- Conditions
def a_b_together (a b : ℝ) : Prop := a + b = W / 6
def a_alone (a : ℝ) : Prop := a = W / 10
def a_efficiency_increase (a' a : ℝ) : Prop := a' = (1.5 * a)
def b_efficiency_decrease (b' b : ℝ) : Prop := b' = (0.75 * b)

-- Question and proof goal
theorem b_days_with_decreased_efficiency (a b a' b': ℝ) 
  (h1 : a_b_together a b) 
  (h2 : a_alone a) 
  (h3 : a_efficiency_increase a' a) 
  (h4 : b_efficiency_decrease b' b) : 
  (W / b') = 20 := 
sorry

end b_days_with_decreased_efficiency_l716_716555


namespace replace_preserve_mean_variance_l716_716403

theorem replace_preserve_mean_variance:
  ∀ (a b c : ℤ), 
    let initial_set := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map (λ x, x : ℤ) in
    let new_set := (initial_set.erase a).++[b, c] in
    let mean (s : List ℤ) := (s.sum : ℚ) / s.length in
    let variance (s : List ℤ) :=
      let m := mean s in
      (s.map (λ x, (x - m) ^ 2)).sum / s.length in
    mean initial_set = 0 ∧ variance initial_set = 10 ∧
    ((mean new_set = 0 ∧ variance new_set = 10) ↔ ((a = -4 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 5))) :=
sorry

end replace_preserve_mean_variance_l716_716403


namespace basketball_shooting_test_probability_l716_716571

noncomputable def probability_of_passing_test : ℝ :=
  let p := 0.6 in
  let probability_2_shots := (3.choose 2) * (p^2) * ((1 - p)^1) in
  let probability_3_shots := (3.choose 3) * (p^3) * ((1 - p)^0) in
  probability_2_shots + probability_3_shots

theorem basketball_shooting_test_probability : probability_of_passing_test = 0.648 :=
  sorry

end basketball_shooting_test_probability_l716_716571


namespace servings_in_box_l716_716585

theorem servings_in_box (total_cereal : ℕ) (serving_size : ℕ) (total_cereal_eq : total_cereal = 18) (serving_size_eq : serving_size = 2) :
  total_cereal / serving_size = 9 :=
by
  sorry

end servings_in_box_l716_716585


namespace regular_polygon_sides_l716_716160

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716160


namespace line_circle_intersection_l716_716328

-- Define the circle in parametric form
def circle_x (θ : ℝ) : ℝ := -1 + 2 * Real.cos θ
def circle_y (θ : ℝ) : ℝ := 3 + 2 * Real.sin θ

-- Define the line in parametric form
def line_x (t : ℝ) : ℝ := 2 * t - 1
def line_y (t : ℝ) : ℝ := 6 * t - 1

-- Circle's center and radius
def circle_center : ℝ × ℝ := (-1, 3)
def circle_radius : ℝ := 2

-- Line in standard form: Ax + By + C = 0
def A : ℝ := 3
def B : ℝ := -1
def C : ℝ := 2

-- Distance from a point (x₀, y₀) to a line Ax + By + C = 0
def distance (x₀ y₀ : ℝ) : ℝ := |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- The proof statement
theorem line_circle_intersection (d : ℝ) (h₀ : d = distance (-1) 3) (h₁ : d < circle_radius) : 
  ∃ x θ t, (circle_x θ = x ∧ line_x t = x) ∨ (circle_y θ = x ∧ line_y t = x) :=
sorry

end line_circle_intersection_l716_716328


namespace regular_polygon_sides_l716_716147

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716147


namespace binary_representation_253_l716_716986

theorem binary_representation_253 (x y : ℕ) (h : nat.bits 253 = [1, 1, 1, 1, 1, 1, 0, 1]) :
  (y - x = 6) :=
begin
  have hx : x = 1, from by {
    -- x is the number of 0's in the list representation [1, 1, 1, 1, 1, 1, 0, 1]
    sorry
  },
  have hy : y = 7, from by {
    -- y is the number of 1's in the list representation [1, 1, 1, 1, 1, 1, 0, 1]
    sorry
  },
  rw [hx, hy],
  exact rfl,
end

end binary_representation_253_l716_716986


namespace regular_polygon_sides_l716_716189

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716189


namespace paint_grid_condition_l716_716268

variables {a b c d e A B C D E : ℕ}

def is_valid (n : ℕ) : Prop := n = 2 ∨ n = 3

theorem paint_grid_condition 
  (ha : is_valid a) (hb : is_valid b) (hc : is_valid c) 
  (hd : is_valid d) (he : is_valid e) (hA : is_valid A) 
  (hB : is_valid B) (hC : is_valid C) (hD : is_valid D) 
  (hE : is_valid E) :
  a + b + c + d + e = A + B + C + D + E :=
sorry

end paint_grid_condition_l716_716268


namespace magnitude_calculation_l716_716970

open Complex

-- Define the complex number
def z : ℂ := 2 + 2 * Complex.i

-- State the conditions as definitions
def magnitude_identity (z : ℂ) (n : ℕ) : Prop :=
  abs (z^n) = (abs z)^n

def complex_modulus (a b : ℂ) : ℝ := Real.sqrt (a.re^2 + b.im^2)

-- State the proof problem
theorem magnitude_calculation : abs (z^8) = 4096 := by
  -- Using the given conditions
  have h1 : abs (z^8) = (abs z)^8 := by sorry
  have h2 : abs z = Real.sqrt (2^2 + 2^2) := by sorry
  -- Show that magnitude is 4096
  sorry

end magnitude_calculation_l716_716970


namespace work_completion_by_C_l716_716028

theorem work_completion_by_C
  (A_work_rate : ℝ)
  (B_work_rate : ℝ)
  (C_work_rate : ℝ)
  (A_days_worked : ℝ)
  (B_days_worked : ℝ)
  (C_days_worked : ℝ)
  (A_total_days : ℝ)
  (B_total_days : ℝ)
  (C_completion_partial_work : ℝ)
  (H1 : A_work_rate = 1 / 40)
  (H2 : B_work_rate = 1 / 40)
  (H3 : A_days_worked = 10)
  (H4 : B_days_worked = 10)
  (H5 : C_days_worked = 10)
  (H6 : C_completion_partial_work = 1/2) :
  C_work_rate = 1 / 20 :=
by
  sorry

end work_completion_by_C_l716_716028


namespace total_earning_correct_l716_716898

-- Definitions based on conditions
def daily_wage_c : ℕ := 105
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

-- Given the ratio of their daily wages
def ratio_a : ℕ := 3
def ratio_b : ℕ := 4
def ratio_c : ℕ := 5

-- Now we calculate the daily wages based on the ratio
def unit_wage : ℕ := daily_wage_c / ratio_c
def daily_wage_a : ℕ := ratio_a * unit_wage
def daily_wage_b : ℕ := ratio_b * unit_wage

-- Total earnings are calculated by multiplying daily wages and days worked
def total_earning_a : ℕ := days_worked_a * daily_wage_a
def total_earning_b : ℕ := days_worked_b * daily_wage_b
def total_earning_c : ℕ := days_worked_c * daily_wage_c

def total_earning : ℕ := total_earning_a + total_earning_b + total_earning_c

-- Theorem to prove
theorem total_earning_correct : total_earning = 1554 := by
  sorry

end total_earning_correct_l716_716898


namespace Jamal_crayon_cost_l716_716424

/-- Jamal bought 4 half dozen colored crayons at $2 per crayon. 
    He got a 10% discount on the total cost, and an additional 5% discount on the remaining amount. 
    After paying in US Dollars (USD), we want to know how much he spent in Euros (EUR) and British Pounds (GBP) 
    given that 1 USD is equal to 0.85 EUR and 1 USD is equal to 0.75 GBP. 
    This statement proves that the total cost was 34.884 EUR and 30.78 GBP. -/
theorem Jamal_crayon_cost :
  let number_of_crayons := 4 * 6
  let initial_cost := number_of_crayons * 2
  let first_discount := 0.10 * initial_cost
  let cost_after_first_discount := initial_cost - first_discount
  let second_discount := 0.05 * cost_after_first_discount
  let final_cost_usd := cost_after_first_discount - second_discount
  let final_cost_eur := final_cost_usd * 0.85
  let final_cost_gbp := final_cost_usd * 0.75
  final_cost_eur = 34.884 ∧ final_cost_gbp = 30.78 := 
by
  sorry

end Jamal_crayon_cost_l716_716424


namespace regular_polygon_sides_l716_716175

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716175


namespace induction_proof_l716_716479

open Nat

noncomputable def S (n : ℕ) : ℚ :=
  match n with
  | 0     => 0
  | (n+1) => S n + 1 / ((n+1) * (n+2))

theorem induction_proof : ∀ n : ℕ, S n = n / (n + 1) := by
  intro n
  induction n with
  | zero => 
    -- Base case: S(1) = 1/2
    sorry
  | succ n ih =>
    -- Induction step: Assume S(n) = n / (n + 1), prove S(n+1) = (n+1) / (n+2)
    sorry

end induction_proof_l716_716479


namespace poly_prime_or_composite_l716_716367

theorem poly_prime_or_composite (a : ℕ) :
  (∃ p : ℕ, p.prime ∧ a^4 - 3 * a^2 + 9 = p) ↔ (a = 1 ∨ a = 2) ∨
  (∃ b c : ℕ, b > 1 ∧ c > 1 ∧ a^4 - 3 * a^2 + 9 = b * c ∧ a > 2) :=
by
  -- Proof omitted.
sorry

end poly_prime_or_composite_l716_716367


namespace original_ladies_work_in_six_days_l716_716841

variable (L : ℕ) -- number of ladies
variable (D : ℕ) -- number of days

/-- The original condition that the group (2L ladies) will complete half work in 3 days. -/
def work_condition (L : ℕ) : Prop :=
  ∃ (k : ℕ), k = 2 * L ∧ (k * 3 = ⌊D / 2⌋ * 3)

theorem original_ladies_work_in_six_days (L : ℕ) (D : ℕ) (h : work_condition L) : D = 6 := by
  sorry

end original_ladies_work_in_six_days_l716_716841


namespace total_coffee_needed_l716_716351

-- Conditions as definitions
def weak_coffee_amount_per_cup : ℕ := 1
def strong_coffee_amount_per_cup : ℕ := 2 * weak_coffee_amount_per_cup
def cups_of_weak_coffee : ℕ := 12
def cups_of_strong_coffee : ℕ := 12

-- Prove that the total amount of coffee needed equals 36 tablespoons
theorem total_coffee_needed : (weak_coffee_amount_per_cup * cups_of_weak_coffee) + (strong_coffee_amount_per_cup * cups_of_strong_coffee) = 36 :=
by
  sorry

end total_coffee_needed_l716_716351


namespace regular_polygon_sides_l716_716078

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716078


namespace find_sin_alpha_l716_716388

-- given conditions
def C1_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, -1 + t * Real.sin α)

def C2_cartesian_eqn (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

def C1_C2_intersect (α : ℝ) (t_1 t_2 : ℝ) : Prop :=
  t_1 + t_2 = 2 * Real.cos α + 4 * Real.sin α

def midpoint_condition (t_1 t_2 : ℝ) : Prop :=
  ∣ (t_1 + t_2) / 2 ∣ = 2

-- problem statement
theorem find_sin_alpha (α : ℝ) (t_1 t_2 : ℝ) 
  (h1 : C1_C2_intersect α t_1 t_2) 
  (h2 : midpoint_condition t_1 t_2) : 
  Real.sin α = 3 / 5 ∨ Real.sin α = 1 :=
sorry

end find_sin_alpha_l716_716388


namespace regular_polygon_sides_l716_716093

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716093


namespace tom_reads_pages_l716_716653

-- Definition of conditions
def initial_speed : ℕ := 12   -- pages per hour
def speed_factor : ℕ := 3
def time_period : ℕ := 2     -- hours

-- Calculated speeds
def increased_speed (initial_speed speed_factor : ℕ) : ℕ := initial_speed * speed_factor
def total_pages (increased_speed time_period : ℕ) : ℕ := increased_speed * time_period

-- Theorem statement
theorem tom_reads_pages :
  total_pages (increased_speed initial_speed speed_factor) time_period = 72 :=
by
  -- Omitting proof as only theorem statement is required
  sorry

end tom_reads_pages_l716_716653


namespace probability_of_intersecting_chords_l716_716493

-- Conditions of the problem
def points_on_circle : Finset (Fin 2020) := Finset.univ

-- Define what it means for a chord to intersect
def chords_intersect (A B C D : Fin 2020) : Prop :=
  ∃ (i j k l : Fin 2020), 
    A = i ∧ B = k ∧ C = j ∧ D = l ∧ 
    (i < j < k < l ∨ j < l < i < k ∨ k < i < l < j ∨ l < j < k < i)

-- Probability space over selection of 4 points from 2020 points
def selection_space : ProbabilitySpace (Finset (Fin 2020)) :=
  ⟨Finset.powersetOfCard points_on_circle 4, sorry ⟩

-- Lean statement to prove the correctness of the answer
theorem probability_of_intersecting_chords :  probability 
  { S ∈ selection_space | ∃ (A B C D : Fin 2020), S = {A, B, C, D} ∧ chords_intersect A B C D } = 1 / 12 :=
by
  sorry

end probability_of_intersecting_chords_l716_716493


namespace x1_x2_product_l716_716572

theorem x1_x2_product (a b c d x1 x2 : ℝ) (h1 : f x = a * x^3 + b * x^2 + c * x + d)
  (hx_intersections : f 0 = 0 ∧ f x1 = 0 ∧ f x2 = 0)
  (h_extremes : f'.(1) = 0 ∧ f'.(2) = 0) : 
  x1 ≠ 0 ∧ x2 ≠ 0 → (x1 * x2 = 6) :=
sorry

end x1_x2_product_l716_716572


namespace valid_fahrenheit_count_l716_716261

def is_valid_Fahrenheit (F : ℤ) : Prop :=
  let C := Float.floor (5.0 / 9.0 * (F.toReal - 32.0))
  let F' := Float.floor (9.0 / 5.0 * C + 32.0)
  F' = F

def count_valid_Fahrenheit_temps (lo hi : ℕ) : ℕ :=
  (lo.to hi).filter (λ F, is_valid_Fahrenheit F).length

theorem valid_fahrenheit_count :
  count_valid_Fahrenheit_temps 50 2000 = 1091 := 
  sorry

end valid_fahrenheit_count_l716_716261


namespace frequency_stabilizes_as_trials_increase_l716_716962

-- Definitions that reflect understanding of frequency and probability
def is_constant_approx (f : ℕ → ℝ) (p : ℝ) : Prop :=
  ∃ N, ∀ n ≥ N, |f n - p| < ε

-- Problem statement transformed into a Lean declarative statement
theorem frequency_stabilizes_as_trials_increase :
  ∀ (f : ℕ → ℝ) (p : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |f n - p| < ε) → 
  ("C: As the number of trials increases, the frequency will generally stabilize near a constant") := 
sorry

end frequency_stabilizes_as_trials_increase_l716_716962


namespace game_a_greater_than_game_c_l716_716052

-- Definitions of probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probabilities for Game A and Game C based on given conditions
def prob_game_a : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)
def prob_game_c : ℚ :=
  (prob_heads ^ 5) +
  (prob_tails ^ 5) +
  (prob_heads ^ 3 * prob_tails ^ 2) +
  (prob_tails ^ 3 * prob_heads ^ 2)

-- Define the difference
def prob_difference : ℚ := prob_game_a - prob_game_c

-- The theorem to be proved
theorem game_a_greater_than_game_c :
  prob_difference = 3 / 64 :=
by
  sorry

end game_a_greater_than_game_c_l716_716052


namespace pond_water_amount_l716_716059

theorem pond_water_amount : 
  let initial_water := 500 
  let evaporation_rate := 4
  let rain_amount := 2
  let days := 40
  initial_water - days * (evaporation_rate - rain_amount) = 420 :=
by
  sorry

end pond_water_amount_l716_716059


namespace regular_polygon_sides_l716_716197

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716197


namespace regular_polygon_sides_l716_716153

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716153


namespace sum_eq_sqrt_122_l716_716441

theorem sum_eq_sqrt_122 
  (a b c : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h1 : a^2 + b^2 + c^2 = 58) 
  (h2 : a * b + b * c + c * a = 32) :
  a + b + c = Real.sqrt 122 := 
by
  sorry

end sum_eq_sqrt_122_l716_716441


namespace solution_of_equation_l716_716993

theorem solution_of_equation :
  ∃ (x : ℝ), (x > 0) ∧ (1/3) * (4*x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 40*x + 8) ∧
    x = (75 + Real.sqrt 5701) / 2 :=
begin
  sorry
end

end solution_of_equation_l716_716993


namespace regular_polygon_sides_l716_716185

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716185


namespace ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l716_716283

theorem ones_digit_largest_power_of_three_divides_factorial_3_pow_3 :
  (3 ^ 13) % 10 = 3 := by
  sorry

end ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l716_716283


namespace length_of_the_train_l716_716948

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def time_to_cross_seconds : ℝ := 30
noncomputable def bridge_length_meters : ℝ := 205

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def distance_crossed_meters : ℝ := train_speed_mps * time_to_cross_seconds

theorem length_of_the_train 
  (h1 : train_speed_kmph = 45)
  (h2 : time_to_cross_seconds = 30)
  (h3 : bridge_length_meters = 205) : 
  distance_crossed_meters - bridge_length_meters = 170 := 
by
  sorry

end length_of_the_train_l716_716948


namespace quadratic_roots_are_symmetric_l716_716539

theorem quadratic_roots_are_symmetric (p q : ℝ) (α β : ℝ) :
  (α + β = -p) ∧ (α * β = q) ↔ 
  (∃ x : ℝ, polynomial.eval x (polynomial.C q + polynomial.C p * polynomial.X + polynomial.X ^ 2) = 0 ∧
             (polynomial.eval α (polynomial.C q + polynomial.C p * polynomial.X + polynomial.X ^ 2) = 0 ∧
             polynomial.eval β (polynomial.C q + polynomial.C p * polynomial.X + polynomial.X ^ 2) = 0)) := 
sorry

end quadratic_roots_are_symmetric_l716_716539


namespace car_owners_without_motorcycle_or_bicycle_l716_716749

-- Given conditions
def num_adults : ℕ := 500
def num_car_owners : ℕ := 420
def num_motorcycle_owners : ℕ := 80
def num_bicycle_owners : ℕ := 200

-- Question translated into a proof statement
theorem car_owners_without_motorcycle_or_bicycle : 
  ∀ (A B C : Finset ℕ), 
    A.card = num_car_owners ∧ B.card = num_motorcycle_owners ∧ C.card = num_bicycle_owners ∧
    A ∪ B ∪ C = Finset.range num_adults →
    A \ (B ∪ C).card = 375 :=
by {
  sorry
}

end car_owners_without_motorcycle_or_bicycle_l716_716749


namespace zebra_average_speed_l716_716617

/-- The tiger runs 5 hours before the zebra starts chasing -/
def tiger_lead_time : ℝ := 5

/-- The zebra catches the tiger in 6 hours -/
def zebra_chase_time : ℝ := 6

/-- The average speed of the tiger -/
def tiger_speed : ℝ := 30

/-- The average speed of the zebra -/
def zebra_speed (V_z : ℝ) : Prop :=
  let tiger_start_distance := tiger_lead_time * tiger_speed
  let tiger_additional_distance := zebra_chase_time * tiger_speed
  let total_distance := tiger_start_distance + tiger_additional_distance
  let zebra_distance := zebra_chase_time * V_z
  zebra_distance = total_distance

theorem zebra_average_speed : zebra_speed 55 :=
begin
  simp [zebra_speed, tiger_lead_time, zebra_chase_time, tiger_speed],
  norm_num,
end

end zebra_average_speed_l716_716617


namespace problem_statement_l716_716000

noncomputable def sequence (M : ℕ) (hM : M > 5) : ℕ → ℚ
| n => if (n ≤ M / 2) then (1 / (2 ^ n)) else -(1 / (2 ^ (M + 1 - n)))

noncomputable def S (n : ℕ) (a_n : ℕ → ℚ) := ∑ i in range n, a_n i

theorem problem_statement (M : ℕ) (hM : M > 5) (a_n : ℕ → ℚ) (hSeq : ∀ k, k > 0 → k ≤ M → a_n k + a_n (M + 1 - k) = 0) 
  (hHalf : ∀ n, n > 0 → n ≤ M / 2 → a_n n = 1 / (2 ^ n)) :
  (¬ (∀ n, S n a_n ≤ 1023 / 1024 → M ≤ 20)) ∧
  ((∃ n, (n, n + 1, n + 2, n + 3, n + 4).formArithProgression (a_n n) (a_n (n + 1)) (a_n (n + 2)) (a_n (n + 3)) (a_n (n + 4)))) ∧
  (∀ p q, p > 0 → q > 0 → p < M → q < M → ∃ i j, (i > 0) → (j > 0) → a_n i + a_n j = S p a_n - S q a_n) ∧
  (∀ r, r > 0 → r ≤ M → ∃ s t, s ≠ t → (s > 0) → (t > 0) → (s ≤ M) → (t ≤ M) → ((a_n r), (a_n s), (a_n t)).mkArithProgression (a_n r) (a_n s) (a_n t)) := sorry

end problem_statement_l716_716000


namespace regular_polygon_sides_l716_716137

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716137


namespace regular_polygon_sides_l716_716128

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716128


namespace paul_min_steps_to_one_is_six_l716_716826

-- Define the conditions
def initial_number : ℕ := 19

-- Define the possible operations
def step (n : ℕ) : List ℕ := [n + 1, n / 2, n / 3].filter (λ x => x > 0)

-- Define a function to find the minimum number of steps to reach 1 from the initial number
def min_steps_to_one (n : ℕ) : ℕ :=
  if n = 1 then 0
  else 1 + min (min (min_steps_to_one (n + 1))) (min (min_steps_to_one (n / 2))) (min_steps_to_one (n / 3))

theorem paul_min_steps_to_one_is_six : min_steps_to_one initial_number = 6 := by
  sorry

end paul_min_steps_to_one_is_six_l716_716826


namespace y_completes_work_in_seventy_days_l716_716041

def work_days (mahesh_days : ℕ) (mahesh_work_days : ℕ) (rajesh_days : ℕ) (y_days : ℕ) : Prop :=
  let mahesh_rate := (1:ℝ) / mahesh_days
  let rajesh_rate := (1:ℝ) / rajesh_days
  let work_done_by_mahesh := mahesh_rate * mahesh_work_days
  let remaining_work := (1:ℝ) - work_done_by_mahesh
  let rajesh_remaining_work_days := remaining_work / rajesh_rate
  let y_rate := (1:ℝ) / y_days
  y_rate = rajesh_rate

theorem y_completes_work_in_seventy_days :
  work_days 35 20 30 70 :=
by
  -- This is where the proof would go
  sorry

end y_completes_work_in_seventy_days_l716_716041


namespace sum_first_13_terms_l716_716389

variable {a : ℕ → ℝ} (d : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + 2 * a 9 = 8

theorem sum_first_13_terms :
  (arithmetic_sequence a d) → (given_condition a) → 
  (∑ i in Finset.range 13, a i) = 26 :=
by
  sorry

end sum_first_13_terms_l716_716389


namespace count_multiples_of_7_not_14_lt_400_l716_716731

theorem count_multiples_of_7_not_14_lt_400 : 
  ∃ (n : ℕ), n = 29 ∧ ∀ (m : ℕ), (m < 400 ∧ m % 7 = 0 ∧ m % 14 ≠ 0) ↔ (∃ k : ℕ, 1 ≤ k ∧ k ≤ 29 ∧ m = 7 * (2 * k - 1)) :=
by
  sorry

end count_multiples_of_7_not_14_lt_400_l716_716731


namespace isosceles_right_triangle_inscribed_square_area_l716_716384

theorem isosceles_right_triangle_inscribed_square_area (area : ℝ) 
  (h₁ : ∃ (s: ℝ), s^2 = 484) 
  (h₂ : ∃ (l: ℝ), l = 2 * (sqrt 484)) 
  (h₃ : ∀ (S: ℝ), 3*S = 2 * (sqrt 484) * sqrt 2) :
  area = 3872/9 := 
sorry

end isosceles_right_triangle_inscribed_square_area_l716_716384


namespace regular_polygon_sides_l716_716182

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716182


namespace regular_polygon_sides_l716_716125

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716125


namespace minimum_value_of_f_l716_716444

noncomputable def f (x y : ℝ) : ℝ := 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4

theorem minimum_value_of_f :
  ∃ (x y : ℝ), ∀ (x' y' : ℝ), f x y ≤ f x' y' :=
  by
    use (0, -0.5)
    simp [f]
    linarith
    sorry

end minimum_value_of_f_l716_716444


namespace regular_polygon_sides_l716_716130

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716130


namespace rectangle_same_color_l716_716639

/-- In a 3 × 7 grid where each square is either black or white, 
  there exists a rectangle whose four corners are of the same color. -/
theorem rectangle_same_color (grid : Fin 3 × Fin 7 → Bool) :
  ∃ (r1 r2 : Fin 3) (c1 c2 : Fin 7), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid (r1, c1) = grid (r1, c2) ∧ grid (r2, c1) = grid (r2, c2) :=
by
  sorry

end rectangle_same_color_l716_716639


namespace job_completion_time_for_B_l716_716583

/-- Define the assumptions related to A and B's work rates and the combined work fraction. -/
theorem job_completion_time_for_B :
  (∀ (x : ℝ), 8 * (1/15 + 1/x) = 14/15 → x = 20) := 
begin
  assume x h,
  sorry,
end

end job_completion_time_for_B_l716_716583


namespace universal_proposition_l716_716552

def is_multiple_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

def is_even (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

theorem universal_proposition : 
  (∀ x : ℕ, is_multiple_of_two x → is_even x) :=
by
  sorry

end universal_proposition_l716_716552


namespace original_decimal_number_l716_716905

theorem original_decimal_number (x : ℝ) (h : 0.375 = (x / 1000) * 10) : x = 37.5 :=
sorry

end original_decimal_number_l716_716905


namespace regular_polygon_sides_l716_716149

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716149


namespace infinite_possible_matrices_A_squared_l716_716439

theorem infinite_possible_matrices_A_squared (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A^4 = 0) :
  ∃ (S : Set (Matrix (Fin 3) (Fin 3) ℝ)), (∀ B ∈ S, B = A^2) ∧ S.Infinite :=
sorry

end infinite_possible_matrices_A_squared_l716_716439


namespace sum_of_squares_divisors_l716_716521

theorem sum_of_squares_divisors :
  ∃ (n : ℤ), (∀ (n : ℤ), (n^2 + n + 1).dvd (n^2013 + 61) → n = 0 ∨ n = -1 ∨ n = 5 ∨ n = -6) ∧ 
  (0^2 + (-1)^2 + 5^2 + (-6)^2 = 62) :=
by
  sorry

end sum_of_squares_divisors_l716_716521


namespace dice_roll_prime_probability_l716_716876

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def prime_probability (n : ℕ) : ℚ :=
  if n = 1 then 1 else if is_prime n then 58 / 216 else 0

theorem dice_roll_prime_probability :
  (∑ n in finset.range (18 + 1), if is_prime n then (1/ (6 * 6 * 6)) else 0) = 58 / 216 :=
by
  -- Define the range of sums
  let sums := finset.range (18 + 1)
  -- Define the probabilities for prime sums
  let prime_sums := sums.filter is_prime
  -- Calculate probability
  have h : (∑ n in prime_sums, 1 / 216) = 58 / 216 := sorry
  exact h

end dice_roll_prime_probability_l716_716876


namespace tangent_addition_l716_716831

theorem tangent_addition (α β γ : ℝ) :
  tan (α + β + γ) = (tan α + tan β + tan γ - tan α * tan β * tan γ) / 
                    (1 - tan α * tan β - tan β * tan γ - tan γ * tan α) :=
by
  sorry

end tangent_addition_l716_716831


namespace product_divisible_by_60_l716_716838

theorem product_divisible_by_60 {a : ℤ} : 
  60 ∣ ((a^2 - 1) * a^2 * (a^2 + 1)) := 
by sorry

end product_divisible_by_60_l716_716838


namespace regular_polygon_sides_l716_716097

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716097


namespace regular_polygon_sides_l716_716094

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716094


namespace trapezoid_area_l716_716888

-- Define vertices of the unit square
def A : ℝ×ℝ := (0, 0)
def B : ℝ×ℝ := (1, 0)
def C : ℝ×ℝ := (1, 1)
def D : ℝ×ℝ := (0, 1)

-- Define vertices of the trapezoid
variables (a : ℝ) (h_a : 0 ≤ a ∧ a ≤ 0.5)

-- Vertices of the trapezoid
def P : ℝ×ℝ := (a, 0)
def Q : ℝ×ℝ := (1, a)
def R : ℝ×ℝ := (1 - a, 1)
def S : ℝ×ℝ := (0, 1 - a)

-- Definition to check midpoint conditions
def midpoints_parallel : Prop :=
  let mPQ := ((P a).fst + (Q a).fst) / 2, ((P a).snd + (Q a).snd) / 2 in
  let mRS := ((R a).fst + (S a).fst) / 2, ((R a).snd + (S a).snd) / 2 in
  let mQR := ((Q a).fst + (R a).fst) / 2, ((Q a).snd + (R a).snd) / 2 in
  let mSP := ((S a).fst + (P a).fst) / 2, ((S a).snd + (P a).snd) / 2 in
  mPQ.snd = mRS.snd ∧ mQR.fst = mSP.fst

-- The main theorem
theorem trapezoid_area (H : midpoints_parallel a h_a) : (1 - 2 * a) = (1 / 2) * ((P a).fst - (S a).fst + (Q a).fst - (R a).fst) := sorry

end trapezoid_area_l716_716888


namespace min_value_of_expr_l716_716445

noncomputable def min_value_expr (x y z : ℝ) : ℝ :=
  (x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1) / (x * y * z)

theorem min_value_of_expr :
  ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → min_value_expr x y z ≥ 216 :=
by {
  intros x y z hx hy hz,
  -- Further proof steps would go here
  sorry
}

end min_value_of_expr_l716_716445


namespace distance_AP_correct_l716_716642

-- Define the conditions
def unit_square : Prop :=
  -- Definitions of the vertices and the square itself can be simplified for this purpose
  true

def inscribed_circle : Prop :=
  ∃ (ω : ℝ × ℝ → ℝ) (O : ℝ × ℝ), 
    -- Circle ω centered at (0, 1/3) with radius 1/3
    (O = (0, 1/3)) ∧ 
    (∀ P : ℝ × ℝ, ω P = (P.1 - O.1)^2 + (P.2 - O.2)^2) ∧ 
    (ω (0, 0) = (1/3)^2)

def line_AM_intersection : Prop :=
  ∃ (A M P : ℝ × ℝ), 
    -- A at (0, 1), M at (0, 0), P at (0, 2/3)
    (A = (0, 1)) ∧ 
    (M = (0, 0)) ∧ 
    (P = (0, 2/3)) ∧ 
    -- Line AM is the vertical line x = 0 intersecting circle ω at P
    (P ∈ (λ (x : ℝ) (y : ℝ), (x - 0)^2 + (y - 1/3)^2 = (1/3)^2) (P.2))

theorem distance_AP_correct :
  unit_square →
  inscribed_circle →
  line_AM_intersection →
  ∀ (A P : ℝ × ℝ), (A = (0, 1)) → (P = (0, 2/3)) → (sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 1/3) :=
by
  intros _ _ _ A P hA hP
  sorry

end distance_AP_correct_l716_716642


namespace find_value_of_expression_l716_716851

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem find_value_of_expression (a b : ℝ) (h : quadratic_function a b (-1) = 0) :
  2 * a - 2 * b = -4 :=
sorry

end find_value_of_expression_l716_716851


namespace weight_ratio_mars_moon_l716_716854

theorem weight_ratio_mars_moon :
  (∀ iron carbon other_elements_moon other_elements_mars wt_moon wt_mars : ℕ, 
    wt_moon = 250 ∧ 
    iron = 50 ∧ 
    carbon = 20 ∧ 
    other_elements_moon + 50 + 20 = 100 ∧ 
    other_elements_moon * wt_moon / 100 = 75 ∧ 
    other_elements_mars = 150 ∧ 
    wt_mars = (other_elements_mars * wt_moon) / other_elements_moon
  → wt_mars / wt_moon = 2) := 
sorry

end weight_ratio_mars_moon_l716_716854


namespace servings_in_box_l716_716588

def totalCereal : ℕ := 18
def servingSize : ℕ := 2

theorem servings_in_box : totalCereal / servingSize = 9 := by
  sorry

end servings_in_box_l716_716588


namespace regular_polygon_sides_l716_716152

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716152


namespace circle_radius_and_circumference_l716_716844

theorem circle_radius_and_circumference (A : ℝ) (hA : A = 64 * Real.pi) :
  ∃ r C : ℝ, r = 8 ∧ C = 2 * Real.pi * r :=
by
  -- statement ensures that with given area A, you can find r and C satisfying the conditions.
  sorry

end circle_radius_and_circumference_l716_716844


namespace cube_painting_distinct_ways_dodecahedron_painting_distinct_ways_l716_716033

-- Proof Problem for Part (a)
theorem cube_painting_distinct_ways : 
  let colors := 6,
      faces := 6,
      rotations := 24,
      total_colorings := colors.factorial
  in total_colorings / rotations = 30 := 
begin
  sorry
end

-- Proof Problem for Part (b)
theorem dodecahedron_painting_distinct_ways :
  let colors := 12,
      faces := 12,
      rotations := 60,
      total_colorings := colors.factorial
  in total_colorings / rotations = (11.factorial / 5) := 
begin
  sorry
end

end cube_painting_distinct_ways_dodecahedron_painting_distinct_ways_l716_716033


namespace derivative_at_one_l716_716443

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one :
  let f' (x : ℝ) : ℝ := Real.exp x + x * Real.exp x in
  f' 1 = 2 * Real.exp 1 :=
by
  let f' (x : ℝ) := Real.exp x + x * Real.exp x
  show f' 1 = 2 * Real.exp 1
  sorry

end derivative_at_one_l716_716443


namespace sequence_product_l716_716303

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = (1 + a n) / (1 - a n)

theorem sequence_product (a : ℕ → ℚ) (h : sequence a) :
  (∏ i in Finset.range 2013, a (i + 1)) = 2 :=
sorry

end sequence_product_l716_716303


namespace bank_loan_interest_rate_approx_5_percent_l716_716909

theorem bank_loan_interest_rate_approx_5_percent :
  ∀ (loan_amount total_repaid interest_amount received_payment installments_per_year monthly_payment : ℝ)
  (h1 : loan_amount = 120)
  (h2 : total_repaid = 120)
  (h3 : interest_amount = 6)
  (h4 : received_payment = 114)
  (h5 : installments_per_year = 12)
  (h6 : monthly_payment = 10),
  (let r := (interest_amount / received_payment) * 100 in abs (r - 5) < 0.5) :=
by
  sorry

end bank_loan_interest_rate_approx_5_percent_l716_716909


namespace smallest_positive_angle_l716_716264

theorem smallest_positive_angle (k : ℤ) : (∃ α : ℝ, α = 400 + k * 360 ∧ α > 0 ∧ α < 360) → α = 40 :=
by
  intro hα
  cases hα with α hα
  cases hα with hα1 hα2
  cases hα2 with hα2_1 hα2_2
  sorry

end smallest_positive_angle_l716_716264


namespace collinear_points_count_l716_716234

-- Define the set of 27 special points on a cube
def cubePoints : Set Point := 
  {p | p ∈ vertices ∨ p ∈ midpoints ∨ p ∈ faceCenters ∨ p = cubeCenter}

-- The total number of three collinear points
def numCollinearPoints : ℕ := 49

-- The statement we want to prove
theorem collinear_points_count (cubePoints : Set Point) : 
  ∃! n, n = 49 :=
by
  -- Placeholder for the proof
  sorry

end collinear_points_count_l716_716234


namespace return_trip_time_eq_sixty_l716_716605

-- Definitions based on the conditions:
variables (d p w : ℝ)
def plane_speed (p : ℝ) : Prop := w = (1/3) * p
def time_against_wind (d p w : ℝ) : Prop := 120 = d / (p - w)
def time_still_air (d p : ℝ) : ℝ := d / p
def time_with_wind (d p w : ℝ) (t : ℝ) : Prop := (t - 20) = d / (p + w)

-- Theorem to prove:
theorem return_trip_time_eq_sixty (d p w : ℝ) (t : ℝ) 
  (h1 : plane_speed p)
  (h2 : time_against_wind d p w)
  (h3 : t = time_still_air d p)
  (h4 : time_with_wind d p w t) :
  (d / (p + w)) = 60 :=
sorry

end return_trip_time_eq_sixty_l716_716605


namespace avg_marks_second_class_l716_716278

noncomputable def n1 : ℕ := 58
noncomputable def n2 : ℕ := 52
noncomputable def avg1 : ℝ := 67
noncomputable def avg_combined : ℝ := 74.0909090909091
noncomputable def total_students : ℕ := n1 + n2

theorem avg_marks_second_class : 
  let total_marks_combined := avg_combined * total_students in
  let total_marks_first := avg1 * n1 in
  let x := (total_marks_combined - total_marks_first) / n2 in
  x = 81.6153846153846 := 
by
  sorry

end avg_marks_second_class_l716_716278


namespace angelina_speed_from_grocery_to_gym_l716_716034

-- Given conditions
def home_to_grocery_distance : ℕ := 180 -- meters
def grocery_to_gym_distance : ℕ := 240 -- meters
def time_difference : ℕ := 40 -- seconds

-- Define the speeds and times based on the given conditions
def v := 3 / 2
def home_to_grocery_time := home_to_grocery_distance / v
def grocery_to_gym_speed := 2 * v
def grocery_to_gym_time := grocery_to_gym_distance / grocery_to_gym_speed

-- Lean statement
theorem angelina_speed_from_grocery_to_gym :
  grocery_to_gym_speed = 3 := 
by
  have v_nonzero : v ≠ 0 := by sorry
  have eq1 : home_to_grocery_time = home_to_grocery_distance / v := rfl
  have eq2 : grocery_to_gym_time = grocery_to_gym_distance / grocery_to_gym_speed := rfl
  have eq3 : home_to_grocery_time - grocery_to_gym_time = time_difference := by sorry
  show grocery_to_gym_speed = 3 from sorry

end angelina_speed_from_grocery_to_gym_l716_716034


namespace regular_polygon_sides_l716_716077

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716077


namespace min_p_q_sum_l716_716490

theorem min_p_q_sum (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h : 162 * p = q^3) : p + q = 54 :=
sorry

end min_p_q_sum_l716_716490


namespace exists_a_k_eq_half_l716_716432

noncomputable def f (n : ℕ) (a : Fin n → ℝ) (S : Finset (Fin n)) : ℝ :=
  S.prod (λ i, a i) * Sᶜ.prod (λ i, 1 - a i)

theorem exists_a_k_eq_half (n : ℕ) (a : Fin n → ℝ)
  (h1 : ∀ i, 0 < a i ∧ a i < 1)
  (h2 : (Finset.powersetUniv.filter (λ S, S.card % 2 = 1)).sum (λ S, f n a S) = 1 / 2) :
  ∃ k, a k = 1 / 2 :=
by
  sorry

end exists_a_k_eq_half_l716_716432


namespace P_eq_F_l716_716346

noncomputable def P : set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def F : set ℝ := {x | x ≥ 1}

theorem P_eq_F : P = F := by
  sorry

end P_eq_F_l716_716346


namespace palm_trees_total_l716_716595

theorem palm_trees_total
  (forest_palm_trees : ℕ := 5000)
  (desert_palm_trees : ℕ := forest_palm_trees - (3 * forest_palm_trees / 5)) :
  desert_palm_trees + forest_palm_trees = 7000 :=
by
  sorry

end palm_trees_total_l716_716595


namespace option_c_same_function_l716_716236

-- Definitions based on conditions
def f_c (x : ℝ) : ℝ := x^2
def g_c (x : ℝ) : ℝ := 3 * x^6

-- Theorem statement that Option C f(x) and g(x) represent the same function
theorem option_c_same_function : ∀ x : ℝ, f_c x = g_c x := by
  sorry

end option_c_same_function_l716_716236


namespace plane_regions_l716_716889

theorem plane_regions (n : ℕ) (h1 : n = 100) (h_collinear : ∀ (p1 p2 p3 : Point), 
  ¬(collinear p1 p2 p3)) (h_not_parallel : ∀ (A B C D : Point), 
  ¬(parallel (line_through A B) (line_through C D)) ∧ 
  ∀ (A B C D : Point), (intersection (line_through A B) (line_through C D)) ∉ 
  (line_through A C ∪ line_through A D ∪ line_through B C ∪ line_through B D)) :
  regions n = 11778426 := 
sorry

end plane_regions_l716_716889


namespace triangle_area_inscribed_rectangle_area_l716_716508

theorem triangle_area (m n : ℝ) : ∃ (S : ℝ), S = m * n := 
sorry

theorem inscribed_rectangle_area (m n : ℝ) : ∃ (A : ℝ), A = (2 * m^2 * n^2) / (m + n)^2 :=
sorry

end triangle_area_inscribed_rectangle_area_l716_716508


namespace coeff_d_nonzero_l716_716978

noncomputable def Q (x : ℝ) (a b c d f : ℝ) : ℝ := x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + f

theorem coeff_d_nonzero 
  (a b c d f : ℝ) 
  (h0 : Q 0 a b c d f = 0) 
  (h1 : Q 1 a b c d f = 0) 
  (h_distinct : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 → Q x a b c d f = 0 → (x = p ∨ x = q ∨ x = r) ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  : d ≠ 0 :=
begin
  sorry
end

end coeff_d_nonzero_l716_716978


namespace power_function_quadrant_l716_716025

theorem power_function_quadrant (α : ℝ) (h : α ∈ ({-1, 1/2, 1, 2, 3} : set ℝ)) :
  ¬(∃ x : ℝ, x > 0 ∧ x < 1 ∧ x ≠ 0 ∧ x ^ α < 0) :=
sorry

end power_function_quadrant_l716_716025


namespace positive_difference_four_or_more_probability_l716_716883

-- Definitions and assumptions based on the conditions
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.off_diag

def valid_pairs (s : Finset ℕ) (d : ℕ) : Finset (ℕ × ℕ) :=
  pairs s |>.filter (λ p => (p.2 - p.1 = d ∧ p.1 < p.2) ∨ (p.1 - p.2 = d ∧ p.2 < p.1))

-- Main theorem
theorem positive_difference_four_or_more_probability :
  (36 - 14 : ℚ) / 36 = 11 / 18 :=
by
  -- Proof part is omitted
  sorry

end positive_difference_four_or_more_probability_l716_716883


namespace eccentricity_of_ellipse_equation_of_ellipse_l716_716495

-- Define the conditions of the problem
def center_at_origin (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0

def ratio_major_minor (a b : ℝ) : Prop :=
  a = (3/2) * b

def focus_on_y_axis (c : ℝ) : Prop :=
  c = 2

-- Theorems to be proven
theorem eccentricity_of_ellipse (a b c e : ℝ) (h_center : center_at_origin 0 0)
  (h_ratio : ratio_major_minor a b) (h_focus : focus_on_y_axis c) (h_formula : c^2 = a^2 - b^2) :
  e = c / a :=
sorry

theorem equation_of_ellipse (a b : ℝ) (h_center : center_at_origin 0 0)
  (h_ratio : ratio_major_minor a b) (h_focus : focus_on_y_axis 2) (h_a : a = 6 / real.sqrt 5)
  (h_b : b^2 = 16 / 5) :
  ∀ x y : ℝ, (5/36) * y^2 + (5/16) * x^2 = 1 :=
sorry

end eccentricity_of_ellipse_equation_of_ellipse_l716_716495


namespace regular_polygon_sides_l716_716212

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716212


namespace train_cross_pole_time_l716_716040

theorem train_cross_pole_time (length : ℕ) (speed_kmh : ℕ) (conversion_factor : ℝ) (length_speed_equal : length * conversion_factor = speed_kmh) : 
  let speed_mps := (speed_kmh : ℝ) * conversion_factor / 3600 in
  let time_to_cross_pole := (length : ℝ) / speed_mps in
  length = 40 → speed_kmh = 144 → conversion_factor = 1 / 3.6 → time_to_cross_pole = 1 :=
by 
{
  intros,
  sorry
}

end train_cross_pole_time_l716_716040


namespace no_real_solution_l716_716792

theorem no_real_solution (P : ℝ → ℝ) (h_cont : Continuous P) (h_no_fixed_point : ∀ x : ℝ, P x ≠ x) : ∀ x : ℝ, P (P x) ≠ x :=
by
  sorry

end no_real_solution_l716_716792


namespace regular_polygon_sides_l716_716173

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716173


namespace angle_between_bisectors_l716_716307

noncomputable def incenter (α β : ℝ) : Prop := sorry

theorem angle_between_bisectors 
  (α β : ℝ) 
  (ABC : Triangle) 
  (A_is_alpha : ABC.angle A = α) 
  (B_is_beta : ABC.angle B = β) 
  (O_center : Point)
  (O_is_incenter : incenter α β)
  (O_1 O_2 O_3 : Point)
  (O_1_bisector_C : is_bisector_exterior_angle O_1 (ABC.angle C))
  (O_2_bisector_A : is_bisector_exterior_angle O_2 (ABC.angle A))
  (O_3_bisector_B : is_bisector_exterior_angle O_3 (ABC.angle B))
  (CO_1 : Line)
  (O_2O_3 : Line)
  (CO_1_line : CO_1 = line_through O_center O_1)
  (O_2O_3_line : O_2O_3 = line_through O_2 O_3) :
  angle_between CO_1 O_2O_3 = 90 :=
sorry

end angle_between_bisectors_l716_716307


namespace f_a_minus_2_lt_0_l716_716683

theorem f_a_minus_2_lt_0 (f : ℝ → ℝ) (m a : ℝ) (h1 : ∀ x, f x = (m + 1 - x) * (x - m + 1)) (h2 : f a > 0) : f (a - 2) < 0 := 
sorry

end f_a_minus_2_lt_0_l716_716683


namespace maintain_mean_and_variance_l716_716419

def initial_set : Finset ℤ :=
  {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset ℤ) : ℚ :=
  (s.sum id : ℚ) / s.card

def variance (s : Finset ℤ) : ℚ :=
  (s.sum (λ x, (x * x : ℚ))) / (s.card : ℚ) - (mean s)^2

theorem maintain_mean_and_variance :
  ∃ (a b c : ℤ), a ∈ initial_set ∧
                 b ∉ initial_set ∧ 
                 c ∉ initial_set ∧ 
                 mean initial_set = mean (initial_set.erase a ∪ {b, c}) ∧
                 variance initial_set = variance (initial_set.erase a ∪ {b, c})
  :=
begin
  sorry
end

end maintain_mean_and_variance_l716_716419


namespace range_of_a_l716_716344

theorem range_of_a (a : ℝ) :
  let A := { x : ℝ | a * x^2 - 2 * x + 1 = 0 }
  in (A = {1} ∨ A = {-1} ∨ A = {1/2} ∨ A = ∅) ↔ (a ≥ 1 ∨ a ≤ -1 ∨ a = 0) :=
by
  sorry

end range_of_a_l716_716344


namespace empire_state_building_height_l716_716463

theorem empire_state_building_height (h_top_floor : ℕ) (h_antenna_spire : ℕ) (total_height : ℕ) :
  h_top_floor = 1250 ∧ h_antenna_spire = 204 ∧ total_height = h_top_floor + h_antenna_spire → total_height = 1454 :=
by
  sorry

end empire_state_building_height_l716_716463


namespace semi_minor_axis_length_l716_716241

theorem semi_minor_axis_length
  (passes_origin : True)
  (foci : (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (hF1 : F1 = (1,0) ) (hF2 : F2 = (3,0))) :
  exists b : ℝ, b = Real.sqrt 3 :=
by 
  sorry

end semi_minor_axis_length_l716_716241


namespace range_of_non_monotonicity_l716_716371

theorem range_of_non_monotonicity (k : ℝ) :
  (∃ x ∈ Set.Ioo k (k + 2), Deriv (λ x => x^3 - 12*x) x = 0) → (-4 < k ∧ k < -2) ∨ (0 < k ∧ k < 2) :=
by 
  sorry

end range_of_non_monotonicity_l716_716371


namespace regular_polygon_sides_l716_716129

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716129


namespace equivalent_problem_l716_716622

noncomputable def problem := ∀ θ : ℝ,
  (θ > π / 2 ∧ θ < π → sin θ * tan θ < 0) ∧
  ¬ (sin θ * tan θ < 0 → θ > π / 2 ∧ θ < π) ∧
  (sin 1 * cos 2 * tan 3 > 0) ∧
  (θ > 3 * π / 2 ∧ θ < 2 * π → sin (π + θ) > 0)

theorem equivalent_problem : problem :=
by
  sorry

end equivalent_problem_l716_716622


namespace replace_one_number_preserves_mean_and_variance_l716_716408

section
open Set

def original_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def new_set (a b c : ℤ) : Set ℤ := 
  if a ∈ original_set then ((original_set.erase a) ∪ {b, c})
  else original_set

def mean (s : Set ℤ) : ℚ := (s.sum id : ℚ) / s.size

def sum_of_squares (s : Set ℤ) : ℚ := ↑(s.sum (λ x, x^2))

theorem replace_one_number_preserves_mean_and_variance :
  ∃ a b c : ℤ, a ∈ original_set ∧ 
    (mean (new_set a b c) = mean original_set) ∧ 
    (sum_of_squares (new_set a b c) = sum_of_squares original_set + 10) :=
  sorry

end

end replace_one_number_preserves_mean_and_variance_l716_716408


namespace log_product_arithmetic_sequence_l716_716760

theorem log_product_arithmetic_sequence (a : ℕ → ℝ)
    (h1 : ∀ n, a (n + 1) = a n + d)
    (h2 : a 4 + a 5 = 4) :
  Real.log 2 (2 ^ a 0 * 2 ^ a 1 * 2 ^ a 2 * 2 ^ a 3 * 2 ^ a 4 * 2 ^ a 5 * 2 ^ a 6 * 2 ^ a 7 * 2 ^ a 8 * 2 ^ a 9) = 20 :=
begin
  sorry
end

end log_product_arithmetic_sequence_l716_716760


namespace initial_money_is_500_l716_716031

-- Define the conditions
variable (M : ℝ)  -- Initial amount of money

-- Condition: After spending 1/3 on clothes, remaining money
def remainingAfterClothes := (2 / 3) * M

-- Condition: After spending 1/5 of remaining on food, remaining money
def remainingAfterFood := (4 / 5) * remainingAfterClothes

-- Condition: After spending 1/4 of remaining on travel, remaining money
def remainingAfterTravel := (3 / 4) * remainingAfterFood

-- Condition: Remaining money is Rs 200
def finalAmount := 200

-- The proof statement
theorem initial_money_is_500 (h : remainingAfterTravel M = finalAmount) : M = 500 :=
by
  sorry

end initial_money_is_500_l716_716031


namespace locus_circumcircles_thales_circles_l716_716007

noncomputable theory

open EuclideanGeometry

-- Define the three distinct points
variables {K L M : Point}

-- Define the condition for isosceles right triangles
def is_right_isosceles_triangle (A B C : Point) : Prop :=
  is_right ∠A B C ∧ dist A C = dist A B ∧ dist A C = dist B C

-- Define the locus of the circumcenters 
def locus_of_circumcenters (K L M : Point) : Set Point :=
  {O | ∃ (A B C : Point), 
    is_right_isosceles_triangle A B C ∧ 
    A = K ∧ 
    B = L ∧ 
    collinear {A, C, M} ∧
    O = midpoint C (reflection C M)} 

theorem locus_circumcircles_thales_circles :
  locus_of_circumcenters K L M = thales_circles_union :=
sorry

end locus_circumcircles_thales_circles_l716_716007


namespace digit_421_of_15_over_26_l716_716013

theorem digit_421_of_15_over_26 : 
  (421 % 6 = 1) → ((\frac{15}{26} : ℚ) = 0.576923) → ("576923".to_list.nth 0 = some '5') :=
sorry

end digit_421_of_15_over_26_l716_716013


namespace problem_part_I_problem_part_II_l716_716333

noncomputable def f (a : ℝ) (x : ℝ) := (Real.sin x)^2 + a * (Real.cos x)^2

theorem problem_part_I (a : ℝ) (ha : a = -2) :
  ∀ x, f a (x + Real.pi) = f a x :=
begin
  intro x,
  -- We need to assert f(a, x) and f(a, x + π) are the same given a = -2.
  have h_f : f a x = (Real.sin x)^2 - 2 * (Real.cos x)^2,
  { rw ha, refl },
  calc f a (x + Real.pi) = (Real.sin (x + Real.pi))^2 - 2 * (Real.cos (x + Real.pi))^2 : by rw h_f
                    ... = (Real.sin x)^2 - 2 * (-(Real.cos x))^2 : by simp [Real.sin_add, Real.cos_add]
                    ... = (Real.sin x)^2 - 2 * (Real.cos x)^2 : by rw [neg_square, mul_neg_eq_neg_mul_symm]
                    ... = f a x : by rw h_f,
end

theorem problem_part_II (a : ℝ) (ha : a = -2) :
  ∀ x ∈ Icc (Real.pi / 24) (11 * Real.pi / 24),
    -sqrt 2 / 2 - 1 ≤ f a x ∧ f a x ≤ sqrt 2 - 1 :=
begin
  intros x hx,
  split,
  { -- The minimum value on the interval
    calc -sqrt 2 / 2 - 1 = sqrt 2 * -1 / 2 - 1 : by rw neg_div
                    ... ≤ sqrt 2 * (Real.sin (2 * x - Real.pi / 4)) - 1 : sorry -- exact inequality
                    ... = f a x : sorry
  },
  { -- The maximum value on the interval
    calc f a x = -Real.cos (2 * x) + (Real.sin x)^2 - 1 : sorry -- by definition after substitution
           ... ≤ sqrt 2 - sqrt 2 - 1 : sorry
           ... = sqrt 2 - 1 : sorry,
  },
end

#check problem_part_I    -- check the first theorem
#check problem_part_II   -- check the second theorem

end problem_part_I_problem_part_II_l716_716333


namespace tom_reading_problem_l716_716652

theorem tom_reading_problem :
  ∀ (initial_speed : ℕ) (increase_factor : ℕ) (time_hours : ℕ),
    initial_speed = 12 →
    increase_factor = 3 →
    time_hours = 2 →
    (initial_speed * increase_factor * time_hours = 72) :=
by
  intros initial_speed increase_factor time_hours h_initial h_increase h_time
  rw [h_initial, h_increase, h_time]
  norm_num
  sorry

end tom_reading_problem_l716_716652


namespace right_triangle_area_l716_716015

def hypotenuse := 17
def leg1 := 15
def leg2 := 8
def area := (1 / 2:ℝ) * leg1 * leg2 

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hl1 : a = 15) (hl2 : c = 17) :
  area = 60 := by 
  sorry

end right_triangle_area_l716_716015


namespace largest_integer_not_exceeding_500x_l716_716616

noncomputable def cube_shadow_problem : ℕ :=
  let edge_length := 2
  let height_above := 3
  let shadow_add_area := 98
  let total_shadow_area := shadow_add_area + edge_length * edge_length
  let side_length_shadow := real.sqrt total_shadow_area
  let x := side_length_shadow - edge_length
  let result := 500 * x
  nat.floor result

theorem largest_integer_not_exceeding_500x : cube_shadow_problem = 4050 := 
sorry

end largest_integer_not_exceeding_500x_l716_716616


namespace shaded_percentage_l716_716546

def is_shaded (i j : ℕ) : Prop := (i + j) % 2 = 1

theorem shaded_percentage (n : ℕ) (h : n = 6) : 
  (Σ i, Σ j, if is_shaded i j then 1 else 0).val / (n * n) * 100 = 50 :=
by
  sorry

end shaded_percentage_l716_716546


namespace regular_polygon_sides_l716_716134

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716134


namespace shifted_parabola_l716_716957

theorem shifted_parabola (x : ℝ) : 
  (let y := x^2 in (let x := x + 2 in y)) = (x - 2)^2 := sorry

end shifted_parabola_l716_716957


namespace leila_total_cakes_l716_716789

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 :=
by sorry

end leila_total_cakes_l716_716789


namespace jack_estimate_larger_l716_716879

variable {x y a b : ℝ}

theorem jack_estimate_larger (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (ha : 0 < a) (hb : 0 < b) : 
  (x + a) - (y - b) > x - y :=
by
  sorry

end jack_estimate_larger_l716_716879


namespace equation_D_is_linear_l716_716895

-- Definitions according to the given conditions
def equation_A (x y : ℝ) := x + 2 * y = 3
def equation_B (x : ℝ) := 3 * x - 2
def equation_C (x : ℝ) := x^2 + x = 6
def equation_D (x : ℝ) := (1 / 3) * x - 2 = 3

-- Properties of a linear equation
def is_linear (eq : ℝ → Prop) : Prop :=
∃ a b c : ℝ, (∃ x : ℝ, eq x = (a * x + b = c)) ∧ a ≠ 0

-- Specifying that equation_D is linear
theorem equation_D_is_linear : is_linear equation_D :=
by
  sorry

end equation_D_is_linear_l716_716895


namespace sum_x_y_eq_9_l716_716705

theorem sum_x_y_eq_9 (x y : ℝ) (h1 : x ≥ 5) (h2 : x ≤ 5) (h3 : y = real.sqrt (x - 5) + real.sqrt (5 - x) + 4) : 
  x + y = 9 := 
sorry

end sum_x_y_eq_9_l716_716705


namespace count_elements_in_A_l716_716452

def A : Set (Fin 10 → ℤ) := 
  {x | ∀ i : Fin 10, x i ∈ {-1, 0, 1}}

def count_satisfying_elements : ℕ := 
  Finset.card {x ∈ Finset.univ.filter (λ x : Fin 10 → ℤ, 
    1 ≤ (Finset.univ : Finset (Fin 10)).sum (λ i, |x i|) ∧ 
    (Finset.univ : Finset (Fin 10)).sum (λ i, |x i|) ≤ 9)}.toFinset

theorem count_elements_in_A :
  count_satisfying_elements = 3^10 - 2^10 - 1 :=
sorry

end count_elements_in_A_l716_716452


namespace regular_polygon_sides_l716_716138

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716138


namespace boat_speed_in_still_water_l716_716556

-- Define the conditions
def speed_of_stream : ℝ := 3 -- (speed in km/h)
def time_downstream : ℝ := 1 -- (time in hours)
def time_upstream : ℝ := 1.5 -- (time in hours)

-- Define the goal by proving the speed of the boat in still water
theorem boat_speed_in_still_water : 
  ∃ V_b : ℝ, (V_b + speed_of_stream) * time_downstream = (V_b - speed_of_stream) * time_upstream ∧ V_b = 15 :=
by
  sorry -- (Proof will be provided here)

end boat_speed_in_still_water_l716_716556


namespace sum_of_2004_elements_l716_716568

theorem sum_of_2004_elements (N: ℕ): ∃ N2004: ℕ, ∀ n: ℕ, n ≥ N2004 → 
  ∃ (a: Fin 2004 → ℕ), (∀ i: Fin 2003, a i < a ⟨i.1 + 1, Nat.lt_of_succ_lt_succ i.2⟩) ∧
  (∀ i: Fin 2003, a i ∣ a ⟨i.1 + 1, Nat.lt_of_succ_lt_succ i.2⟩) ∧ 
  (n = ∑ i: Fin 2004, a i) := sorry

end sum_of_2004_elements_l716_716568


namespace quadratic_function_inequality_l716_716715

def quadratic_symmetry (b c : ℝ) : Prop :=
  ∀ x : ℝ, f(x) = x^2 + b*x + c

theorem quadratic_function_inequality {b c : ℝ} (h_symmetry : ∀ x : ℝ, f(x) = x^2 + b*x + c) :
  let f (x: ℝ) := x^2 + b * x + c in
  f(1) < f(2) ∧ f(2) < f(-1) := by
  sorry

end quadratic_function_inequality_l716_716715


namespace relationship_abc_l716_716681

noncomputable def a : ℝ := 2 ^ (-4)
noncomputable def b : ℝ := 4 ^ 2.1
noncomputable def c : ℝ := Real.log (0.125) / Real.log (4)

theorem relationship_abc : c < a ∧ a < b := by
  sorry

end relationship_abc_l716_716681


namespace length_BC_l716_716382

namespace TriangleLength

-- Definitions
variables {A B C : Type} [acute_triangle A B C]
variable (area_ABC : ℝ)
variable (AB : ℝ)
variable (AC : ℝ)

-- Assumptions
def conditions := (area_ABC = 10 * real.sqrt 3) ∧ (AB = 8) ∧ (AC = 5)

-- Theorem to prove
theorem length_BC : conditions area_ABC AB AC → ∃ BC : ℝ, BC = 7 :=
  by {
    assume h : conditions area_ABC AB AC,
    sorry
  }
  
end TriangleLength

end length_BC_l716_716382


namespace x_proportionality_find_x_value_l716_716706

theorem x_proportionality (m n : ℝ) (x z : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h3 : x = 4) (h4 : z = 8) :
  ∃ k, ∀ z : ℝ, x = k / z^8 := 
sorry

theorem find_x_value (m n : ℝ) (k : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h5 : k = 67108864) :
  ∀ z, (z = 32 → x = 1 / 16) :=
sorry

end x_proportionality_find_x_value_l716_716706


namespace lcm_of_denominators_l716_716861

theorem lcm_of_denominators (x : ℕ) [NeZero x] : Nat.lcm (Nat.lcm x (2 * x)) (3 * x^2) = 6 * x^2 :=
by
  sorry

end lcm_of_denominators_l716_716861


namespace xiao_zhang_net_displacement_xiao_zhang_profit_l716_716842

noncomputable def distances : List ℤ := [-13, 6, 9, -12, 11, -9, 6]

def net_displacement (distances : List ℤ) : ℤ := distances.sum

def total_distance (distances : List ℤ) : ℤ := distances.map Int.natAbs |>.sum

def revenue_per_kilometer : ℝ := 2.9
def cost_per_kilometer : ℝ := 1.2

def profit (distance : ℤ) (revenue_per_km : ℝ) (cost_per_km : ℝ) : ℝ :=
  let distance_float := distance.nat_abs.toFloat
  (distance_float * (revenue_per_km - cost_per_km)).to_real

theorem xiao_zhang_net_displacement :
  net_displacement distances = -2 := 
  by
  sorry

theorem xiao_zhang_profit :
  profit (total_distance distances) revenue_per_kilometer cost_per_kilometer = 112.2 := 
  by
  sorry

end xiao_zhang_net_displacement_xiao_zhang_profit_l716_716842


namespace real_roots_polynomials_l716_716998

noncomputable def poly_deg1 := [Polynomial.C 1 + Polynomial.X, -Polynomial.X + Polynomial.C 1]
noncomputable def poly_deg2 := [Polynomial.X^2 + Polynomial.X - Polynomial.C 1, Polynomial.X^2 - Polynomial.X - Polynomial.C 1]
noncomputable def poly_deg3 := [Polynomial.X^3 + Polynomial.X^2 - Polynomial.X - Polynomial.C 1, Polynomial.X^3 - Polynomial.X^2 - Polynomial.X + Polynomial.C 1]

theorem real_roots_polynomials :
  ∀ (p : Polynomial ℝ), (p.coeffs = [1, 1, -1] ∨ p.coeffs = [1, -1, -1] ∨ 
                          p.coeffs = [1, 0, -1] ∨ p.coeffs = [-1, 0, -1] ∨
                          p.coeffs = [1, 1, 0, -1] ∨ p.coeffs = [1, -1, 0, 1] ∨
                          p.coeffs = [-1, 1, 0, -1] ∨ p.coeffs = [-1, -1, 0, 1]) ↔
                          (∀ (x : ℝ), p.is_root x) :=
begin
  sorry
end

end real_roots_polynomials_l716_716998


namespace tom_reading_problem_l716_716651

theorem tom_reading_problem :
  ∀ (initial_speed : ℕ) (increase_factor : ℕ) (time_hours : ℕ),
    initial_speed = 12 →
    increase_factor = 3 →
    time_hours = 2 →
    (initial_speed * increase_factor * time_hours = 72) :=
by
  intros initial_speed increase_factor time_hours h_initial h_increase h_time
  rw [h_initial, h_increase, h_time]
  norm_num
  sorry

end tom_reading_problem_l716_716651


namespace value_of_f_neg_1_expression_for_f_x_lt_0_l716_716322

variable (f : ℝ → ℝ)

noncomputable def isOddFunction := ∀ x : ℝ, f(-x) = -f(x)
noncomputable def f_x_positive (x : ℝ) := f x = x^2 - 4*x + 3

theorem value_of_f_neg_1 (h1 : isOddFunction f) (h2 : ∀ x > 0, f(x) = x^2 - 4*x + 3) : f (-1) = 0 :=
by
  sorry

theorem expression_for_f_x_lt_0 (h1 : isOddFunction f) (h2 : ∀ x > 0, f(x) = x^2 - 4*x + 3) : ∀ x < 0, f x = -x^2 - 4*x - 3 :=
by
  sorry

end value_of_f_neg_1_expression_for_f_x_lt_0_l716_716322


namespace sum_inv_S_eq_l716_716797

open Nat

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℕ := 2 + (n - 1) * 2
def S (n : ℕ) : ℕ := n * (n + 1)

-- Define the sequence 1/S(n)
def inv_S (n : ℕ) : ℚ := 1 / (S n)

-- Sum of the first 10 terms of 1/S(n)
def T_10 : ℚ := ∑ i in range 10, inv_S (i + 1)

theorem sum_inv_S_eq : T_10 = 10 / 11 := by
  sorry

end sum_inv_S_eq_l716_716797


namespace incorrect_average_l716_716540

theorem incorrect_average :
  let numbers := [12, 13, 14, 510, 520, 530, 1115, 1, 1252140, 2345]
  let supposed_average := 858.5454545454545
  let count := 10
  let sum := 1253190
  ∑ n in numbers, n = 1253190 →
  supposed_average ≠ sum / count := by
  intro h
  have calculated_average : sum / count = 125319 := by norm_num
  sorry

end incorrect_average_l716_716540


namespace middle_integer_is_five_l716_716517

-- Define the conditions of the problem
def consecutive_one_digit_positive_odd_integers (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  a + 2 = b ∧ b + 2 = c ∨ a + 2 = c ∧ c + 2 = b

def sum_is_one_seventh_of_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 7

-- Define the theorem to prove
theorem middle_integer_is_five :
  ∃ (b : ℤ), consecutive_one_digit_positive_odd_integers (b - 2) b (b + 2) ∧
             sum_is_one_seventh_of_product (b - 2) b (b + 2) ∧
             b = 5 :=
sorry

end middle_integer_is_five_l716_716517


namespace trig_identity_l716_716971

-- The problem statement
theorem trig_identity : 
  sin(15 * Real.pi / 180) * sin(105 * Real.pi / 180) - cos(15 * Real.pi / 180) * cos(105 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_l716_716971


namespace regular_polygon_sides_l716_716157

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716157


namespace shift_parabola_two_units_right_l716_716952

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end shift_parabola_two_units_right_l716_716952


namespace coefficient_of_aminus1_in_binomial_expansion_l716_716392

theorem coefficient_of_aminus1_in_binomial_expansion :
  let general_term := fun (k : ℕ) => (Nat.choose 8 k) * (-2)^k * a^(8 - 3*k/2)
  8 - 3 * 6 / 2 = -1 →
  (Nat.choose 8 6) * (-2)^6 = 1792 := 
by
  intros general_term
  sorry

end coefficient_of_aminus1_in_binomial_expansion_l716_716392


namespace find_a_minus_square_l716_716298

theorem find_a_minus_square (a : ℝ) (h1 : |2004 - a| + real.sqrt (a - 2005) = a) (h2 : a ≥ 2005) : a - 2004^2 = 2005 :=
by
  sorry

end find_a_minus_square_l716_716298


namespace parabola_directrix_equation_l716_716868

theorem parabola_directrix_equation (h₀ : 0 < 8) :
  ∀ (x : ℝ) (y : ℝ), x = 4 → y^2 = -16 * x := 
by
  intros x y hx
  rw hx
  sorry

end parabola_directrix_equation_l716_716868


namespace triangle_semicircle_inequality_l716_716531

-- Step d): Define the problem statement in Lean 4
theorem triangle_semicircle_inequality 
  (r : ℝ) (A B C : ℝ × ℝ)
  (h_circle : ∀ P : ℝ × ℝ, P = A ∨ P = B ∨ P = C → P.1^2 + P.2^2 = r^2)
  (h_diameter : distance A B = 2 * r)
  (h_C_rb : C ≠ A ∧ C ≠ B)
  (AC : ℝ := distance A C)
  (BC : ℝ := distance B C)
  (s : ℝ := AC + BC) :
  s^2 ≤ 8 * r^2 :=
sorry

end triangle_semicircle_inequality_l716_716531


namespace solution_is_correct_l716_716347

-- Define the vector AB
def vectorAB : ℝ × ℝ := (3, 7)

-- Define the vector BC
def vectorBC : ℝ × ℝ := (-2, 3)

-- Compute the vector AC based on AB and BC
def vectorAC : ℝ × ℝ := (vectorAB.1 + vectorBC.1, vectorAB.2 + vectorBC.2)

-- Compute -1/2 * vector AC
def negHalfVectorAC : ℝ × ℝ := (-1/2 * vectorAC.1, -1/2 * vectorAC.2)

-- Theorem proving the equivalence
theorem solution_is_correct : negHalfVectorAC = (-1/2, -5) :=
by
  -- Placeholder proof
  sorry

end solution_is_correct_l716_716347


namespace john_needs_more_usd_l716_716428

noncomputable def additional_usd (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ) : ℝ :=
  let eur_to_usd := 1 / 0.84
  let sgd_to_usd := 1 / 1.34
  let jpy_to_usd := 1 / 110.35
  let total_needed_usd := needed_eur * eur_to_usd + needed_sgd * sgd_to_usd
  let total_has_usd := has_usd + has_jpy * jpy_to_usd
  total_needed_usd - total_has_usd

theorem john_needs_more_usd :
  ∀ (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ),
    needed_eur = 7.50 → needed_sgd = 5.00 → has_usd = 2.00 → has_jpy = 500 →
    additional_usd needed_eur needed_sgd has_usd has_jpy = 6.13 :=
by
  intros needed_eur needed_sgd has_usd has_jpy
  intros hneeded_eur hneeded_sgd hhas_usd hhas_jpy
  unfold additional_usd
  rw [hneeded_eur, hneeded_sgd, hhas_usd, hhas_jpy]
  sorry

end john_needs_more_usd_l716_716428


namespace blue_face_probability_l716_716376

def sides : ℕ := 12
def green_faces : ℕ := 5
def blue_faces : ℕ := 4
def red_faces : ℕ := 3

theorem blue_face_probability : 
  (blue_faces : ℚ) / sides = 1 / 3 :=
by
  sorry

end blue_face_probability_l716_716376


namespace num_mappings_l716_716805

theorem num_mappings (f : ℕ → ℤ) (h1 : ∀ x, x ∈ {1, 2, 3}) (h2 : ∀ x ∈ {1, 2, 3}, f x ∈ {-1, 0, 1})
  (h3 : f 3 = f 1 + f 2) : 
  ∃ count : ℕ, count = 7 := 
sorry

end num_mappings_l716_716805


namespace side_length_of_box_is_correct_l716_716547

noncomputable def costPerBox := 0.70
noncomputable def totalCost := 357.0
noncomputable def totalVolume := 3060000.0

-- Number of boxes needed
noncomputable def numberOfBoxes := totalCost / costPerBox

-- Volume per box
noncomputable def volumePerBox := totalVolume / numberOfBoxes

-- Side length of a cubic box
noncomputable def sideLength := volumePerBox^(1/3:ℝ)

theorem side_length_of_box_is_correct : sideLength = 18.17 := by
  sorry

end side_length_of_box_is_correct_l716_716547


namespace canoes_more_than_kayaks_l716_716042

theorem canoes_more_than_kayaks (C K : ℕ)
  (h1 : 14 * C + 15 * K = 288)
  (h2 : C = 3 * K / 2) :
  C - K = 4 :=
sorry

end canoes_more_than_kayaks_l716_716042


namespace solve_integer_equation_l716_716549

theorem solve_integer_equation : ∃ (x : ℕ), (2 * x)^3 - x = 726 ∧ x = 6 :=
by
  use 6
  split
  · sorry -- proof that (2 * 6)^3 - 6 = 726
  · rfl

end solve_integer_equation_l716_716549


namespace unique_root_in_first_quadrant_l716_716497

open Complex

theorem unique_root_in_first_quadrant : 
  ∃ z : ℂ, (z ^ 7 = -1 + sqrt 3 * Complex.I) ∧ 
           (arg z = 2 * real.pi / 21) ∧ 
           (re z > 0 ∧ im z > 0) ∧ 
           (∀ k ∈ Finset.range 7, 
            ¬ (arg (z * exp(Complex.I * 2 * k * real.pi / 7)) > 0 ∧ 
              arg (z * exp(Complex.I * 2 * k * real.pi / 7)) < real.pi / 2  ∧ 
              z * exp(Complex.I * 2 * k * real.pi / 7) ≠ z)) :=
by 
sorry

end unique_root_in_first_quadrant_l716_716497


namespace div_equal_octagons_l716_716773

-- Definitions based on the conditions
def squareArea (n : ℕ) := n * n
def isDivisor (m n : ℕ) := n % m = 0

-- Main statement
theorem div_equal_octagons (n : ℕ) (hn : n = 8) :
  (2 ∣ squareArea n) ∨ (4 ∣ squareArea n) ∨ (8 ∣ squareArea n) ∨ (16 ∣ squareArea n) :=
by
  -- We shall show the divisibility aspect later.
  sorry

end div_equal_octagons_l716_716773


namespace prove_f_neg_a_l716_716456

def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 11) : f (-a) = -9 := 
  by sorry

end prove_f_neg_a_l716_716456


namespace roots_quadratic_l716_716899

theorem roots_quadratic (a b : ℝ) (h : ∀ x : ℝ, x^2 - 7 * x + 7 = 0 → (x = a) ∨ (x = b)) :
  a^2 + b^2 = 35 :=
sorry

end roots_quadratic_l716_716899


namespace application_outcomes_l716_716008

theorem application_outcomes :
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  (choices_A * choices_B * choices_C) = 18 :=
by
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  show (choices_A * choices_B * choices_C = 18)
  sorry

end application_outcomes_l716_716008


namespace tesla_ratio_l716_716655

variables (s c e : ℕ)
variables (h1 : e = s + 10) (h2 : c = 6) (h3 : e = 13)

theorem tesla_ratio : s / c = 1 / 2 :=
by
  sorry

end tesla_ratio_l716_716655


namespace find_f_of_5_l716_716810

theorem find_f_of_5 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, 2 * x^2 - 4 * x + 3 ≤ f(x) ∧ f(x) ≤ 3 * x^2 - 6 * x + 4) 
  (h2 : f(3) = 11) : 
  f(5) = 41 := 
  sorry

end find_f_of_5_l716_716810


namespace regular_polygon_sides_l716_716118

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716118


namespace regular_polygon_sides_l716_716174

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716174


namespace regular_polygon_sides_l716_716104

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716104


namespace regular_polygon_sides_l716_716179

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716179


namespace area_and_perimeter_of_traced_shape_l716_716504

theorem area_and_perimeter_of_traced_shape 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) :
  let ρ := (a + b - c) / 2 in
  let r := ρ / 2 in
  let t := (1 / 2) * a * b in
  let ts := (3 / 8) * t + ((ρ^2 * π) / 4) in
  let Ps := (a + b + c) / 2 + (ρ * π) / 2 in
  ∃(area perimeter : ℝ), 
    area = (3 / 8) * a * b + ((a + b - c)^2 * π) / 16 ∧ 
    perimeter = (a + b + c) / 2 + ((a + b - c) * π) / 2 :=
sorry

end area_and_perimeter_of_traced_shape_l716_716504


namespace correct_statements_l716_716235

-- A quality inspector takes a sample from a uniformly moving production line every 10 minutes for a certain indicator test.
def statement1 := false -- This statement is incorrect because this is systematic sampling, not stratified sampling.

-- In the frequency distribution histogram, the sum of the areas of all small rectangles is 1.
def statement2 := true -- This is correct.

-- In the regression line equation \(\hat{y} = 0.2x + 12\), when the variable \(x\) increases by one unit, the variable \(y\) definitely increases by 0.2 units.
def statement3 := false -- This is incorrect because y increases on average by 0.2 units, not definitely.

-- For two categorical variables \(X\) and \(Y\), calculating the statistic \(K^2\) and its observed value \(k\), the larger the observed value \(k\), the more confident we are that “X and Y are related”.
def statement4 := true -- This is correct.

-- We need to prove that the correct statements are only statement2 and statement4.
theorem correct_statements : (statement1 = false ∧ statement2 = true ∧ statement3 = false ∧ statement4 = true) → (statement2 ∧ statement4) :=
by sorry

end correct_statements_l716_716235


namespace inequality_proof_l716_716682

theorem inequality_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a ^ 0.6 < a ^ 0.4) :
  let m := 0.6 * log a 0.6,
      n := 0.4 * log a 0.6,
      p := 0.6 * log a 0.4 in
  p > m ∧ m > n :=
by
  sorry

end inequality_proof_l716_716682


namespace adam_completes_work_in_10_days_l716_716819

theorem adam_completes_work_in_10_days (W : ℝ) (A : ℝ)
  (h1 : (W / 25) + A = W / 20) :
  W / 10 = (W / 100) * 10 :=
by
  sorry

end adam_completes_work_in_10_days_l716_716819


namespace square_table_production_l716_716942

theorem square_table_production (x y : ℝ) :
  x + y = 5 ∧ 50 * x * 4 = 300 * y → 
  x = 3 ∧ y = 2 ∧ 50 * x = 150 :=
by
  sorry

end square_table_production_l716_716942


namespace regular_polygon_sides_l716_716088

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716088


namespace regular_polygon_sides_l716_716178

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716178


namespace regular_polygon_sides_l716_716222

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716222


namespace find_a_odd_function_l716_716703

theorem find_a_odd_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, 0 < x → f x = 1 + a^x)
  (h3 : 0 < a)
  (h4 : a ≠ 1)
  (h5 : f (-1) = -3 / 2) :
  a = 1 / 2 :=
by
  sorry

end find_a_odd_function_l716_716703


namespace arithmetic_sequence_S13_l716_716050

theorem arithmetic_sequence_S13 (a1 a7 a13 : ℤ) (h : a1 + a7 + a13 = 6) : 
  let S13 := 13 * (a1 + a7 + a13) / 6 in
  S13 = 26 := 
by 
  intro S13
  sorry

end arithmetic_sequence_S13_l716_716050


namespace surface_area_of_circumscribed_sphere_l716_716224

-- Define the conditions:
def regular_tetrahedron_edge := sqrt 2

-- Explaining that the tetrahedron can fit inside a cube
-- with the circumscribed sphere having the same radius as the diagonal of the cube
def cube_edge_quantity (a : ℝ) := (sqrt 2) / 2 * a

-- Define the radius of the circumscribed sphere
def circumscribed_radius (a : ℝ) : ℝ := 1 / 2 * sqrt ((1 / 2 * a^2) + 
                                                   (1 / 2 * a^2) + 
                                                   (1 / 2 * a^2))

-- Define the surface area of the circumscribed sphere
def surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

open Real (sqrt pi)

-- Prove the statement:
theorem surface_area_of_circumscribed_sphere : 
  let a := regular_tetrahedron_edge
  let R := circumscribed_radius a
  surface_area (R * sqrt 2) = 3 * pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l716_716224


namespace value_of_a_minus_b_l716_716363

theorem value_of_a_minus_b (a b c : ℝ) 
    (h1 : 2011 * a + 2015 * b + c = 2021)
    (h2 : 2013 * a + 2017 * b + c = 2023)
    (h3 : 2012 * a + 2016 * b + 2 * c = 2026) : 
    a - b = -2 := 
by
  sorry

end value_of_a_minus_b_l716_716363


namespace value_of_b_l716_716319

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end value_of_b_l716_716319


namespace passenger_rides_each_car_once_l716_716557

noncomputable theory

open ProbabilityTheory

variable (Ω : Type) [ProbabilitySpace Ω]

/-- The probability that a passenger will ride in each of the 2 cars exactly once given two rides. -/
theorem passenger_rides_each_car_once (Rides : (Fin 2) → Ω) (eventA eventB : Event Ω) 
  (hA : eventA = {ω | (Rides 0) ω = 0 ∨ (Rides 1) ω = 1}) 
  (hB : eventB = {ω | (Rides 0) ω = 1 ∨ (Rides 1) ω = 0}) 
  (hIndependent : indep (λ _, Rides 0) (λ _, Rides 1)) :
  (condProb eventA {ω | true}) = 1/2 :=
by sorry

end passenger_rides_each_car_once_l716_716557


namespace binary_representation_253_l716_716988

theorem binary_representation_253 (x y : ℕ) (h : nat.bits 253 = [1, 1, 1, 1, 1, 1, 0, 1]) :
  (y - x = 6) :=
begin
  have hx : x = 1, from by {
    -- x is the number of 0's in the list representation [1, 1, 1, 1, 1, 1, 0, 1]
    sorry
  },
  have hy : y = 7, from by {
    -- y is the number of 1's in the list representation [1, 1, 1, 1, 1, 1, 0, 1]
    sorry
  },
  rw [hx, hy],
  exact rfl,
end

end binary_representation_253_l716_716988


namespace solution_1_solution_2_l716_716335

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |x - 3|

theorem solution_1 (x : ℝ) : (f x (-1) >= 2) ↔ (x >= 2) :=
by
  sorry

theorem solution_2 (a : ℝ) : 
  (∃ x : ℝ, f x a <= -(a / 2)) ↔ (a <= 2 ∨ a >= 6) :=
by
  sorry

end solution_1_solution_2_l716_716335


namespace consecutive_integers_sum_divisor_sq_l716_716869

theorem consecutive_integers_sum_divisor_sq (n : ℕ) (h_pos : 0 < n) :
  let sum := 10 * n + 45
      sum_of_squares := 10 * n^2 + 90 * n + 285
  in sum_of_squares % sum = 0 → (n = 1 ∨ n = 12) :=
sorry

end consecutive_integers_sum_divisor_sq_l716_716869


namespace regular_polygon_sides_l716_716107

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716107


namespace smallest_whole_number_larger_than_perimeter_l716_716024

theorem smallest_whole_number_larger_than_perimeter {s : ℝ} (h1 : 16 < s) (h2 : s < 30) :
  61 > 7 + 23 + s :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l716_716024


namespace regular_polygon_sides_l716_716209

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716209


namespace regular_polygon_sides_l716_716089

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716089


namespace distinct_nat_pairs_identity_l716_716361

theorem distinct_nat_pairs_identity :
  let p := 11, q := 6, r := 5,
      p₁ := 10, q₁ := 9, r₁ := 1 in
  p^2 + q^2 + r^2 = p₁^2 + q₁^2 + r₁^2 ∧
  p^4 + q^4 + r^4 = p₁^4 + q₁^4 + r₁^4 := 
by
  sorry

end distinct_nat_pairs_identity_l716_716361


namespace square_diagonal_l716_716745

theorem square_diagonal (s d : ℝ) (h : 4 * s = 40) : d = s * Real.sqrt 2 → d = 10 * Real.sqrt 2 :=
by
  sorry

end square_diagonal_l716_716745


namespace domain_of_function_l716_716661

def domain_function : set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3 ∧ x ≠ 0}

theorem domain_of_function :
  (∀ x : ℝ, 9 - x^2 ≥ 0 → x + 1 > 0 → log 2 (x + 1) ≠ 0 → x ∈ domain_function) :=
by
  intros x h1 h2 h3
  -- Explanation:
  -- h1 : 9 - x^2 ≥ 0 (from the first condition)
  -- h2 : x + 1 > 0  (from the second condition)
  -- h3 : log 2 (x + 1) ≠ 0 (from the third condition)
  sorry

end domain_of_function_l716_716661


namespace claire_has_gerbils_l716_716638

-- Definitions based on conditions
variables (G H : ℕ)
variables (h1 : G + H = 90) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25)

-- Main statement to prove
theorem claire_has_gerbils : G = 60 :=
sorry

end claire_has_gerbils_l716_716638


namespace mean_and_variance_unchanged_with_replacement_l716_716413

def S : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

structure Replacement :=
  (a b c : ℤ)
  (is_in_S : a ∈ S)
  (b_plus_c_eq_a : b + c = a)
  (b_sq_plus_c_sq_minus_a_sq_eq_ten : b^2 + c^2 - a^2 = 10)

def replacements : List Replacement :=
  [ { a := 4, b := -1, c := 5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num },
    { a := -4, b := 1, c := -5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num } ]

theorem mean_and_variance_unchanged_with_replacement (r : Replacement) :
  -- assuming r comes from the list of replacements
  ∃ (r ∈ replacements), r = r :=
begin
  sorry
end

end mean_and_variance_unchanged_with_replacement_l716_716413


namespace person_walk_distance_l716_716226

theorem person_walk_distance (m : ℝ) (l : ℝ) (stone_mass : ℝ) (board_overhang : ℝ) : 
    l = 24 → stone_mass = 2 * m → board_overhang = (2 / 3) * l → 
    ∃ (x : ℝ), x = 20 :=
by
  intros h1 h2 h3 
  use 20
  have h4 : x = 20, sorry
  exact h4

end person_walk_distance_l716_716226


namespace jackies_lotion_bottles_l716_716778

theorem jackies_lotion_bottles (L: ℕ) : 
  (10 + 10) + 6 * L + 12 = 50 → L = 3 :=
by
  sorry

end jackies_lotion_bottles_l716_716778


namespace square_side_bound_l716_716941

theorem square_side_bound {ABC : Triangle} {s r : ℝ} 
  (h1 : InscribedSquareInTriangle ABC s)
  (h2 : InscribedCircleRadius ABC r) :
  sqrt 2 * r < s ∧ s < 2 * r :=
sorry

end square_side_bound_l716_716941


namespace grocer_coffee_problem_l716_716060

theorem grocer_coffee_problem :
  let initial_stock := 400
  let initial_decaf_percentage := 0.20
  let second_batch_decaf_percentage := 0.60
  let total_final_decaf_percentage := 0.28000000000000004
  let d1 := initial_stock * initial_decaf_percentage
  let total_coffee_stock x := initial_stock + x
  let total_decaf_coffee x := d1 + second_batch_decaf_percentage * x
  ∃ x, (total_decaf_coffee x) / (total_coffee_stock x) = total_final_decaf_percentage → x = 100 :=
begin
  sorry
end

end grocer_coffee_problem_l716_716060


namespace smallest_number_with_ten_divisors_l716_716865

/-- 
  Theorem: The smallest natural number n that has exactly 10 positive divisors is 48.
--/
theorem smallest_number_with_ten_divisors : 
  ∃ (n : ℕ), (∀ (p1 p2 p3 p4 p5 : ℕ) (a1 a2 a3 a4 a5 : ℕ), 
    n = p1^a1 * p2^a2 * p3^a3 * p4^a4 * p5^a5 → 
    n.factors.count = 10) 
    ∧ n = 48 := sorry

end smallest_number_with_ten_divisors_l716_716865


namespace find_valid_n_l716_716659

open Nat

noncomputable def is_prime (n : ℕ) : Prop := sorry

def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧ is_prime (n^n + 1) ∧ (n^n + 1 < 10^18)

theorem find_valid_n : { n : ℕ // satisfies_conditions n } = {1, 2, 4} :=
  sorry

end find_valid_n_l716_716659


namespace complex_real_imag_opposites_l716_716373

theorem complex_real_imag_opposites (b : ℝ) : 
  let z : ℂ := (2 - complex.I * b) / (1 + 2 * complex.I) in
  (z.re = -z.im) → b = -2 / 3 := sorry

end complex_real_imag_opposites_l716_716373


namespace charles_housesitting_hours_l716_716974

theorem charles_housesitting_hours :
  ∀ (earnings_per_hour_housesitting earnings_per_hour_walking_dog number_of_dogs_walked total_earnings : ℕ),
  earnings_per_hour_housesitting = 15 →
  earnings_per_hour_walking_dog = 22 →
  number_of_dogs_walked = 3 →
  total_earnings = 216 →
  ∃ h : ℕ, 15 * h + 22 * 3 = 216 ∧ h = 10 :=
by
  intros
  sorry

end charles_housesitting_hours_l716_716974


namespace fraction_of_number_add_6_is_11_10_results_in_fraction_l716_716589

theorem fraction_of_number_add_6_is_11_10_results_in_fraction :
  ∃ f : ℚ, f * 10 + 6 = 11 ∧ f = 1 / 2 :=
by
  use 1 / 2
  split
  · calc
    (1 / 2) * 10 + 6 = 5 + 6 := by norm_num
    ... = 11 := by norm_num
  · rfl

end fraction_of_number_add_6_is_11_10_results_in_fraction_l716_716589


namespace common_property_of_rhombus_and_rectangle_l716_716859

structure Rhombus :=
  (bisect_perpendicular : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_not_equal : ∀ d₁ d₂ : ℝ, ¬(d₁ = d₂))

structure Rectangle :=
  (bisect_each_other : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_equal : ∀ d₁ d₂ : ℝ, d₁ = d₂)

theorem common_property_of_rhombus_and_rectangle (R : Rhombus) (S : Rectangle) :
  ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0) :=
by
  -- Assuming the properties of Rhombus R and Rectangle S
  sorry

end common_property_of_rhombus_and_rectangle_l716_716859


namespace area_of_smaller_circle_l716_716882

theorem area_of_smaller_circle (r R : ℝ) (hR : R = 2 * r) (PA AB : ℝ) 
  (h_PA : PA = 5) (h_AB : AB = 5) :
  ∃ (area : ℝ), area = \frac{25}{8} * Real.pi ∧ area = Real.pi * r^2 := 
by
  use (\frac{25}{8} * Real.pi)
  split
  sorry
  sorry

end area_of_smaller_circle_l716_716882


namespace probability_penny_nickel_heads_l716_716491

noncomputable def num_outcomes : ℕ := 2^4
noncomputable def num_successful_outcomes : ℕ := 2 * 2

theorem probability_penny_nickel_heads :
  (num_successful_outcomes : ℚ) / num_outcomes = 1 / 4 :=
by
  sorry

end probability_penny_nickel_heads_l716_716491


namespace total_animals_l716_716522

theorem total_animals (initial_elephants initial_hippos : ℕ) 
  (ratio_female_hippos : ℚ)
  (births_per_female_hippo : ℕ)
  (newborn_elephants_diff : ℕ)
  (he : initial_elephants = 20)
  (hh : initial_hippos = 35)
  (rfh : ratio_female_hippos = 5 / 7)
  (bpfh : births_per_female_hippo = 5)
  (ned : newborn_elephants_diff = 10) :
  ∃ (total_animals : ℕ), total_animals = 315 :=
by sorry

end total_animals_l716_716522


namespace solve_for_t_l716_716485

theorem solve_for_t (t : ℝ) : 4 * 4^t + real.sqrt (16 * 16^t) = 64 ↔ t = 3 / 2 := 
by
  sorry

end solve_for_t_l716_716485


namespace regular_polygon_sides_l716_716214

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716214


namespace regular_polygon_sides_l716_716133

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716133


namespace Doris_needs_3_weeks_l716_716650

-- Definitions based on conditions
def hourly_wage : ℕ := 20
def monthly_expenses : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturdays_hours : ℕ := 5
def weekdays_per_week : ℕ := 5

-- Total hours per week
def total_hours_per_week := (weekday_hours_per_day * weekdays_per_week) + saturdays_hours

-- Weekly earnings
def weekly_earnings := hourly_wage * total_hours_per_week

-- Number of weeks needed for monthly expenses
def weeks_needed := monthly_expenses / weekly_earnings

-- Proposition to prove
theorem Doris_needs_3_weeks :
  weeks_needed = 3 := 
by
  sorry

end Doris_needs_3_weeks_l716_716650


namespace electric_car_travel_distance_l716_716064

theorem electric_car_travel_distance {d_electric d_diesel : ℕ} 
  (h1 : d_diesel = 120) 
  (h2 : d_electric = d_diesel + 50 * d_diesel / 100) : 
  d_electric = 180 := 
by 
  sorry

end electric_car_travel_distance_l716_716064


namespace beetle_percent_less_distance_l716_716964

theorem beetle_percent_less_distance
  (ant_distance_m : ℕ := 600)
  (time_minutes : ℕ := 10)
  (beetle_speed_kmh : ℝ := 2.7) :
  let ant_speed_kmh := (ant_distance_m : ℝ) * 6 / 1000
  in (((ant_speed_kmh - beetle_speed_kmh) / ant_speed_kmh) * 100) = 25 := 
by
  -- Placeholder for the proof
  sorry

end beetle_percent_less_distance_l716_716964


namespace intersect_curves_l716_716509

theorem intersect_curves (R : ℝ) (hR : R > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x - y - 2 = 0) ↔ R ≥ Real.sqrt 2 :=
sorry

end intersect_curves_l716_716509


namespace john_purchased_grinder_l716_716429

noncomputable def purchase_price_of_grinder (profit : ℝ) (mobile_purchase : ℝ) (mobile_profit : ℝ) (mobile_sell : ℝ) : ℝ :=
  have grinder_sell : ℝ := (G - 0.04 * G)
  have mobile_sell : ℝ := (mobile_purchase + mobile_profit)
  G

theorem john_purchased_grinder (G : ℝ)
  (mobile_purchase : ℝ) (mobile_sell : ℝ) (profit : ℝ)
  (grinder_loss_percent : ℝ) (mobile_profit_percent : ℝ)
  (total_profit : ℝ) (total_sell_price : ℝ) (total_purchase_price : ℝ) :
  (mobile_purchase = 8000) →
  (grinder_loss_percent = 0.04) →
  (mobile_profit_percent = 0.15) →
  (total_profit = 600) →
  (total_sell_price = 0.96 * G + 9200) →
  (total_purchase_price = G + 8000) →
  total_sell_price = total_purchase_price + total_profit →
  G = 15000 :=
by
  sorry

end john_purchased_grinder_l716_716429


namespace parallel_lines_slope_l716_716545

-- Define the given conditions
def line1_slope (x : ℝ) : ℝ := 6
def line2_slope (c : ℝ) (x : ℝ) : ℝ := 3 * c

-- State the proof problem
theorem parallel_lines_slope (c : ℝ) : 
  (∀ x : ℝ, line1_slope x = line2_slope c x) → c = 2 :=
by
  intro h
  -- Intro provides a human-readable variable and corresponding proof obligation
  -- The remainder of the proof would follow here, but instead,
  -- we use "sorry" to indicate an incomplete proof
  sorry

end parallel_lines_slope_l716_716545


namespace regular_polygon_sides_l716_716187

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716187


namespace regular_polygon_sides_l716_716183

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716183


namespace exists_segment_with_many_proper_intersections_l716_716430

open Function Set

-- Define the set P and its properties
variable {P : Set (ℝ × ℝ)}
variable (n : ℕ)
variable (P_finite : P.finite)
variable (P_size : P_card = n)
variable (P_at_least_two : n ≥ 2)
variable (no_three_collinear : ∀ (A B C ∈ P), ¬ Collinear A B C)

-- Define the set S of all segments with endpoints in P
def Segment (A B : (ℝ × ℝ)) := {A, B}

def S := { s | ∃ A B ∈ P, s = Segment A B }

-- Define the intersection property
def intersects_properly (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ P ∈ s1 ∩ s2, P ∉ s1 ∩ s2 ∧ P ∉ endpoints s1 ∩ endpoints s2

-- Main theorem
theorem exists_segment_with_many_proper_intersections :
  ∃ s0 ∈ S, (card { s ∈ S | intersects_properly s0 s } ≥ (1 / 15) * (binom (n-2) 2)) :=
sorry

end exists_segment_with_many_proper_intersections_l716_716430


namespace broken_line_traversal_impossible_l716_716043

theorem broken_line_traversal_impossible (n : ℕ) (h : n = 100) :
  ¬ ∃ (line : list (ℕ × ℕ)), (∀ s ∈ line, odd (segment_length s)) ∧ (∀ (s1 s2 : (ℕ × ℕ)), consecutive s1 s2 line → is_perpendicular s1 s2) ∧ (non_self_intersecting line) ∧ (passes_through_all_vertices line (n × n)) :=
by
  sorry

end broken_line_traversal_impossible_l716_716043


namespace prob_one_first_class_is_correct_prob_second_class_is_correct_l716_716578

noncomputable def total_pens : ℕ := 6
noncomputable def first_class_pens : ℕ := 4
noncomputable def second_class_pens : ℕ := 2
noncomputable def draws : ℕ := 2

noncomputable def total_ways : ℕ := choose total_pens draws

noncomputable def exactly_one_first_class_ways : ℕ := (choose first_class_pens 1) * (choose second_class_pens 1)
noncomputable def prob_exactly_one_first_class : ℚ := exactly_one_first_class_ways / total_ways

noncomputable def at_least_one_second_class_ways : ℕ := (choose total_pens draws) - (choose first_class_pens 2)
noncomputable def prob_at_least_one_second_class : ℚ := at_least_one_second_class_ways / total_ways

theorem prob_one_first_class_is_correct :
  prob_exactly_one_first_class = 8 / 15 := by
  sorry

theorem prob_second_class_is_correct :
  prob_at_least_one_second_class = 3 / 5 := by
  sorry

end prob_one_first_class_is_correct_prob_second_class_is_correct_l716_716578


namespace move_digit_produces_ratio_l716_716285

theorem move_digit_produces_ratio
  (a b : ℕ)
  (h_original_eq : ∃ x : ℕ, x = 10 * a + b)
  (h_new_eq : ∀ (n : ℕ), 10^n * b + a = (3 * (10 * a + b)) / 2):
  285714 = 10 * a + b :=
by
  -- proof steps would go here
  sorry

end move_digit_produces_ratio_l716_716285


namespace cos_sum_identity_l716_716830

theorem cos_sum_identity :
  (cos 0) + (cos (2 * Real.pi / 5)) + (cos (4 * Real.pi / 5)) +
  (cos (6 * Real.pi / 5)) + (cos (8 * Real.pi / 5)) = 0 →
  (cos (2 * Real.pi / 5)) + (cos (4 * Real.pi / 5)) = -1 / 2 :=
by
  sorry

end cos_sum_identity_l716_716830


namespace shift_parabola_l716_716958

theorem shift_parabola (x : ℝ) : 
    let y := x^2 in 
    ∃ (x' : ℝ), (x' = x - 2) ∧ (y = x'^2) :=
    sorry

end shift_parabola_l716_716958


namespace grandson_age_is_5_l716_716850

-- Definitions based on the conditions
def grandson_age_months_eq_grandmother_years (V B : ℕ) : Prop := B = 12 * V
def combined_age_eq_65 (V B : ℕ) : Prop := B + V = 65

-- Main theorem stating that under these conditions, the grandson's age is 5 years
theorem grandson_age_is_5 (V B : ℕ) (h₁ : grandson_age_months_eq_grandmother_years V B) (h₂ : combined_age_eq_65 V B) : V = 5 :=
by sorry

end grandson_age_is_5_l716_716850


namespace total_cats_left_l716_716604

noncomputable def initial_siamese : ℕ := 38
noncomputable def initial_house : ℕ := 25
noncomputable def initial_persian : ℕ := 15
noncomputable def initial_maine_coon : ℕ := 12

noncomputable def sold_siamese_percentage : ℚ := 0.60
noncomputable def sold_house_percentage : ℚ := 0.40
noncomputable def sold_persian_percentage : ℚ := 0.75
noncomputable def sold_maine_coon_percentage : ℚ := 0.50

noncomputable def sold_cats (initial : ℕ) (percentage : ℚ) : ℕ :=
  floor (percentage * initial)

noncomputable def cats_left (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

theorem total_cats_left :
  let
    siamese_sold := sold_cats initial_siamese sold_siamese_percentage,
    house_sold := sold_cats initial_house sold_house_percentage,
    persian_sold := sold_cats initial_persian sold_persian_percentage,
    maine_coon_sold := sold_cats initial_maine_coon sold_maine_coon_percentage,
    siamese_left := cats_left initial_siamese siamese_sold,
    house_left := cats_left initial_house house_sold,
    persian_left := cats_left initial_persian persian_sold,
    maine_coon_left := cats_left initial_maine_coon maine_coon_sold
  in
    siamese_left + house_left + persian_left + maine_coon_left = 41 := by
  sorry

end total_cats_left_l716_716604


namespace unique_not_in_range_of_g_l716_716498

noncomputable def g (m n p q : ℝ) (x : ℝ) : ℝ := (m * x + n) / (p * x + q)

theorem unique_not_in_range_of_g (m n p q : ℝ) (hne1 : m ≠ 0) (hne2 : n ≠ 0) (hne3 : p ≠ 0) (hne4 : q ≠ 0)
  (h₁ : g m n p q 23 = 23) (h₂ : g m n p q 53 = 53) (h₃ : ∀ (x : ℝ), x ≠ -q / p → g m n p q (g m n p q x) = x) :
  ∃! x : ℝ, ¬ ∃ y : ℝ, g m n p q y = x ∧ x = -38 :=
sorry

end unique_not_in_range_of_g_l716_716498


namespace midpoints_concyclic_l716_716800

variables {A B C D E X Y : Point}
variables (Γ : Circle)

-- Given conditions
hypothesis (hABCinscribed : inscribed_triangle Γ A B C) 
hypothesis (hXYonΓ : on_circle X Γ ∧ on_circle Y Γ) 
hypothesis (hXYintersectsAB : intersects XY AB D) 
hypothesis (hXYintersectsAC : intersects XY AC E)

-- Midpoints definitions
def midpoint (P Q : Point) : Point := sorry -- Placeholder for the midpoint definition

let K := midpoint B E
let L := midpoint C D
let M := midpoint D E
let N := midpoint X Y

-- Theorem statement
theorem midpoints_concyclic : concyclic_points K L M N :=
sorry

end midpoints_concyclic_l716_716800


namespace decreasing_condition_l716_716569

open Real

theorem decreasing_condition (n : ℕ) :
  (∀ x > 0, deriv (λ x, (n^2 - 3*n + 3) * x^(2*n - 3)) x < 0) ↔ (n = 1) := sorry

end decreasing_condition_l716_716569


namespace range_f_l716_716704

-- Define the function f and its properties
def f (x : ℝ) : ℝ :=
if h : x <= 0 then (Real.exp x - 1)
else -(Real.exp (-x) - 1)

-- State the theorem
theorem range_f : Set.range f = Set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end range_f_l716_716704


namespace regular_polygon_sides_l716_716205

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716205


namespace sum_of_two_numbers_l716_716515

theorem sum_of_two_numbers (S L : ℝ) (h1 : S = 10.0) (h2 : 7 * S = 5 * L) : S + L = 24.0 :=
by
  -- proof goes here
  sorry

end sum_of_two_numbers_l716_716515


namespace regular_polygon_sides_l716_716140

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716140


namespace expected_value_is_20_point_5_l716_716603

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def coin_heads_probability : ℚ := 1 / 2

noncomputable def expected_value : ℚ :=
  coin_heads_probability * (penny_value + nickel_value + dime_value + quarter_value)

theorem expected_value_is_20_point_5 :
  expected_value = 20.5 := by
  sorry

end expected_value_is_20_point_5_l716_716603


namespace smallest_addition_to_divisible_l716_716542

theorem smallest_addition_to_divisible (n : ℕ) (p : ℕ) (remainder : ℕ) (to_add : ℕ) : 
  n = 956734 → 
  p = 751 → 
  n % p = remainder → 
  to_add = p - remainder → 
  n + to_add % p = 0 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end smallest_addition_to_divisible_l716_716542


namespace proposition_p_equiv_l716_716855

theorem proposition_p_equiv (p : Prop) :
  (¬ p ↔ ∀ x : ℝ, 0 < x → sqrt x > x + 1) →
  (p ↔ ∃ x : ℝ, 0 < x ∧ sqrt x ≤ x + 1) :=
by sorry

end proposition_p_equiv_l716_716855


namespace all_three_use_media_l716_716963

variable (U T R M T_and_M T_and_R R_and_M T_and_R_and_M : ℕ)

theorem all_three_use_media (hU : U = 180)
  (hT : T = 115)
  (hR : R = 110)
  (hM : M = 130)
  (hT_and_M : T_and_M = 85)
  (hT_and_R : T_and_R = 75)
  (hR_and_M : R_and_M = 95)
  (h_union : U = T + R + M - T_and_R - T_and_M - R_and_M + T_and_R_and_M) :
  T_and_R_and_M = 80 :=
by
  sorry

end all_three_use_media_l716_716963


namespace regular_polygon_sides_l716_716105

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716105


namespace blueberry_pies_count_l716_716818

theorem blueberry_pies_count :
  ∃ (n : ℕ), 
    (let peach_pies := 5;
         apple_pies := 4;
         peach_cost_per_pound := 2;
         apple_cost_per_pound := 1;
         blueberry_cost_per_pound := 1;
         pounds_per_pie := 3;
         total_spent := 51;
         total_peach_cost := peach_pies * peach_cost_per_pound * pounds_per_pie;
         total_apple_cost := apple_pies * apple_cost_per_pound * pounds_per_pie;
         total_fruit_cost := total_spent;
         total_blueberry_cost := total_fruit_cost - (total_peach_cost + total_apple_cost);
         blueberry_pie_cost := blueberry_cost_per_pound * pounds_per_pie in
    n * blueberry_pie_cost = total_blueberry_cost)
    :=
begin
  use 3,
  sorry
end

end blueberry_pies_count_l716_716818


namespace shortest_remaining_side_l716_716613

theorem shortest_remaining_side (a b : ℝ) (h1 : a = 7) (h2 : b = 24) (right_triangle : ∃ c, c^2 = a^2 + b^2) : a = 7 :=
by
  sorry

end shortest_remaining_side_l716_716613


namespace increase_in_area_l716_716527

theorem increase_in_area :
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  area_increase = 13 :=
by
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  sorry

end increase_in_area_l716_716527


namespace number_of_students_in_class_l716_716873

theorem number_of_students_in_class (S : ℕ) 
  (h1 : ∀ n : ℕ, 4 * n ≠ 0 → S % 4 = 0) -- S is divisible by 4
  (h2 : ∀ G : ℕ, 3 * G ≠ 0 → (S * 3) % 4 = G) -- Number of students who went to the playground (3/4 * S) is integer
  (h3 : ∀ B : ℕ, G - B ≠ 0 → (G * 2) / 3 = 10) -- Number of girls on the playground
  : S = 20 := sorry

end number_of_students_in_class_l716_716873


namespace enclosed_area_eq_2pi_l716_716890

theorem enclosed_area_eq_2pi :
  ∀ x y : ℝ, (x-1)^2 + (y-1)^2 = 2 * |x-1| + 2 * |y-1| → 
  ∃ r : ℝ, r = 2 ∧ (x-1)^2 + (y-1)^2 = r * 2π :=
sorry

end enclosed_area_eq_2pi_l716_716890


namespace necessary_but_not_sufficient_for_parallel_l716_716692

-- Define the conditions
variables {α : Type*} [sm : add_comm_group α] (plane : s)
variables {l : Type*} (A B : α)

-- Distance function from a point to a plane
variable (dist_to_plane : α → ℝ)

-- Conditions: A and B are distinct points on line l
-- distances of A and B to the plane are 'a' and 'b' respectively
variable (a b : ℝ)
hypothesis h_dist_A : dist_to_plane A = a
hypothesis h_dist_B : dist_to_plane B = b

-- We need to prove that "a = b" is a necessary but not sufficient condition for "l ∥ plane"
theorem necessary_but_not_sufficient_for_parallel (h : ∀ x y, dist_to_plane x = dist_to_plane y → x = y): 
  a = b → ∀ x y : α, x = y := sorry

end necessary_but_not_sufficient_for_parallel_l716_716692


namespace locus_of_P_l716_716712

variables {x y : ℝ}
variables {x0 y0 : ℝ}

-- The initial ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 16 = 1

-- Point M is on the ellipse
def point_M (x0 y0 : ℝ) : Prop :=
  ellipse x0 y0

-- The equation of P, symmetric to transformations applied to point Q derived from M
theorem locus_of_P 
  (hx0 : x0^2 / 20 + y0^2 / 16 = 1) :
  ∃ x y, (x^2 / 20 + y^2 / 36 = 1) ∧ y ≠ 0 :=
sorry

end locus_of_P_l716_716712


namespace shift_parabola_l716_716960

theorem shift_parabola (x : ℝ) : 
    let y := x^2 in 
    ∃ (x' : ℝ), (x' = x - 2) ∧ (y = x'^2) :=
    sorry

end shift_parabola_l716_716960


namespace min_abs_E_l716_716541

noncomputable def E (x : ℝ) : ℝ := sorry

theorem min_abs_E (h : ∀ x : ℝ, abs (E x) + abs (x + 6) + abs (x - 5) ≥ 11) : ∃ x : ℝ, abs (E x) = 0 := by
  have h₀ : abs (E (-0.5)) + abs (-0.5 + 6) + abs (-0.5 - 5) = 11 := sorry
  have h₁ : abs (-0.5 + 6) = 5.5 := sorry
  have h₂ : abs (-0.5 - 5) = 5.5 := sorry
  have h₃ : abs (E (-0.5)) = 0 := by
    linarith [h₀, h₁, h₂]
  use -0.5
  exact h₃
  sorry

end min_abs_E_l716_716541


namespace regular_polygon_sides_l716_716142

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716142


namespace solution_to_equation_l716_716866

theorem solution_to_equation :
  ∀ x : ℝ, (abs (sqrt ((x - 2) ^ 2) - 1) = x) ↔ (x = 1 / 2) :=
by
  intro x
  sorry

end solution_to_equation_l716_716866


namespace division_subtraction_l716_716968

theorem division_subtraction : 144 / (12 / 3) - 5 = 31 := by
  sorry

end division_subtraction_l716_716968


namespace corn_height_after_10_weeks_l716_716634

def growth_week1 : ℝ := 2

def growth_week2 : ℝ := 2 * growth_week1

def growth_week3 : ℝ := 4 * growth_week2

def growth_week4 : ℝ := growth_week1 + growth_week2 + growth_week3

def growth_week5 : ℝ := (growth_week4 / 2) - 3

def growth_week6 : ℝ := 2 * growth_week5

def average_growth_first_six_weeks : ℝ :=
  (growth_week1 + growth_week2 + growth_week3 + growth_week4 + growth_week5 + growth_week6) / 6

def growth_week7 : ℝ := average_growth_first_six_weeks + 1

def growth_week8 : ℝ := growth_week7 - 5

def growth_week9 : ℝ := growth_week5 + growth_week6

def growth_week10 : ℝ := growth_week9 * 1.5

def total_growth_10_weeks : ℝ :=
  growth_week1 + growth_week2 + growth_week3 + growth_week4 + growth_week5 +
  growth_week6 + growth_week7 + growth_week8 + growth_week9 + growth_week10

theorem corn_height_after_10_weeks : total_growth_10_weeks = 147.66 := by
  sorry

end corn_height_after_10_weeks_l716_716634


namespace regular_polygon_sides_l716_716144

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716144


namespace determine_b_minus_a_l716_716903

theorem determine_b_minus_a (a b : ℝ) (h : {a, 1} = {0, a + b}) : b - a = 1 := 
sorry

end determine_b_minus_a_l716_716903


namespace increasing_interval_a_geq_neg2_l716_716339

theorem increasing_interval_a_geq_neg2
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + 2 * (a - 2) * x + 5)
  (h_inc : ∀ x > 4, f (x + 1) > f x) :
  a ≥ -2 :=
sorry

end increasing_interval_a_geq_neg2_l716_716339


namespace square_root_combination_l716_716374

theorem square_root_combination (a : ℝ) (h : 1 + a = 4 - 2 * a) : a = 1 :=
by
  -- proof goes here
  sorry

end square_root_combination_l716_716374


namespace exists_line_with_projection_less_than_two_over_pi_l716_716695

theorem exists_line_with_projection_less_than_two_over_pi (segments : set (ℝ × ℝ)) (h_fin : segments.finite) (h_len : (∑ s in segments, (√((s.2.1 - s.1.1)^2 + (s.2.2 - s.1.2)^2))) = 1) :
  ∃ l : ℝ × ℝ, (∑ s in segments, |(s.2.1 - s.1.1) * l.1 + (s.2.2 - s.1.2) * l.2| / (√(l.1^2 + l.2^2))) < (2 / Real.pi) := 
by
  sorry

end exists_line_with_projection_less_than_two_over_pi_l716_716695


namespace find_parabola_and_ellipse_l716_716327

noncomputable def parabola_vertex (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

noncomputable def ellipse_eq (a b x y : ℝ) : Prop :=
  x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1

noncomputable def intersects_at (curve1 curve2 : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

noncomputable def eccentricity_relation (a b p : ℝ) : Prop :=
  let e_parabola := 1 / 2 in
  let c := real.sqrt (a ^ 2 - b ^ 2) in
  let e_ellipse := c / a in
  e_ellipse = e_parabola / 2

theorem find_parabola_and_ellipse
  (a b : ℝ) (M_x M_y : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : intersects_at (parabola_vertex 2) (ellipse_eq a b) M_x M_y) 
  (h4 : eccentricity_relation a b 2) :
  parabola_vertex 2 1 1 ∧ ellipse_eq a b 1 1 :=
  sorry

end find_parabola_and_ellipse_l716_716327


namespace cos_given_condition_l716_716295

theorem cos_given_condition (α : ℝ) (h : sin (π / 3 + α) = 1 / 3) : cos (5 * π / 6 + α) = - (1 / 3) :=
by
  sorry

end cos_given_condition_l716_716295


namespace parallel_lines_condition_l716_716045

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y: ℝ, (x + a * y + 6 = 0) ↔ ((a - 2) * x + 3 * y + 2 * a = 0)) ↔ a = -1 :=
by
  sorry

end parallel_lines_condition_l716_716045


namespace double_bed_heavier_l716_716871

-- Define the problem conditions
variable (S D B : ℝ)
variable (h1 : 5 * S = 50)
variable (h2 : 2 * S + 4 * D + 3 * B = 180)
variable (h3 : 3 * B = 60)

-- Define the goal to prove
theorem double_bed_heavier (S D B : ℝ) (h1 : 5 * S = 50) (h2 : 2 * S + 4 * D + 3 * B = 180) (h3 : 3 * B = 60) : D - S = 15 :=
by
  sorry

end double_bed_heavier_l716_716871


namespace total_students_l716_716532

-- Define two-thirds of the class having brown eyes
def brown_eyes (T : ℤ) : Prop :=
  (2 / 3 : ℤ) * T = 2 * T / 3

-- Define half of the students with brown eyes have black hair
def black_hair_given_brown_eyes (T : ℤ) : Prop :=
  (1 / 2 : ℤ) * (2 * T / 3) = (1 / 3 : ℤ) * T

-- Given 6 students have brown eyes and black hair
def students_with_brown_eyes_black_hair : Prop :=
  (1 / 3 : ℤ) * ?m_1 = 6

-- The theorem to prove
theorem total_students (T : ℤ) (H1 : brown_eyes T) (H2 : black_hair_given_brown_eyes T) (H3 : students_with_brown_eyes_black_hair T) : T = 18 := by
  sorry

end total_students_l716_716532


namespace tangent_circle_bisector_l716_716626

theorem tangent_circle_bisector (C₁ C₂ : Circle) (P : Point) (A B C : Point) 
  (HP : tangent C₁ C₂ P) (HA₁ : tangent C₁ A) (HA₂ : intersects C₂ A B C) :
  (external_tangent C₁ C₂ → is_external_bisector (angle B P C) (line P A)) ∧
  (internal_tangent C₁ C₂ → is_internal_bisector (angle B P C) (line P A)) :=
sorry

end tangent_circle_bisector_l716_716626


namespace regular_polygon_sides_l716_716208

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716208


namespace regular_polygon_sides_l716_716132

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716132


namespace ellipse_standard_equation_line_pass_through_midpoint_l716_716813

noncomputable def ellipse_equation 
(a b : ℝ) (h1 : a > b) (h2 : b > 0) (B : ℝ × ℝ) (F : ℝ × ℝ) 
(hB : B = (0, 2)) (hF : F = (-2, 0)) (h_line : ∀ (x y : ℝ), x - y + 2 = 0 ↔ x = y - 2) :
  Prop := 
  (B.1 - F.1)^2 + (B.2 - F.2)^2 = a^2 + b^2

theorem ellipse_standard_equation 
(a b : ℝ) (h1 : a > b) (h2 : b > 0) (B : ℝ × ℝ) (F : ℝ × ℝ)
(hB : B = (0, 2)) (hF : F = (-2, 0)) (h_line : ∀ (x y : ℝ), x - y + 2 = 0 ↔ x = y - 2) :
  ellipse_equation a b h1 h2 B F hB h_line = (∃ (E : ℝ × ℝ → Prop), E = (λ x, x.1^2 / 8 + x.2^2 / 4 = 1)) :=
sorry

noncomputable def line_equation 
(P Q A : ℝ × ℝ) (h_mid : A = (-1, 1)) 
(x1 y1 x2 y2 : ℝ) (h_PQ : ∀ (x y : ℝ), 
(P = (x1, y1) ∧ Q = (x2, y2)) ∧ 
  (P.1 + Q.1 = 2 * A.1) ∧ 
  (P.2 + Q.2 = 2 * A.2)) : 
  Prop := 
  ∀ (x y : ℝ), P = (x, y) → Q = (x, y) → 
    (A.1 + x) / 2 = - 1 ∧ 
    (A.2 + y) / 2 = 1

theorem line_pass_through_midpoint
(P Q A : ℝ × ℝ) (h_mid : A = (-1, 1)) 
(x1 y1 x2 y2 : ℝ) (h_PQ : ∀ (x y : ℝ), 
(P = (x1, y1) ∧ Q = (x2, y2)) ∧ 
  (P.1 + Q.1 = 2 * A.1) ∧ 
  (P.2 + Q.2 = 2 * A.2)) :
  line_equation P Q A h_mid x1 y1 x2 y2 h_PQ = (∃ (l : ℝ × ℝ → Prop), l = (λ x, x.1 - 2 * x.2 + 3 = 0)) :=
sorry

end ellipse_standard_equation_line_pass_through_midpoint_l716_716813


namespace fraction_equals_decimal_l716_716640

def fraction := 1 / 4
def decimal := 0.250000000

theorem fraction_equals_decimal : fraction = decimal := 
by 
  sorry

end fraction_equals_decimal_l716_716640


namespace sum_of_two_is_zero_l716_716465

theorem sum_of_two_is_zero (x y z t : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ht : t ≠ 0)
  (h₁ : x + y + z + t = 0) 
  (h₂ : (1 / x) + (1 / y) + (1 / z) + (1 / t) = 0) :
  ∃ a b, a ∈ ({x, y, z, t} : set ℝ) ∧ b ∈ ({x, y, z, t} : set ℝ) ∧ a + b = 0 := 
by 
  sorry

end sum_of_two_is_zero_l716_716465


namespace minimum_value_of_a_l716_716679

theorem minimum_value_of_a (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2 - b * c) 
  (h2 : (1/2) * b * c * (Real.sin A) = (3 * Real.sqrt 3) / 4)
  (h3 : A = Real.arccos (1/2)) :
  a ≥ Real.sqrt 3 := sorry

end minimum_value_of_a_l716_716679


namespace train_distance_in_2_hours_l716_716057

-- Definitions of conditions
def time_per_interval : ℝ := 2.25 -- time in minutes per interval
def distance_per_interval : ℝ := 2 -- distance in miles per interval
def total_time : ℝ := 120 -- total time in minutes

-- Theorem statement
theorem train_distance_in_2_hours :
  (total_time / time_per_interval).to_int * distance_per_interval = 106 := sorry

end train_distance_in_2_hours_l716_716057


namespace rival_awards_l716_716779

theorem rival_awards (jessie_multiple : ℕ) (scott_awards : ℕ) (rival_multiple : ℕ) 
  (h1 : jessie_multiple = 3) 
  (h2 : scott_awards = 4) 
  (h3 : rival_multiple = 2) 
  : (rival_multiple * (jessie_multiple * scott_awards) = 24) :=
by 
  sorry

end rival_awards_l716_716779


namespace find_greatest_a_l716_716662

theorem find_greatest_a
  (a : ℝ)
  (h : (\frac{9 * (√((3 * a)^2 + 1)) - 9 * a^2 - 1}{√(1 + 3 * a^2) + 2} = 3)) :
  a <= √(13 / 3) :=
sorry

end find_greatest_a_l716_716662


namespace range_of_m_l716_716370

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, sin x ^ 2 + 2 * sin x - 1 + m = 0) → (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l716_716370


namespace replace_one_number_preserves_mean_and_variance_l716_716407

section
open Set

def original_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def new_set (a b c : ℤ) : Set ℤ := 
  if a ∈ original_set then ((original_set.erase a) ∪ {b, c})
  else original_set

def mean (s : Set ℤ) : ℚ := (s.sum id : ℚ) / s.size

def sum_of_squares (s : Set ℤ) : ℚ := ↑(s.sum (λ x, x^2))

theorem replace_one_number_preserves_mean_and_variance :
  ∃ a b c : ℤ, a ∈ original_set ∧ 
    (mean (new_set a b c) = mean original_set) ∧ 
    (sum_of_squares (new_set a b c) = sum_of_squares original_set + 10) :=
  sorry

end

end replace_one_number_preserves_mean_and_variance_l716_716407


namespace regular_polygon_sides_l716_716092

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716092


namespace problem1_problem2_l716_716350

-- Definitions of conditions and vectors
def vec_m (x : ℝ) := (sqrt 3 * sin x, sin x)
def vec_n (x : ℝ) := (cos x, sin x)

-- Given vectors m and n are parallel, and x is in [0, π/2]
theorem problem1 (x : ℝ) (h1 : x ∈ Icc 0 (π / 2)) (h2 : (vec_m x).1 / (vec_n x).1 = (vec_m x).2 / (vec_n x).2) :
  x = π / 6 ∨ x = 0 :=
sorry

-- Define the function f(x)
def f (x : ℝ) := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2

-- Given a function f(x), prove the smallest positive period and the interval of monotonic increase
theorem problem2 :
  (∃ T, T = π ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3  → f' x > 0) :=
sorry

end problem1_problem2_l716_716350


namespace imon_no_entanglement_l716_716601

-- Definition of an imon and operations on imons
noncomputable def imon := ℕ
def is_entangled (G : SimpleGraph imon) (v1 v2 : imon) := G.Adj v1 v2

def remove_imon (G : SimpleGraph imon) (v : imon) : SimpleGraph imon :=
  G.induce {x | x ≠ v}

def duplicate_imons (G : SimpleGraph imon) : SimpleGraph (imon × bool) :=
  let new_verts := {v | true}
  have : ∀ v1 v2, G.Adj v1 v2 → G.Adj (v1, tt) (v2, tt), from assume _ _ h, h,
  G.map (λ v, (v, tt)) ∪ G.map (λ v, (v, ff)) ∪
  {((v, tt), (v, ff)) | v ∈ vertexSet G}

-- The main theorem statement
theorem imon_no_entanglement (G : SimpleGraph imon) : ∃ n : ℕ, ∀ G' : SimpleGraph imon, 
  (G'.order = 0) → G'.induce (set.mulServer G.vertexSet n) = ∅ :=
sorry

end imon_no_entanglement_l716_716601


namespace international_society_proof_l716_716966

noncomputable def members : Finset ℕ := Finset.range (1978 + 1)

-- Define the condition that members are divided into 6 countries
def in_country (member : ℕ) (country : Finset ℕ) : Prop := member ∈ country

-- Define the proof that there exists a member whose number is the sum of 
-- the numbers of two other compatriots or is twice the number of another compatriot.
theorem international_society_proof :
  ∃ (country : Finset ℕ),
    (∀ m ∈ country, ∃ x y ∈ country, m = x + y ∨ m = 2 * x) :=
by
  -- Placeholder for proof; details of proof to be filled in
  sorry

end international_society_proof_l716_716966


namespace interest_rate_Y_is_17_l716_716462

open Real

-- Definitions for conditions
def total_investment : Real := 100000
def interest_rate_X : Real := 0.23
def additional_interest_Y : Real := 200
def investment_in_X : Real := 42000

-- Derived definitions
def investment_in_Y : Real := total_investment - investment_in_X
def interest_from_X : Real := interest_rate_X * investment_in_X
def interest_from_Y : Real := interest_from_X + additional_interest_Y
def interest_rate_Y : Real := (interest_from_Y / investment_in_Y) * 100

theorem interest_rate_Y_is_17 :
  interest_rate_Y = 17 := by
  -- Proof is omitted
  sorry

end interest_rate_Y_is_17_l716_716462


namespace train_pass_time_correct_l716_716039

noncomputable def train_time_to_pass_post (length_of_train : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  length_of_train / speed_mps

theorem train_pass_time_correct :
  train_time_to_pass_post 60 36 = 6 := by
  sorry

end train_pass_time_correct_l716_716039


namespace triathlon_speed_requirements_l716_716460

theorem triathlon_speed_requirements (x : ℝ) (h : x > 0) : 
  (800 / x) + (4000 / (3 * x)) + (20000 / (7.5 * x)) ≤ 80 ↔ 
  (x ≥ 100) ∧ (3 * x ≥ 300) ∧ (7.5 * x ≥ 750) :=
by
  have h1 : (800 / x) + (4000 / (3 * x)) + (20000 / (7.5 * x)) = 80 := /* Step from solution */
    sorry,
  have h2 : x = 100 := /* Step solving for x */
    sorry,
  exact Iff.intro
    (fun h => 
      have : x ≥ 100 := by sorry,
      have : 3 * x ≥ 300 := by sorry,
      have : 7.5 * x ≥ 750 := by sorry,
      ⟨this, this, this⟩)
    (fun ⟨h1, h2, h3⟩ => by sorry)

end triathlon_speed_requirements_l716_716460


namespace total_animal_count_l716_716525

theorem total_animal_count (initial_hippos : ℕ) (initial_elephants : ℕ) 
  (female_hippo_factor : ℚ) (newborn_per_female_hippo : ℕ) 
  (extra_newborn_elephants : ℕ) 
  (h_initial_hippos : initial_hippos = 35)
  (h_initial_elephants : initial_elephants = 20)
  (h_female_hippo_factor : female_hippo_factor = 5 / 7)
  (h_newborn_per_female_hippo : newborn_per_female_hippo = 5)
  (h_extra_newborn_elephants : extra_newborn_elephants = 10) : 
  (initial_elephants + initial_hippos + 
  (initial_hippos * female_hippo_factor).to_nat * newborn_per_female_hippo + 
  (initial_hippos * female_hippo_factor).to_nat * newborn_per_female_hippo + 
  extra_newborn_elephants) = 315 :=
by
  sorry

end total_animal_count_l716_716525


namespace percentage_increase_to_348_l716_716366

noncomputable def SharonCurrentSalary (new_salary : ℕ) (increase_rate : ℝ) : ℕ :=
  new_salary / (1 + increase_rate)

theorem percentage_increase_to_348 
  (S : ℕ)
  (new_salary_348 : ℕ := 348)
  (new_salary_330 : ℕ := 330)
  (ten_percent_increase : ℝ := 0.10)
  (S := SharonCurrentSalary new_salary_330 ten_percent_increase) :
  ∃ (P : ℕ), S + ((P : ℝ) / 100) * S = new_salary_348 ∧ P = 16 :=
by
  sorry

end percentage_increase_to_348_l716_716366


namespace regular_polygon_sides_l716_716180

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716180


namespace kite_angle_in_circle_l716_716526

theorem kite_angle_in_circle (n : ℕ) (h1 : n = 10) (h2 : ∑ (i : ℕ) in (list.range n).map (λ x, 36), i = 360) (x : ℕ) :
  2 * x = 360 / n → x = 162 := 
by 
  sorry

end kite_angle_in_circle_l716_716526


namespace theta_possibilities_l716_716309

noncomputable def theta_values (θ : ℝ) : Prop :=
  ∀ x₁ ∈ set.Icc 0 (π / 2), ∃ x₂ ∈ set.Icc (-π / 2) 0, 2 * Real.sin x₁ = 2 * Real.cos (x₂ + θ) + 1

theorem theta_possibilities :
  ∀ θ : ℝ, theta_values θ →
    θ = 5 * π / 6 ∨ θ = 2 * π / 3 :=
by
  sorry

end theta_possibilities_l716_716309


namespace regular_polygon_sides_l716_716158

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716158


namespace smallest_n_satisfies_l716_716672

noncomputable def smallest_n : ℕ :=
  778556334111889667445223

theorem smallest_n_satisfies (N : ℕ) : 
  (N > 0 ∧ ∃ k : ℕ, ∀ m:ℕ, N * 999 = (7 * ((10^k - 1) / 9) )) → N = smallest_n :=
begin
  sorry
end 

end smallest_n_satisfies_l716_716672


namespace find_m_if_polynomial_is_square_l716_716365

theorem find_m_if_polynomial_is_square (m : ℝ) :
  (∀ x, ∃ k : ℝ, x^2 + 2 * (m - 3) * x + 16 = (x + k)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_if_polynomial_is_square_l716_716365


namespace general_formula_l716_716343

noncomputable def a : ℕ → ℚ
| 0     := 1 -- Not used, since sequence starts from n=1
| 1     := 3 / 2
| (n+1) := n * (a n - 1) / (4 * n * (n + 1) * a n - 4 * n^2 - 3 * n + 1) + 1

theorem general_formula (n : ℕ) (h : n > 0) : 
  a n = (1 : ℚ) / (2 * n * (2 * n - 1)) + 1 :=
by {
  -- Proof steps go here
  sorry
}

end general_formula_l716_716343


namespace mean_and_variance_unchanged_with_replacement_l716_716411

def S : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

structure Replacement :=
  (a b c : ℤ)
  (is_in_S : a ∈ S)
  (b_plus_c_eq_a : b + c = a)
  (b_sq_plus_c_sq_minus_a_sq_eq_ten : b^2 + c^2 - a^2 = 10)

def replacements : List Replacement :=
  [ { a := 4, b := -1, c := 5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num },
    { a := -4, b := 1, c := -5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num } ]

theorem mean_and_variance_unchanged_with_replacement (r : Replacement) :
  -- assuming r comes from the list of replacements
  ∃ (r ∈ replacements), r = r :=
begin
  sorry
end

end mean_and_variance_unchanged_with_replacement_l716_716411


namespace smallest_m_for_R_m_l_eq_l_l716_716817

def l1_angle := Real.pi / 30
def l2_angle := Real.pi / 20
def l_angle := Real.arctan (1 / 11)
def beta_minus_alpha := l2_angle - l1_angle
def angle_per_iteration := 2 * beta_minus_alpha

theorem smallest_m_for_R_m_l_eq_l :
  ∃ (m : ℕ), 0 < m ∧ m * angle_per_iteration = 2 * Real.pi ∧ m = 60 :=
by
  sorry

end smallest_m_for_R_m_l_eq_l_l716_716817


namespace range_of_a_l716_716310

-- Define the propositions
def p (x : ℝ) := (x - 1) * (x - 2) > 0
def q (a x : ℝ) := x^2 + (a - 1) * x - a > 0

-- Define the solution sets
def A := {x : ℝ | p x}
def B (a : ℝ) := {x : ℝ | q a x}

-- State the proof problem
theorem range_of_a (a : ℝ) : 
  (∀ x, p x → q a x) ∧ (∃ x, ¬p x ∧ q a x) → -2 < a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l716_716310


namespace regular_polygon_sides_l716_716079

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716079


namespace word_count_l716_716503

namespace MaginariaWords

theorem word_count :
  let alphabet_size := 25
  let valid_words :=
    let without_b (n : Nat) := (alphabet_size - 1) ^ n
    let all_words (n : Nat) := alphabet_size ^ n
    List.sum (List.map 
      (λ n => all_words n - without_b n) 
      [1, 2, 3, 4, 5])
  in 
  valid_words = 1860701 := sorry

end MaginariaWords

end word_count_l716_716503


namespace problem1_problem2_l716_716297

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Define the conditions and questions as Lean statements

-- First problem: Prove that if A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem problem1 (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := 
  sorry

-- Second problem: Prove that if A a ⊆ B, then a ∈ (-∞, 0] ∪ [4, ∞)
theorem problem2 (a : ℝ) (h1 : A a ⊆ B) : a ≤ 0 ∨ a ≥ 4 := 
  sorry

end problem1_problem2_l716_716297


namespace car_clock_problem_l716_716489

-- Define the conditions and statements required for the proof
variable (t₀ : ℕ) -- Initial time in minutes corresponding to 2:00 PM
variable (t₁ : ℕ) -- Time in minutes when the accurate watch shows 2:40 PM
variable (t₂ : ℕ) -- Time in minutes when the car clock shows 2:50 PM
variable (t₃ : ℕ) -- Time in minutes when the car clock shows 8:00 PM
variable (rate : ℚ) -- Rate of the car clock relative to real time

-- Define the initial condition
def initial_time := (t₀ = 0)

-- Define the time gain from 2:00 PM to 2:40 PM on the accurate watch
def accurate_watch_time := (t₁ = 40)

-- Define the time gain for car clock from 2:00 PM to 2:50 PM
def car_clock_time := (t₂ = 50)

-- Define the rate of the car clock relative to real time as 5/4
def car_clock_rate := (rate = 5/4)

-- Define the car clock reading at 8:00 PM
def car_clock_later := (t₃ = 8 * 60)

-- Define the actual time corresponding to the car clock reading 8:00 PM
def actual_time : ℚ := (t₀ + (t₃ - t₀) * (4/5))

-- Define the statement theorem using the defined conditions and variables
theorem car_clock_problem 
  (h₀ : initial_time t₀) 
  (h₁ : accurate_watch_time t₁) 
  (h₂ : car_clock_time t₂) 
  (h₃ : car_clock_rate rate) 
  (h₄ : car_clock_later t₃) 
  : actual_time t₀ t₃ = 8 * 60 + 24 :=
by sorry

end car_clock_problem_l716_716489


namespace a_is_multiple_of_2_l716_716029

theorem a_is_multiple_of_2 (a : ℕ) (h1 : 0 < a) (h2 : (4 ^ a) % 10 = 6) : a % 2 = 0 :=
sorry

end a_is_multiple_of_2_l716_716029


namespace cot_difference_l716_716771

theorem cot_difference (P Q R S T : Point)
  (h1 : median P S Q R)
  (h2 : angle P S T = 30)
  (h3 : midpoint S Q R)
  (h4 : altitude P T Q R) :
  |cot(angle Q P R) - cot(angle R P Q)| = 2 :=
sorry

end cot_difference_l716_716771


namespace arcade_fraction_spent_l716_716944

noncomputable def weekly_allowance : ℚ := 2.25 
def y (x : ℚ) : ℚ := 1 - x
def remainding_after_toy (x : ℚ) : ℚ := y x - (1/3) * y x

theorem arcade_fraction_spent : 
  ∃ x : ℚ, remainding_after_toy x = 0.60 ∧ x = 3/5 :=
by
  sorry

end arcade_fraction_spent_l716_716944


namespace triangle_area_l716_716806

variables {a d S : ℝ}
variables (A B C M : Type*)

-- Let ABC be an equilateral triangle with side length a.
-- Let M be a point at distance d from the centroid of triangle ABC.

def equilateral_triangle (a : ℝ) (A B C : Type*) : Prop :=
-- Definitions for the properties of an equilateral triangle
sorry

def distance_from_centroid (d : ℝ) (M : Type*) (A B C : Type*) : Prop :=
-- Definition for the distance from point M to the centroid of triangle ABC
sorry
  
theorem triangle_area (h1 : equilateral_triangle a A B C) (h2 : distance_from_centroid d M A B C) :
    S = (sqrt 3 / 12) * (a^2 - 3 * d^2) :=
by
  sorry

end triangle_area_l716_716806


namespace prove_f_neg_a_l716_716455

def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 11) : f (-a) = -9 := 
  by sorry

end prove_f_neg_a_l716_716455


namespace area_of_circle_R_is_25_percent_of_circle_S_l716_716741

theorem area_of_circle_R_is_25_percent_of_circle_S (D_S : ℝ) (hD_R : D_R = 0.5 * D_S) : 
  let A_S := π * (D_S / 2)^2 in
  let A_R := π * (D_R / 2)^2 in
  (A_R / A_S) * 100 = 25 := by 
sorry

end area_of_circle_R_is_25_percent_of_circle_S_l716_716741


namespace total_dogs_equation_l716_716256

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end total_dogs_equation_l716_716256


namespace negation_of_proposition_l716_716856

theorem negation_of_proposition (a b : ℝ) (h : a > b → a^2 > b^2) : a ≤ b → a^2 ≤ b^2 :=
by
  sorry

end negation_of_proposition_l716_716856


namespace problem_statement_l716_716476

theorem problem_statement 
  (x y : ℝ) 
  (h : x + y = 1) 
  (m n : ℕ) :
  x^(m+1) * ∑ j in Finset.range (n+1), Nat.choose (m+j) j * y^j +
  y^(n+1) * ∑ i in Finset.range (m+1), Nat.choose (n+i) i * x^i = 1 :=
sorry

end problem_statement_l716_716476


namespace eq_roots_iff_m_values_l716_716649

theorem eq_roots_iff_m_values (m : ℝ) :
  (∃ x : ℝ, 3 * x^2 - (m+2) * x + 15 = 0) ∧ ∀ x₁ x₂ : ℝ, 3 * x₁^2 - (m+2) * x₁ + 15 = 0 → 3 * x₂^2 - (m+2) * x₂ + 15 = 0 → x₁ = x₂ →
  m = 6 * real.sqrt 5 - 2 ∨ m = -6 * real.sqrt 5 - 2 :=
sorry

end eq_roots_iff_m_values_l716_716649


namespace total_age_difference_is_12_l716_716519

variables (A B C D : ℕ)

-- Conditions
axiom C_is_12_years_younger_than_A : C = A - 12
axiom age_difference_equation : A + B = B + C + D

-- Proof problem
theorem total_age_difference_is_12 : D = 12 :=
by
  sorry

end total_age_difference_is_12_l716_716519


namespace hiker_catches_up_in_time_l716_716598

-- Define the problem conditions as constants
constant hiker_rate : ℝ := 8  -- hiker's rate in miles per hour
constant cyclist_rate : ℝ := 28  -- cyclist's rate in miles per hour
constant time_passed : ℝ := 7 / 60  -- time passed in hours, which is 7 minutes

-- Define the proof problem
theorem hiker_catches_up_in_time :
  let distance_cyclist := cyclist_rate * time_passed in
  let distance_hiker := hiker_rate * time_passed in
  let distance_to_cover := distance_cyclist - distance_hiker in
  let time_hours := distance_to_cover / hiker_rate in
  let time_minutes := time_hours * 60 in
  time_minutes = 17.5 :=
by
  -- Placeholder for proof
  sorry

end hiker_catches_up_in_time_l716_716598


namespace regular_polygon_sides_l716_716154

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716154


namespace regular_polygon_sides_l716_716106

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716106


namespace change_in_expression_l716_716643

-- Define the original mathematical function y
def y (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define b as a positive constant
variable (b : ℝ)
variable (hb : b > 0)

-- Define the changes in the expression when x is replaced by x + b and x - b
def Δy_add (x : ℝ) : ℝ := y (x + b) - y x
def Δy_sub (x : ℝ) : ℝ := y (x - b) - y x

-- State the theorem in Lean 4
theorem change_in_expression (x : ℝ) (hb : b > 0) : 
  (Δy_add x = 3*x^2*b + 3*x*b^2 + b^3 - 2*b) ∧ 
  (Δy_sub x = -3*x^2*b + 3*x*b^2 - b^3 + 2*b) := by
sorry

end change_in_expression_l716_716643


namespace triangle_area_correct_l716_716769

def is_angle_60_degrees (angle : ℝ) : Prop :=
  angle = 60

def is_altitude (altitude : ℝ) : Prop :=
  altitude = 2

def triangle_area_proof_problem (angle : ℝ) (altitude : ℝ) : ℝ :=
  if is_angle_60_degrees angle ∧ is_altitude altitude then
    (2 * Real.sqrt 3) / 3
  else
    0

theorem triangle_area_correct :
  ∀ (angle : ℝ) (altitude : ℝ), 
    is_angle_60_degrees angle ∧ is_altitude altitude →
    triangle_area_proof_problem angle altitude = (2 * Real.sqrt 3) / 3 :=
begin
  sorry
end

end triangle_area_correct_l716_716769


namespace cos_diff_angle_l716_716729

variables (x : ℝ)

def vec_a := (Real.cos x, Real.sin x)
def vec_b := (Real.sqrt 2, Real.sqrt 2)
def dot_product := vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2

theorem cos_diff_angle :
  dot_product = (8 : ℝ) / 5 → Real.cos (x - Real.pi / 4) = 4 / 5 :=
by
  intros h
  sorry

end cos_diff_angle_l716_716729


namespace count_integers_with_consecutive_ones_l716_716352

def has_two_consecutive_ones (n : ℕ) : Prop :=
  ∃ i : ℕ, i < 10 ∧ n / 10^(10-i) % 100 = 11

theorem count_integers_with_consecutive_ones :
  { x : ℕ // 10^10 ≤ x ∧ x < 2 * 10^10 ∧ (∀ d, (x / 10^d) % 10 = 1 ∨ (x / 10^d) % 10 = 2) ∧ has_two_consecutive_ones x }.card =
  1815 := 
sorry

end count_integers_with_consecutive_ones_l716_716352


namespace smallest_n_with_10_divisors_l716_716862

def has_exactly_10_divisors (n : ℕ) : Prop :=
  let divisors : ℕ → ℕ := λ n, (n.divisors).card;
  n.divisors.count = 10

theorem smallest_n_with_10_divisors : ∃ n : ℕ, has_exactly_10_divisors n ∧ ∀ m : ℕ, has_exactly_10_divisors m → n ≤ m :=
begin
  use 48,
  split,
  { 
    -- proof that 48 has exactly 10 divisors
    sorry 
  },
  {
    -- proof that 48 is the smallest such number
    sorry
  }

end smallest_n_with_10_divisors_l716_716862


namespace relatively_prime_positive_integers_l716_716440

theorem relatively_prime_positive_integers (a b : ℕ) (h1 : a > b) (h2 : gcd a b = 1) (h3 : (a^3 - b^3) / (a - b)^3 = 91 / 7) : a - b = 1 := 
by 
  sorry

end relatively_prime_positive_integers_l716_716440


namespace integer_solutions_l716_716736

theorem integer_solutions (x : ℤ) : 
  (∃ (integer_values : Finset ℤ), integer_values = { -1, 0, 1, 2 } ∧ 
    ∀ val ∈ integer_values, (∃ k : ℤ, (6 * val + 3) = k * (2 * val - 1))) ∧ integer_values.card = 4 := 
sorry

end integer_solutions_l716_716736


namespace negation_of_p_l716_716738

variable {x : ℝ}

def p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_p_l716_716738


namespace square_diagonals_perpendicular_l716_716510

-- Definitions related to square and rectangle properties

def is_square (A : Type) [DecidableRel (@eq A)] (s : set A) : Prop :=
  ∃ (a b c d : A), s = {a, b, c, d} ∧ 
  congruent a b ∧ congruent b c ∧ congruent c d ∧ congruent d a ∧ 
  right_angle a b c ∧ right_angle b c d ∧ right_angle c d a ∧ right_angle d a b

def is_rectangle (A : Type) [DecidableRel (@eq A)] (r : set A) : Prop :=
  ∃ (a b c d : A), r = {a, b, c, d} ∧ 
  parallel a b c d ∧ parallel b c d a ∧ congruent a b ∧ congruent b c ∧ congruent c d ∧ congruent d a ∧
  right_angle a b c ∧ right_angle b c d ∧ right_angle c d a ∧ right_angle d a b

-- Theorem to be proven: Square has property that diagonals are perpendicular
theorem square_diagonals_perpendicular {A : Type} [DecidableRel (@eq A)] 
  (s r : set A) (Square : is_square A s) (Rectangle : is_rectangle A r) :
  Diagonal_perpendicular_to_each_other s := sorry

end square_diagonals_perpendicular_l716_716510


namespace cello_viola_pairs_l716_716919

theorem cello_viola_pairs (cellos violas : Nat) (p_same_tree : ℚ) (P : Nat)
  (h_cellos : cellos = 800)
  (h_violas : violas = 600)
  (h_p_same_tree : p_same_tree = 0.00020833333333333335)
  (h_equation : P * ((1 : ℚ) / cellos * (1 : ℚ) / violas) = p_same_tree) :
  P = 100 := 
by
  sorry

end cello_viola_pairs_l716_716919


namespace parallelogram_l716_716474

noncomputable def midpoint {α : Type*} [AddCommGroup α] [Module ℝ α] (a b : α) : α :=
(a + b) / 2

theorem parallelogram {α : Type*} [AddCommGroup α] [Module ℝ α]
  {A B C D P Q R S M : α}
  (hP : midpoint A B = P)
  (hQ : midpoint B C = Q)
  (hR : midpoint C D = R)
  (hS : midpoint D A = S)
  (hAPMS : ∃ u v : ℝ, u • (P - A) + v • (S - A) = M - A) :
  ∃ u v : ℝ, u • (R - C) + v • (Q - C) = M - C :=
sorry

end parallelogram_l716_716474


namespace regular_polygon_sides_l716_716126

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716126


namespace total_coins_l716_716257

-- Defining the conditions
def stack1 : Nat := 4
def stack2 : Nat := 8

-- Statement of the proof problem
theorem total_coins : stack1 + stack2 = 12 :=
by
  sorry

end total_coins_l716_716257


namespace general_formula_for_sequence_a_sum_of_sequence_b_l716_716516

-- Definition of the sequence S_n
def S_n (n : ℕ) : ℚ := (n^2 + 3 * n) / 2

-- (I) General formula for the sequence {a_n}
def a_n (n : ℕ) : ℕ := n + 1

theorem general_formula_for_sequence_a (n : ℕ) (hn : n ≥ 1) :
  a_n n = (S_n n - S_n (n - 1)).toNat :=
by
  sorry

-- Definition of the sequence b_n
def b_n (n : ℕ) : ℚ := 1 / (a_n (2 * n - 1) * a_n (2 * n + 1))

-- Definition of T_n, the sum of the first n terms of sequence {b_n}
def T_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b_n (i + 1)

-- (II) Sum of the first n terms of the sequence {b_n}
theorem sum_of_sequence_b (n : ℕ) :
  T_n n = n / (4 * n + 4) :=
by
  sorry

end general_formula_for_sequence_a_sum_of_sequence_b_l716_716516


namespace regular_polygon_sides_l716_716211

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716211


namespace exists_even_function_b_l716_716714

-- Define the function f(x) = 2x^2 - b*x
def f (b x : ℝ) : ℝ := 2 * x^2 - b * x

-- Define the condition for f being an even function: f(-x) = f(x)
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the existence of a b in ℝ such that f is an even function
theorem exists_even_function_b :
  ∃ b : ℝ, is_even_function (f b) :=
by
  sorry

end exists_even_function_b_l716_716714


namespace mean_and_variance_unchanged_with_replacement_l716_716412

def S : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

structure Replacement :=
  (a b c : ℤ)
  (is_in_S : a ∈ S)
  (b_plus_c_eq_a : b + c = a)
  (b_sq_plus_c_sq_minus_a_sq_eq_ten : b^2 + c^2 - a^2 = 10)

def replacements : List Replacement :=
  [ { a := 4, b := -1, c := 5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num },
    { a := -4, b := 1, c := -5, is_in_S := sorry, b_plus_c_eq_a := by norm_num, b_sq_plus_c_sq_minus_a_sq_eq_ten := by norm_num } ]

theorem mean_and_variance_unchanged_with_replacement (r : Replacement) :
  -- assuming r comes from the list of replacements
  ∃ (r ∈ replacements), r = r :=
begin
  sorry
end

end mean_and_variance_unchanged_with_replacement_l716_716412


namespace regular_polygon_sides_l716_716170

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716170


namespace regular_polygon_sides_l716_716198

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716198


namespace total_students_l716_716533

-- Define two-thirds of the class having brown eyes
def brown_eyes (T : ℤ) : Prop :=
  (2 / 3 : ℤ) * T = 2 * T / 3

-- Define half of the students with brown eyes have black hair
def black_hair_given_brown_eyes (T : ℤ) : Prop :=
  (1 / 2 : ℤ) * (2 * T / 3) = (1 / 3 : ℤ) * T

-- Given 6 students have brown eyes and black hair
def students_with_brown_eyes_black_hair : Prop :=
  (1 / 3 : ℤ) * ?m_1 = 6

-- The theorem to prove
theorem total_students (T : ℤ) (H1 : brown_eyes T) (H2 : black_hair_given_brown_eyes T) (H3 : students_with_brown_eyes_black_hair T) : T = 18 := by
  sorry

end total_students_l716_716533


namespace coeff_x2_expansion_l716_716846

/-- The coefficient of x^2 in the expansion of (2 - x + x^2) * (1 + 2 * x) ^ 6 is 109. -/
theorem coeff_x2_expansion : 
  let f : Polynomial ℚ := Polynomial.C 2 - Polynomial.X + Polynomial.X^2
  let g : Polynomial ℚ := (Polynomial.C 1 + Polynomial.C 2 * Polynomial.X) ^ 6
  f * g.coeff 2 = 109 := 
by
  sorry

end coeff_x2_expansion_l716_716846


namespace ratio_side_length_to_perimeter_l716_716940

theorem ratio_side_length_to_perimeter (s : ℝ) (hs : s = 15) : (s : ℝ) / (4 * s) = 1 / 4 :=
by simp [hs]; field_simp; norm_num

-- Placeholder to ensure the code compiles without providing the actual proof.
example : ratio_side_length_to_perimeter 15 rfl := by sorry

end ratio_side_length_to_perimeter_l716_716940


namespace min_n_guess_correct_l716_716055

theorem min_n_guess_correct (p : ℚ) (bound : ℚ) : 
  (∀ n : ℕ, (p = 2/3) → (bound = 0.05) → (p^n < bound) → 8 ≤ n) :=
begin
  intros n p_eq bound_eq p_n_lt_bound,
  sorry
end

end min_n_guess_correct_l716_716055


namespace triangle_area_l716_716017

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end triangle_area_l716_716017


namespace area_isosceles_right_triangle_l716_716502

open Real

-- Define the condition that the hypotenuse of an isosceles right triangle is 4√2 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = (4 * sqrt 2)^2

-- State the theorem to prove the area of the triangle is 8 square units
theorem area_isosceles_right_triangle (a b : ℝ) (h : hypotenuse a b) : 
  a = b → 1/2 * a * b = 8 := 
by 
  intros
  -- Proof steps are not required, so we use 'sorry'
  sorry

end area_isosceles_right_triangle_l716_716502


namespace count_valid_n_l716_716666

theorem count_valid_n :
  ∃ (count : ℕ), count = 9 ∧ 
  (∀ (n : ℕ), 0 < n ∧ n ≤ 2000 ∧ ∃ (k : ℕ), 21 * n = k * k ↔ count = 9) :=
by
  sorry

end count_valid_n_l716_716666


namespace square_value_zero_l716_716734

variable {a b : ℝ}

theorem square_value_zero (h1 : a > b) (h2 : -2 * a - 1 < -2 * b + 0) : 0 = 0 := 
by
  sorry

end square_value_zero_l716_716734


namespace tan_arccot_sin_arccot_l716_716641

noncomputable def angle : ℝ := Real.arccot (3/5)

theorem tan_arccot (h : ∀ θ, θ = angle → Real.cot θ = 3 / 5) :
  Real.tan angle = 5 / 3 :=
by
  have θ := angle
  sorry

theorem sin_arccot (h : ∀ θ, θ = angle → Real.cot θ = 3 / 5) :
  Real.sin angle = (5 * Real.sqrt 34) / 34 :=
by
  have θ := angle
  sorry

end tan_arccot_sin_arccot_l716_716641


namespace rival_awards_eq_24_l716_716784

-- Definitions
def Scott_awards : ℕ := 4
def Jessie_awards (scott: ℕ) : ℕ := 3 * scott
def Rival_awards (jessie: ℕ) : ℕ := 2 * jessie

-- Theorem to prove
theorem rival_awards_eq_24 : Rival_awards (Jessie_awards Scott_awards) = 24 := by
  sorry

end rival_awards_eq_24_l716_716784


namespace rani_time_difference_is_correct_l716_716027

def rani_time_difference : ℝ :=
  let old_time_per_mile : ℝ := (165 : ℝ) / 20
  let new_time_per_mile : ℝ := (180 : ℝ) / 12
  new_time_per_mile - old_time_per_mile

theorem rani_time_difference_is_correct : rani_time_difference = 6.75 :=
by
  sorry

end rani_time_difference_is_correct_l716_716027


namespace fencing_cost_l716_716505

def total_cost_of_fencing 
  (length breadth cost_per_meter : ℝ)
  (h1 : length = 62)
  (h2 : length = breadth + 24)
  (h3 : cost_per_meter = 26.50) : ℝ :=
  2 * (length + breadth) * cost_per_meter

theorem fencing_cost : total_cost_of_fencing 62 38 26.50 (by rfl) (by norm_num) (by norm_num) = 5300 := 
by 
  sorry

end fencing_cost_l716_716505


namespace valid_times_96_l716_716847

open Set

def valid_digits : Set Nat := {1, 2, 3, 4, 5, 6}

def is_valid_time (hours minutes seconds : Nat) : Prop :=
  let digits := (show hours, minutes, seconds).toString.toList.map (λ c => c.toNat - '0'.toNat)
  digits.toSet ⊆ valid_digits ∧ digits.countp (λ d => d ∈ valid_digits) = 6

def count_valid_times : Nat :=
  (List.fin_range 24).sum (λ h =>
    (List.fin_range 60).sum (λ m =>
      (List.fin_range 60).count (λ s => is_valid_time h m s)
    )
  )

theorem valid_times_96 : count_valid_times = 96 := by
  sorry

end valid_times_96_l716_716847


namespace parabola_distance_l716_716721

open Real

-- Define the parabola and its focus and directrix
noncomputable def parabola (x y : ℝ) := x^2 = 4 * y

def focus : ℝ × ℝ := (0, 1)

def directrix (y : ℝ) := y = -1

-- Define point A on the directrix
def A (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define point B as the intersection of line AF and the parabola
def B (m n : ℝ) := parabola m n

-- Define vectors FA and FB
def vector_FA (a : ℝ) : ℝ × ℝ :=
  let F := focus
  let A := A a
  (A.1 - F.1, A.2 - F.2)

def vector_FB (m n : ℝ) : ℝ × ℝ :=
  let F := focus
  (m - F.1, n - F.2)

-- Given vector equation FA = -4 * FB
def condition (a m n : ℝ) : Prop :=
  vector_FA a = (-4) • vector_FB m n

-- We need to prove |BF| = 5/2
theorem parabola_distance (a m n : ℝ) (hB : B m n) (h_condition : condition a m n) :
  abs ((sqrt (m^2 + (n - 1)^2)) = 5/2) :=
sorry

end parabola_distance_l716_716721


namespace businessmen_neither_coffee_nor_tea_l716_716628

/-- Definitions of conditions -/
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 6

/-- Statement of the problem -/
theorem businessmen_neither_coffee_nor_tea : 
  (total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end businessmen_neither_coffee_nor_tea_l716_716628


namespace geom_seq_sum_six_div_a4_minus_one_l716_716801

theorem geom_seq_sum_six_div_a4_minus_one (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = a 1 * r^n) 
  (h2 : a 1 = 1) 
  (h3 : a 2 * a 6 - 6 * a 4 - 16 = 0) :
  S 6 / (a 4 - 1) = 9 :=
sorry

end geom_seq_sum_six_div_a4_minus_one_l716_716801


namespace first_player_wins_game_l716_716574

theorem first_player_wins_game :
  ∀ (coins : ℕ), coins = 2019 →
  (∀ (n : ℕ), n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99) →
  (∀ (m : ℕ), m % 2 = 0 ∧ 2 ≤ m ∧ m ≤ 100) →
  ∃ (f : ℕ → ℕ → ℕ), (∀ (c : ℕ), c <= coins → c = 0) :=
by
  sorry

end first_player_wins_game_l716_716574


namespace remainder_when_b_divided_by_13_eq_8_l716_716446

theorem remainder_when_b_divided_by_13_eq_8 :
  let b := ((2⁻¹ : ℤ/13) + (5⁻¹ : ℤ/13) + (9⁻¹ : ℤ/13))⁻¹ in
  b = 8 := by
sorry

end remainder_when_b_divided_by_13_eq_8_l716_716446


namespace polynomial_divides_l716_716436

theorem polynomial_divides (P : Polynomial ℝ) : (P - Polynomial.X) ∣ (P.eval P - Polynomial.X) :=
sorry

end polynomial_divides_l716_716436


namespace twenty_five_question_test_l716_716823

def not_possible_score (score total_questions correct_points unanswered_points incorrect_points : ℕ) : Prop :=
  ∀ correct unanswered incorrect : ℕ,
    correct + unanswered + incorrect = total_questions →
    correct * correct_points + unanswered * unanswered_points + incorrect * incorrect_points ≠ score

theorem twenty_five_question_test :
  not_possible_score 96 25 4 2 0 :=
by
  sorry

end twenty_five_question_test_l716_716823


namespace james_stop_duration_l716_716425

theorem james_stop_duration : 
  ∀ (speed distance total_time stop_time : ℕ), 
  distance = 360 → 
  speed = 60 → 
  total_time = 7 → 
  stop_time = total_time - (distance / speed) → 
  stop_time = 1 :=
by
  intros speed distance total_time stop_time h_distance h_speed h_total_time h_stop_time
  rw [h_distance, h_speed] at h_stop_time
  simp at h_stop_time
  exact h_stop_time

end james_stop_duration_l716_716425


namespace centroids_collinear_l716_716447

-- Define the conditions and statements of the problem
variables {α β : Type*}
variables (e : ℕ → α → β → Prop) (f : α → β → Prop)
variables (n : ℕ)

-- f is not parallel to any of the lines e_i 
axiom f_not_parallel (i : ℕ) (h : i < n) : ¬ ∃ m : ℝ, ∀ x y : α, f x y ↔ e i x y ∧ y = m * x

-- Define the centroid Sα of intersection points
noncomputable def S_α (α : α) : α × β :=
(let intersections := finset.range n in
 let points := intersections.image (λ i, classical.some (exists_intersection (e i) f (classical.some_spec (exists_exclusion (f_not_parallel i (intersections.mem_range.2 i.is_lt)))) α)) in
 have h : points ≠ ∅ := finset.nonempty.image _ (finset.nonempty_range),
 classical.some (exists_centroid points h))

-- The theorem to be proven
theorem centroids_collinear :
  ∃ g : α → β → Prop, ∀ α : α, ∃ m : ℝ, ∀ x y : α, g x y ↔ S_α x y ∧ y = m * x :=
sorry

end centroids_collinear_l716_716447


namespace regular_polygon_sides_l716_716082

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716082


namespace right_triangle_area_l716_716014

def hypotenuse := 17
def leg1 := 15
def leg2 := 8
def area := (1 / 2:ℝ) * leg1 * leg2 

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hl1 : a = 15) (hl2 : c = 17) :
  area = 60 := by 
  sorry

end right_triangle_area_l716_716014


namespace survey_B_count_l716_716887

-- Define the first number and common difference of the arithmetic sequence
def first_number : ℕ := 8
def common_difference : ℕ := 30

-- Define the range boundaries for survey groups
def survey_B_lower_bound : ℕ := 161
def survey_B_upper_bound : ℕ := 320

-- Define the sequence of numbers selected
def selected_numbers : ℕ → ℕ
| 0       := first_number
| (n + 1) := selected_numbers n + common_difference

-- Define the count of numbers in the sequence that fall into the range [161, 320]
def in_survey_B_range (n : ℕ) : bool :=
  survey_B_lower_bound ≤ selected_numbers n ∧ selected_numbers n ≤ survey_B_upper_bound

-- Prove that the number of selected individuals in the range [161, 320] is 5
theorem survey_B_count : (Finset.range 16).filter (λ n => in_survey_B_range n).card = 5 := 
  by {
    sorry -- Proof not required
  }

end survey_B_count_l716_716887


namespace Royals_wins_50_l716_716857

def num_wins := {45, 35, 40, 50, 60}
def Sharks_win (w: ℕ) := w ∈ num_wins
def Raptors_win (w: ℕ) := w ∈ num_wins
def Royals_win (w: ℕ) := w ∈ num_wins
def Dragons_win (w: ℕ) := w ∈ num_wins
def Knights_win (w: ℕ) := w ∈ num_wins

axiom conditions : ∃ (Sharks_win Raptors_win Royals_win Dragons_win Knights_win : ℕ),
  Sharks_win < Raptors_win ∧ 
  Dragons_win > 30 ∧ 
  Dragons_win < Royals_win ∧ Royals_win < Knights_win ∧ 
  Sharks_win ∈ num_wins ∧ Raptors_win ∈ num_wins ∧
  Royals_win ∈ num_wins ∧ Dragons_win ∈ num_wins ∧ Knights_win ∈ num_wins

theorem Royals_wins_50 : ∃ n, n = 50 ∧ ∃ (Sharks_win Raptors_win Dragons_win Knights_win : ℕ),
  conditions ∧ n ∈ num_wins :=
sorry

end Royals_wins_50_l716_716857


namespace valid_hexadecimal_2015_l716_716551

def is_hexadecimal_digit (d : Char) : Prop :=
  d ∈ ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']

def is_valid_hexadecimal (s : String) : Prop :=
  s.fold true (λ acc c, acc ∧ is_hexadecimal_digit c)

theorem valid_hexadecimal_2015 : is_valid_hexadecimal "2015" :=
  sorry

end valid_hexadecimal_2015_l716_716551


namespace regular_polygon_sides_l716_716159

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716159


namespace regular_polygon_sides_l716_716213

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716213


namespace stone_beads_crystal_beads_ratio_l716_716820

-- Conditions as definitions
def beads_per_bracelet : ℕ := 8
def bracelets_made : ℕ := 20
def nancy_metal_beads : ℕ := 40
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20
def rose_crystal_beads : ℕ := 20

-- The problem statement
theorem stone_beads_crystal_beads_ratio :
  let total_beads_needed := bracelets_made * beads_per_bracelet
  let nancy_total_beads := nancy_metal_beads + nancy_pearl_beads
  let rose_total_beads := total_beads_needed - nancy_total_beads
  let rose_stone_beads := rose_total_beads - rose_crystal_beads in
  rose_stone_beads / rose_crystal_beads = 2 :=
by
  -- Proof is omitted here
  sorry

end stone_beads_crystal_beads_ratio_l716_716820


namespace count_letter_arrangements_l716_716353

/-- The number of 18-letter arrangements made of 6 A's, 6 B's, and 6 C's 
such that no A's are in the first 6 letters, no B's in the next 6 letters,
and no C's in the last 6 letters is equal to ∑ k in range (6 + 1), (choose 6 k)^3 -/
theorem count_letter_arrangements : 
  ∑ k in Finset.range (6 + 1), (Nat.choose 6 k) ^ 3 = ∑ k in Finset.range (6 + 1), (Nat.choose 6 k) ^ 3 := 
sorry

end count_letter_arrangements_l716_716353


namespace triangle_type_is_isosceles_l716_716713

theorem triangle_type_is_isosceles {A B C : ℝ}
  (h1 : A + B + C = π)
  (h2 : ∀ x : ℝ, x^2 - x * (Real.cos A * Real.cos B) + 2 * Real.sin (C / 2)^2 = 0)
  (h3 : ∃ x1 x2 : ℝ, x1 + x2 = Real.cos A * Real.cos B ∧ x1 * x2 = 2 * Real.sin (C / 2)^2 ∧ (x1 + x2 = (x1 * x2) / 2)) :
  A = B ∨ B = C ∨ C = A := 
sorry

end triangle_type_is_isosceles_l716_716713


namespace divide_square_into_equal_octagons_l716_716776

-- Let n be the total number of octagons
variable (n : ℕ)

-- Define that the area of the square is 64 square units
def area_of_square : ℕ := 64

-- Define the valid sizes for octagons
def valid_sizes := {4, 8, 16, 32}

-- Define a predicate to check if the total area is divisible by a valid size
def is_valid_partition (size : ℕ) : Prop :=
  size ∈ valid_sizes ∧ (area_of_square % size = 0)

-- State the theorem for the problem
theorem divide_square_into_equal_octagons :
  ∃ n, ∃ size ∈ valid_sizes, is_valid_partition size → n = area_of_square / size := by
  sorry

end divide_square_into_equal_octagons_l716_716776


namespace regular_polygon_sides_l716_716188

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716188


namespace tangent_distance_identity_l716_716538

-- Definitions based on given conditions
variable {O A M : Point}
variable (d : ℝ) (D : ℝ)
variable [Circle O d]
variable [Tangent (AX : Line) A O]
variable [OnCircle M O]
variable [DistFromLine M AX D]

-- Proof statement
theorem tangent_distance_identity : (MA : ℝ)^2 = d * D := 
 sorry

end tangent_distance_identity_l716_716538


namespace replace_one_number_preserves_mean_and_variance_l716_716409

section
open Set

def original_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def new_set (a b c : ℤ) : Set ℤ := 
  if a ∈ original_set then ((original_set.erase a) ∪ {b, c})
  else original_set

def mean (s : Set ℤ) : ℚ := (s.sum id : ℚ) / s.size

def sum_of_squares (s : Set ℤ) : ℚ := ↑(s.sum (λ x, x^2))

theorem replace_one_number_preserves_mean_and_variance :
  ∃ a b c : ℤ, a ∈ original_set ∧ 
    (mean (new_set a b c) = mean original_set) ∧ 
    (sum_of_squares (new_set a b c) = sum_of_squares original_set + 10) :=
  sorry

end

end replace_one_number_preserves_mean_and_variance_l716_716409


namespace nine_point_circle_exists_l716_716477

variables {ℝ : Type*} [field ℝ]

-- Define a Triangle with three points A, B, C
structure Triangle (ℝ : Type*) :=
(A B C : ℝ × ℝ)

-- Define midpoint of a side
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define foot of altitude from a vertex perpendicular to a side
def foot_of_altitude (A B C : ℝ × ℝ) : ℝ × ℝ := 
let (dx, dy) := (B.1 - C.1, B.2 - C.2) -- direction vector BC
in let (c, d) := (dx * A.1 + dy * A.2, dx * B.2 - dy * B.1) 
in ((c * dx + d * dy) / (dx * dx + dy * dy))

-- The main theorem statement
theorem nine_point_circle_exists (ABC : Triangle ℝ) :
  let A1 := midpoint ABC.B ABC.C,
      B1 := midpoint ABC.A ABC.C,
      C1 := midpoint ABC.A ABC.B,
      A2 := foot_of_altitude ABC.A ABC.B ABC.C,
      B2 := foot_of_altitude ABC.B ABC.A ABC.C,
      C2 := foot_of_altitude ABC.C ABC.A ABC.B
  in ∃ (O : ℝ × ℝ) (r : ℝ), (dist O A1 = r) ∧ (dist O B1 = r) ∧ (dist O C1 = r) ∧ 
      (dist O A2 = r) ∧ (dist O B2 = r) ∧ (dist O C2 = r) :=
sorry -- Proof omitted

end nine_point_circle_exists_l716_716477


namespace regular_polygon_sides_l716_716081

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716081


namespace sequence_sum_zero_l716_716972

theorem sequence_sum_zero : 
  (∑ i in Finset.range 502, (-(3 * (i * 4 + 1)) + (i * 4 + 2) + (i * 4 + 3) - (i * 4 + 4)) = 0) := 
by 
  sorry

end sequence_sum_zero_l716_716972


namespace recurring_decimal_mul_seven_l716_716245

-- Declare the repeating decimal as a definition
def recurring_decimal_0_3 : ℚ := 1 / 3

-- Theorem stating that the product of 0.333... and 7 is 7/3
theorem recurring_decimal_mul_seven : recurring_decimal_0_3 * 7 = 7 / 3 :=
by
  -- Insert proof here
  sorry

end recurring_decimal_mul_seven_l716_716245


namespace integer_root_b_l716_716316

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end integer_root_b_l716_716316


namespace mean_and_variance_unchanged_l716_716396

noncomputable def initial_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
noncomputable def replaced_set_1 : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 5, -1}
noncomputable def replaced_set_2 : Set ℤ := {-5, -3, -2, -1, 0, 1, 2, 3, 4, -5, 1}

noncomputable def mean (s : Set ℤ) : ℚ :=
  (∑ x in s.to_finset, x) / s.to_finset.card

noncomputable def variance (s : Set ℤ) : ℚ :=
  let μ := mean s
  (∑ x in s.to_finset, (x : ℚ) ^ 2) / s.to_finset.card - μ ^ 2

theorem mean_and_variance_unchanged :
  mean initial_set = 0 ∧ variance initial_set = 10 ∧
  (mean replaced_set_1 = 0 ∧ variance replaced_set_1 = 10 ∨
   mean replaced_set_2 = 0 ∧ variance replaced_set_2 = 10) := by
  sorry

end mean_and_variance_unchanged_l716_716396


namespace triangle_inequality_l716_716478

theorem triangle_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (A B C : Type) (triangle : Triangle.shape A B C), a + b > c :=
begin
  sorry
end

end triangle_inequality_l716_716478


namespace find_a_equiv_l716_716311

noncomputable def A (a : ℝ) : Set ℝ := {1, 3, a^2}
noncomputable def B (a : ℝ) : Set ℝ := {1, 2 + a}

theorem find_a_equiv (a : ℝ) (h : A a ∪ B a = A a) : a = 2 :=
by
  sorry

end find_a_equiv_l716_716311


namespace probability_of_selecting_shirt_short_sock_l716_716360

/-
  Define the problem setup:
  6 shirts, 3 pairs of shorts, 8 pairs of socks.
  Total articles of clothing: 17
-/

def total_articles := 6 + 3 + 8  -- total number of articles

def choose (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.fact n / (Nat.fact k * Nat.fact (n - k))) else 0

def total_ways_to_choose_four : ℕ :=
  choose total_articles 4

/-
  Define the favorable outcomes:
  1. At least one shirt.
  2. Exactly one pair of shorts.
  3. Exactly one pair of socks.
-/
def favorable_outcomes : ℕ :=
  let shirts := 6
  let shorts := 3
  let socks := 8
  choose shorts 1 * choose socks 1 * (choose shirts 2 + choose shirts 1)

def expected_probability : ℚ :=
  favorable_outcomes / total_ways_to_choose_four

theorem probability_of_selecting_shirt_short_sock :
  expected_probability = 84 / 397 := by
  sorry

end probability_of_selecting_shirt_short_sock_l716_716360


namespace sum_S3_largest_l716_716308

noncomputable def arithmetic_sequence (n : ℕ) : ℕ → ℝ := λ n, 7 - 2 * n

theorem sum_S3_largest
  (a : ℕ → ℝ)
  (h1 : a 1 = 5)
  (h3 : a 3 = 1) :
  ∀ n, n ≥ 3 → (∑ i in finset.range n, a i) ≤ (∑ i in finset.range 3, a i) :=
begin
  sorry
end

end sum_S3_largest_l716_716308


namespace exists_five_tuple_of_four_digit_numbers_l716_716997

theorem exists_five_tuple_of_four_digit_numbers :
  ∃ (a b c d e : ℕ),
  (1000 ≤ a ∧ a < 10000) ∧
  (1000 ≤ b ∧ b < 10000) ∧
  (1000 ≤ c ∧ c < 10000) ∧
  (1000 ≤ d ∧ d < 10000) ∧
  (1000 ≤ e ∧ e < 10000) ∧
  (a / 1000 = b / 1000) ∧
  (a / 1000 = c / 1000) ∧
  (a / 1000 = d / 1000) ∧
  (a / 1000 = e / 1000) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  ((a + b + c + d + e) % a = 0 ∨
   (a + b + c + d + e) % b = 0 ∨
   (a + b + c + d + e) % c = 0 ∨
   (a + b + c + d + e) % d = 0 ∨
   (a + b + c + d + e) % e = 0) ∧
  ((a, b, c, d, e) = (1020, 1190, 1428, 1717, 1785) ∨
   (a, b, c, d, e) = (1050, 1225, 1470, 1755, 1830) ∨
   (a, b, c, d, e) = (1080, 1260, 1512, 1794, 1875) ∨
   (a, b, c, d, e) = (1190, 1020, 1428, 1717, 1785) ∨
   -- Other permutations of the tuples
   sorry).

end exists_five_tuple_of_four_digit_numbers_l716_716997


namespace Cid_charges_5_for_car_wash_l716_716637

theorem Cid_charges_5_for_car_wash (x : ℝ) :
  5 * 20 + 10 * 30 + 15 * x = 475 → x = 5 :=
by
  intro h
  sorry

end Cid_charges_5_for_car_wash_l716_716637


namespace number_of_subsets_l716_716345

-- Defining the sets M and N
def M (a : ℤ) : Set ℤ := {a^2, 0}
def N (a : ℤ) : Set ℤ := {1, a, 2}

-- Given condition: M ∩ N = {1}
axiom M_inter_N_eq_one (a : ℤ) : M a ∩ N a = {1}

-- To prove: the number of subsets of M ∪ N is 16
theorem number_of_subsets (a : ℤ) (h : a = -1) : 
  Set.card (Set.powerset (M a ∪ N a)) = 16 := by sorry

end number_of_subsets_l716_716345


namespace people_not_buying_coffee_l716_716471

theorem people_not_buying_coffee (total_people : ℕ) (fraction_buying_coffee : ℚ)
  (h_total_people : total_people = 25)
  (h_fraction_buying_coffee : fraction_buying_coffee = 3/5) :
  (total_people - (fraction_buying_coffee * total_people).toNat) = 10 := by
  sorry

end people_not_buying_coffee_l716_716471


namespace rectangle_square_area_percentage_correct_l716_716607

noncomputable def rectangle_square_area_percentage : ℚ :=
  let s := 1 in
  let width := 3 * s in
  let length := (3 / 2) * width in
  let area_square := s^2 in
  let area_rectangle := length * width in
  (area_square / area_rectangle) * 100

theorem rectangle_square_area_percentage_correct :
  rectangle_square_area_percentage = 741 / 100 :=
by
  -- Proof to be provided here
  sorry

end rectangle_square_area_percentage_correct_l716_716607


namespace regular_polygon_sides_l716_716164

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716164


namespace low_paths_count_l716_716926

-- Definitions per problem statement
def is_low_path (path : List (ℕ × ℕ)) (n : ℕ) : Prop :=
  path.head = (0, 0) ∧ 
  path.last = (n, n) ∧ 
  (∀ p ∈ path, p.2 ≤ p.1) ∧
  ∀ i < path.length - 1, path[i + 1] = (path[i].1 + 1, path[i].2) ∨ path[i + 1] = (path[i].1, path[i].2 + 1)

def number_of_low_paths (n k : ℕ) : ℕ :=
  -- Define the function to count the number of such paths
  sorry

def binomial (n k : ℕ) : ℕ :=
  -- Function to compute binomial coefficient
  if h : k ≤ n then Nat.choose n k else 0

-- The statement of what we need to prove
theorem low_paths_count (n k : ℕ) (hkn : 1 ≤ k ∧ k ≤ n) :
  number_of_low_paths n k = (1 / k) * binomial (n-1) (k-1) * binomial n (k-1) := 
sorry

end low_paths_count_l716_716926


namespace line_MN_passes_through_midpoint_P_tangent_PQ_equals_PA_l716_716845

-- Define the circles and points in the theorem statement
variables {S S₁ : Type} [Circle S] [Circle S₁]
variables (A B M N P Q : Point)
variables (hS : Circle S)
variables (hS₁ : Circle S₁)
variables (hAB : Chord S A B)
variables (hM : TangentPoint S₁ M hAB)
variables (hN : ArcPoint S N)
variables (hP : MidpointPoint S P)

-- Prove that the line MN passes through the midpoint P of the other arc
theorem line_MN_passes_through_midpoint_P :
  Line M N ∈ Line P := 
sorry

-- Prove that the length of the tangent PQ to the circle S₁ is equal to PA
theorem tangent_PQ_equals_PA : 
  TangentLength P Q S₁ = SegmentLength P A := 
sorry

end line_MN_passes_through_midpoint_P_tangent_PQ_equals_PA_l716_716845


namespace largest_k_l716_716281

-- Definitions for the problem conditions
def xi_sum_zero (x : Fin 11 → ℝ) : Prop :=
  (∑ i, x i) = 0

def x6_is_median (x : Fin 11 → ℝ) : Prop :=
  let sorted_x := List.sorted x in
  sorted_x.get 5 = x 5

-- The theorem statement
theorem largest_k (x : Fin 11 → ℝ) 
  (h_sum : xi_sum_zero x) 
  (h_median : x6_is_median x) :
  (∑ i, x i^2) ≥ (66 / 5) * x 5^2 :=
by
  sorry

end largest_k_l716_716281


namespace regular_polygon_sides_l716_716070

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716070


namespace length_of_the_train_l716_716949

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def time_to_cross_seconds : ℝ := 30
noncomputable def bridge_length_meters : ℝ := 205

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def distance_crossed_meters : ℝ := train_speed_mps * time_to_cross_seconds

theorem length_of_the_train 
  (h1 : train_speed_kmph = 45)
  (h2 : time_to_cross_seconds = 30)
  (h3 : bridge_length_meters = 205) : 
  distance_crossed_meters - bridge_length_meters = 170 := 
by
  sorry

end length_of_the_train_l716_716949


namespace rival_awards_eq_24_l716_716782

-- Definitions
def Scott_awards : ℕ := 4
def Jessie_awards (scott: ℕ) : ℕ := 3 * scott
def Rival_awards (jessie: ℕ) : ℕ := 2 * jessie

-- Theorem to prove
theorem rival_awards_eq_24 : Rival_awards (Jessie_awards Scott_awards) = 24 := by
  sorry

end rival_awards_eq_24_l716_716782


namespace cylinder_radius_l716_716933

theorem cylinder_radius
  (r : ℝ)
  (h1 : ∃ d, 2 * d = r)
  (c_diameter : 14)
  (c_altitude : 16)
  (axes_coincide : true) :
  r = 28 / 11 := by
  sorry

end cylinder_radius_l716_716933


namespace find_g_eq_minus_x_l716_716449

-- Define the function g and the given conditions.
def g (x : ℝ) : ℝ := sorry

axiom g0 : g 0 = 2
axiom g_xy : ∀ (x y : ℝ), g (x * y) = g ((x^2 + 2 * y^2) / 3) + 3 * (x - y)^2

-- State the problem: proving that g(x) = -x.
theorem find_g_eq_minus_x : ∀ (x : ℝ), g x = -x := by
  sorry

end find_g_eq_minus_x_l716_716449


namespace regular_polygon_sides_l716_716103

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716103


namespace regular_polygon_sides_l716_716194

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716194


namespace pizza_slices_with_all_three_toppings_l716_716575

theorem pizza_slices_with_all_three_toppings : 
  ∀ (a b c d e f g : ℕ), 
  a + b + c + d + e + f + g = 24 ∧ 
  a + d + e + g = 12 ∧ 
  b + d + f + g = 15 ∧ 
  c + e + f + g = 10 → 
  g = 5 := 
by {
  sorry
}

end pizza_slices_with_all_three_toppings_l716_716575


namespace regular_polygon_sides_l716_716221

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716221


namespace probability_of_sum_5_with_three_dice_l716_716902

def fair_six_sided_die : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }

theorem probability_of_sum_5_with_three_dice :
  (∃ (dice_combinations : finset (finset (ℕ × ℕ × ℕ))),
    (∀ x ∈ dice_combinations, ∃ (a b c : fair_six_sided_die), x = (a.val, b.val, c.val) ∧ a.val + b.val + c.val = 5) ∧
    (dice_combinations.card = 6) ∧
    dice_combinations.card / 216 = 1 / 36) :=
by
  sorry

end probability_of_sum_5_with_three_dice_l716_716902


namespace range_of_a_l716_716260

def tensor_op (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, tensor_op (x - a) (x + a) < 1) ↔ (-1/2 < a ∧ a < 3/2) :=
begin
  sorry
end

end range_of_a_l716_716260


namespace solve_inequality_l716_716674

theorem solve_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end solve_inequality_l716_716674


namespace mean_and_variance_unchanged_l716_716397

noncomputable def initial_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
noncomputable def replaced_set_1 : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 5, -1}
noncomputable def replaced_set_2 : Set ℤ := {-5, -3, -2, -1, 0, 1, 2, 3, 4, -5, 1}

noncomputable def mean (s : Set ℤ) : ℚ :=
  (∑ x in s.to_finset, x) / s.to_finset.card

noncomputable def variance (s : Set ℤ) : ℚ :=
  let μ := mean s
  (∑ x in s.to_finset, (x : ℚ) ^ 2) / s.to_finset.card - μ ^ 2

theorem mean_and_variance_unchanged :
  mean initial_set = 0 ∧ variance initial_set = 10 ∧
  (mean replaced_set_1 = 0 ∧ variance replaced_set_1 = 10 ∨
   mean replaced_set_2 = 0 ∧ variance replaced_set_2 = 10) := by
  sorry

end mean_and_variance_unchanged_l716_716397


namespace truck_loads_of_dirt_l716_716932

theorem truck_loads_of_dirt (sand cement total dirt : ℝ)
  (h_sand : sand = 0.17)
  (h_cement : cement = 0.17)
  (h_total : total = 0.67)
  (h_dirt : dirt = total - (sand + cement)) :
  dirt = 0.33 :=
by
  rw [h_sand, h_cement, h_total, h_dirt]
  norm_num
  sorry

end truck_loads_of_dirt_l716_716932


namespace inequality_abc_equality_abc_l716_716829

theorem inequality_abc (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  sqrt ((a + c) * (b + d)) ≥ sqrt (a * b) + sqrt (c * d) :=
by sorry

theorem equality_abc (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  sqrt ((a + c) * (b + d)) = sqrt (a * b) + sqrt (c * d) ↔ (a * d = b * c) :=
by sorry

end inequality_abc_equality_abc_l716_716829


namespace determine_m_l716_716726

-- Definition of complex numbers z1 and z2
def z1 (m : ℝ) : ℂ := m + 2 * Complex.I
def z2 : ℂ := 2 + Complex.I

-- Condition that the product z1 * z2 is a pure imaginary number
def pure_imaginary (c : ℂ) : Prop := c.re = 0 

-- The proof statement
theorem determine_m (m : ℝ) : pure_imaginary (z1 m * z2) → m = 1 := 
sorry

end determine_m_l716_716726


namespace problem_253_base2_l716_716981

def decimal_to_binary (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec f (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc.reverse
      else f (n / 2) ((n % 2) :: acc)
    in f n []

def count_digits (lst : list ℕ) : ℕ × ℕ :=
  lst.foldr (λ d (acc : ℕ × ℕ), if d = 0 then (acc.1 + 1, acc.2) else (acc.1, acc.2 + 1)) (0, 0)

theorem problem_253_base2 :
  let bits := decimal_to_binary 253
  let (x, y) := count_digits bits
  y - x = 6 :=
by
  sorry

end problem_253_base2_l716_716981


namespace sn_eq_2an_minus_1_l716_716709

noncomputable def a_sequence (n : ℕ) : ℕ := sorry
noncomputable def S (n : ℕ) : ℕ := sorry

axiom condition_1 (n : ℕ) : 0 < a_sequence n
axiom condition_2 (n : ℕ) : log (a_sequence (n + 1)) = 1 / 2 * (log (a_sequence n) + log (a_sequence (n + 2)))
axiom condition_3 : a_sequence 3 = 4
axiom condition_4 : S 2 = 3

theorem sn_eq_2an_minus_1 (n : ℕ) : S n = 2 * a_sequence n - 1 := sorry

end sn_eq_2an_minus_1_l716_716709


namespace triangles_with_positive_area_l716_716358

def total_points := 36
def binom (n k : ℕ) : ℕ := Nat.choose n k

def total_triangles := binom total_points 3

def collinear_triangles : ℕ :=
  let rows_cols := 6 * binom 6 3
  let main_diagonals := 2 * binom 6 3
  let sub_diagonals_5 := 4 * binom 5 3
  let sub_diagonals_4 := 6 * binom 4 3
  rows_cols + main_diagonals + sub_diagonals_5 + sub_diagonals_4

def positive_area_triangles := total_triangles - collinear_triangles

theorem triangles_with_positive_area : positive_area_triangles = 6796 := by
  unfold positive_area_triangles total_triangles collinear_triangles total_points
  unfold binom
  simp
  rfl

end triangles_with_positive_area_l716_716358


namespace regular_polygon_sides_l716_716210

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716210


namespace initial_students_l716_716529

def students_got_off : ℕ := 3
def students_left : ℕ := 7

theorem initial_students (h1 : students_got_off = 3) (h2 : students_left = 7) :
    students_got_off + students_left = 10 :=
by
  sorry

end initial_students_l716_716529


namespace regular_polygon_sides_l716_716148

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716148


namespace min_value_reciprocals_l716_716802

open Real

theorem min_value_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + b = 1) :
  (1 / a + 1 / b) = 4 :=
by
  sorry

end min_value_reciprocals_l716_716802


namespace find_all_functions_l716_716275

noncomputable def is_solution (f : ℕ → ℕ) : Prop :=
  f(1) = 1 ∧
  (∀ n, f(n + 2) + (n^2 + 4*n + 3) * f(n) = (2*n + 5) * f(n + 1)) ∧
  (∀ m n, m > n → f(n) ∣ f(m))

def factorial_solution (f : ℕ → ℕ) := 
  ∀ n, f(n) = n! ∨ f(n) = (n+2)! / 6

theorem find_all_functions (f : ℕ → ℕ) (n : ℕ) :
  is_solution f → factorial_solution f := by
  sorry

end find_all_functions_l716_716275


namespace smallest_period_axis_of_symmetry_range_of_x_l716_716716

def f (x : ℝ) : ℝ :=
  (Real.cos x) * (Real.sin (x + π / 3)) - (sqrt 3) * (Real.cos x) ^ 2 + (sqrt 3) / 4

theorem smallest_period (x : ℝ) (k : ℤ) : 
  is_periodic f π :=
sorry

theorem axis_of_symmetry (x : ℝ) (k : ℤ) : 
  f (x) = f (5 * π / 12 + k * π / 2) :=
sorry
  
theorem range_of_x (x : ℝ) (k : ℤ) : 
  (f x ≥ 1 / 4) ↔ (π / 4 + k * π ≤ x ∧ x ≤ 7 * π / 12 + k * π) :=
sorry

end smallest_period_axis_of_symmetry_range_of_x_l716_716716


namespace sum_of_fractions_l716_716976

theorem sum_of_fractions :
  (∑ k in Finset.range 10, (k + 1) / 7) = 55 / 7 :=
by
  sorry

end sum_of_fractions_l716_716976


namespace twenty_percent_greater_l716_716900

theorem twenty_percent_greater (x : ℝ) (h : x = 52 + 0.2 * 52) : x = 62.4 :=
by {
  sorry
}

end twenty_percent_greater_l716_716900


namespace regular_polygon_sides_l716_716171

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716171


namespace range_f2_intersection_points_l716_716340

noncomputable theory

/-- Define the function f along with the conditions. -/
def f (a b c x : ℝ) : ℝ := -x^3 + a * x^2 + b * x + c

variables (a b c : ℝ)

axiom condition1 : ∀ x, x ∈ (-∞, 0) → f a b c x is strictly decreasing
axiom condition2 : ∀ x, x ∈ (0, 1) → f a b c x is strictly increasing
axiom condition3 : ∃ x1 x2 x3, f a b c x1 = 0 ∧ f a b c x2 = 0 ∧ f a b c x3 = 0
axiom condition4 : f a b c 1 = 0

/-- Prove the range of f(2) is (-5/2, ∞). -/
theorem range_f2 : f a b c 2 > -5 / 2 := sorry

/-- Discuss the number of intersection points between the line y = x - 1
and the curve y = f(x). -/
theorem intersection_points : 
  ∀ x, 
    (a > (3/2) ∧ a < (2 : ℝ)^(1/2) - 1) → 1 ∨
    (a = (2 : ℝ)^(1/2) - 1) → 2 ∨
    (a > 2 : ℝ)^(1/2) - 1 ∨ a = 2) → 3 := sorry

end range_f2_intersection_points_l716_716340


namespace divide_square_into_equal_octagons_l716_716775

-- Let n be the total number of octagons
variable (n : ℕ)

-- Define that the area of the square is 64 square units
def area_of_square : ℕ := 64

-- Define the valid sizes for octagons
def valid_sizes := {4, 8, 16, 32}

-- Define a predicate to check if the total area is divisible by a valid size
def is_valid_partition (size : ℕ) : Prop :=
  size ∈ valid_sizes ∧ (area_of_square % size = 0)

-- State the theorem for the problem
theorem divide_square_into_equal_octagons :
  ∃ n, ∃ size ∈ valid_sizes, is_valid_partition size → n = area_of_square / size := by
  sorry

end divide_square_into_equal_octagons_l716_716775


namespace b_is_arithmetic_seq_a_is_strictly_increasing_l716_716701

-- Definitions of the sequences and conditions
variable {a : ℝ}
variable {n : ℕ} (hn : n ≥ 2)

noncomputable def a₁ := a
noncomputable def S : ℕ → ℝ
| 0       := 0
| (n + 1) := S n + a_n

axiom sum_squares (n : ℕ) : S n ^ 2 = 3 * n^2 * a_n + S (n - 1) ^ 2
axiom a_nonzero {n : ℕ} (hn : n ≥ 2) : a_n ≠ 0

def b (n : ℕ) : ℝ := a (2 * n)

-- Problem (1)
theorem b_is_arithmetic_seq (n : ℕ) : b (n + 1) - b n = 6 :=
sorry

-- Problem (2)
theorem a_is_strictly_increasing (a : ℝ) : 
  (9 / 4 < a ∧ a < 15 / 4) ↔ (∀ (n : ℕ), a_n < a_{n + 1}) :=
sorry

end b_is_arithmetic_seq_a_is_strictly_increasing_l716_716701


namespace regular_polygon_sides_l716_716223

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716223


namespace regular_polygon_sides_l716_716102

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716102


namespace number_of_subsets_l716_716680

noncomputable def a := sorry
noncomputable def b := sorry

theorem number_of_subsets (a b : ℝ) (h1 : a > b) (h2 : b > 1)
  (h3 : log a b + log b a = 10 / 3) (h4 : a ^ b = b ^ a) :
  ∃ s : finset ℝ, s.card = 3 ∧ 
  ∀ x ∈ s, x = a ∨ x = b ∨ x = 3 * b ∨ x = b^2 ∨ x = a - 2 * b := 
begin
  sorry
end

end number_of_subsets_l716_716680


namespace distance_z5_from_origin_l716_716991

noncomputable def z_seq : ℕ → ℂ
| 0       => 0
| 1       => 1
| (n + 2) => (z_seq (n + 1))^2 - 1 + complex.i

theorem distance_z5_from_origin : (complex.abs (z_seq 5) = real.sqrt 370) :=
by
  sorry

end distance_z5_from_origin_l716_716991


namespace shift_parabola_l716_716959

theorem shift_parabola (x : ℝ) : 
    let y := x^2 in 
    ∃ (x' : ℝ), (x' = x - 2) ∧ (y = x'^2) :=
    sorry

end shift_parabola_l716_716959


namespace closest_value_approx_l716_716618

/-- Given conditions for the proof -/
variables (M N : ℝ) (log3 : ℝ)
hypothesis h1 : M = 3^361
hypothesis h2 : N = 10^80
hypothesis h3 : log3 = 0.48

/-- The target statement to be proved -/
theorem closest_value_approx (h1 : M = 3^361) (h2 : N = 10^80) (h3 : log3 = 0.48) : M / N ≈ 10^93 :=
begin
  sorry
end

end closest_value_approx_l716_716618


namespace problem1_problem2_l716_716394

-- Problem 1: Prove that {b_n} is an arithmetic sequence and find the general formula for {a_n}.
theorem problem1 (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a (n+1) * a n + a (n+1) - a n = 0) 
  (h₃ : ∀ n : ℕ, b n = 1 / a n) : 
  ∀ n : ℕ, b (n+1) - b n = 1 ∧ a n = 1 / n := 
by 
  sorry

-- Problem 2: Calculate the sum of the first 8 terms of the sequence {c_n}.
theorem problem2 (a : ℕ → ℝ) (b : ℕ → ℝ)
  (c : ℕ → ℤ)
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a (n+1) * a n + a (n+1) - a n = 0) 
  (h₃ : ∀ n : ℕ, b n = 1 / a n)
  (h₄ : ∀ n : ℕ, b n = n) 
  (h₅ : ∀ n : ℕ, c n = Int.floor ((2 * n + 3) / 5)) : 
  ∑ i in Finset.range 8, c (i + 1) = 16 := 
by 
  sorry

end problem1_problem2_l716_716394


namespace regular_polygon_sides_l716_716207

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716207


namespace area_of_figure_enclosed_by_curve_l716_716393

theorem area_of_figure_enclosed_by_curve (θ : ℝ) : 
  ∃ (A : ℝ), A = 4 * Real.pi ∧ (∀ θ, (4 * Real.cos θ)^2 = (4 * Real.cos θ) * 4 * Real.cos θ) :=
sorry

end area_of_figure_enclosed_by_curve_l716_716393


namespace regular_polygon_sides_l716_716155

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716155


namespace B_pow_5_eq_r_B_add_s_I_l716_716811

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![ -2,  3 ], 
                                      ![  4,  5 ]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem B_pow_5_eq_r_B_add_s_I :
  ∃ r s : ℤ, (r = 425) ∧ (s = 780) ∧ (B^5 = r • B + s • I) :=
by
  sorry

end B_pow_5_eq_r_B_add_s_I_l716_716811


namespace program_result_l716_716833

/-- Defining the sum of sums in Lean -/
def sum_of_sums (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ k, (finset.range (k + 1)).sum (λ j, j + 1))

theorem program_result :
  sum_of_sums 100 = 1 + (1 + 2) + (1 + 2 + 3) + ⋯ + (1 + 2 + 3 + ⋯ + 100) :=
sorry

end program_result_l716_716833


namespace binary_representation_253_l716_716987

theorem binary_representation_253 (x y : ℕ) (h : nat.bits 253 = [1, 1, 1, 1, 1, 1, 0, 1]) :
  (y - x = 6) :=
begin
  have hx : x = 1, from by {
    -- x is the number of 0's in the list representation [1, 1, 1, 1, 1, 1, 0, 1]
    sorry
  },
  have hy : y = 7, from by {
    -- y is the number of 1's in the list representation [1, 1, 1, 1, 1, 1, 0, 1]
    sorry
  },
  rw [hx, hy],
  exact rfl,
end

end binary_representation_253_l716_716987


namespace courtyard_length_proof_l716_716920

noncomputable def paving_stone_area (length width : ℝ) : ℝ := length * width

noncomputable def total_area_stones (stone_area : ℝ) (num_stones : ℝ) : ℝ := stone_area * num_stones

noncomputable def courtyard_length (total_area width : ℝ) : ℝ := total_area / width

theorem courtyard_length_proof :
  let stone_length := 2.5
  let stone_width := 2
  let courtyard_width := 16.5
  let num_stones := 99
  let stone_area := paving_stone_area stone_length stone_width
  let total_area := total_area_stones stone_area num_stones
  courtyard_length total_area courtyard_width = 30 :=
by
  sorry

end courtyard_length_proof_l716_716920


namespace volume_difference_l716_716240

def height_A : ℝ := 8
def circumference_A : ℝ := 6
def radius_A : ℝ := circumference_A / (2 * Real.pi)
def volume_A : ℝ := Real.pi * radius_A^2 * height_A

def height_B : ℝ := 6
def circumference_B : ℝ := 8
def radius_B : ℝ := circumference_B / (2 * Real.pi)
def volume_B : ℝ := Real.pi * radius_B^2 * height_B

def positive_difference := abs (volume_B - volume_A)

theorem volume_difference : Real.pi * positive_difference = 24 :=
by
  sorry

end volume_difference_l716_716240


namespace car_travel_l716_716884

namespace DistanceTravel

/- Define the conditions -/
def distance_initial : ℕ := 120
def car_speed : ℕ := 80

/- Define the relationship between y and x -/
def y (x : ℝ) : ℝ := distance_initial - car_speed * x

/- Prove that y is a linear function and verify the value of y at x = 0.8 -/
theorem car_travel (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1.5) : 
  (y x = distance_initial - car_speed * x) ∧ 
  (y x = 120 - 80 * x) ∧ 
  (x = 0.8 → y x = 56) :=
sorry

end DistanceTravel

end car_travel_l716_716884


namespace chocolates_divisible_l716_716563

theorem chocolates_divisible (n m : ℕ) (h1 : n > 0) (h2 : m > 0) : 
  (n ≤ m) ∨ (m % (n - m) = 0) :=
sorry

end chocolates_divisible_l716_716563


namespace two_triangles_share_edge_l716_716293

theorem two_triangles_share_edge 
  (n : ℕ) 
  (h1 : n ≥ 2) 
  (points : Finset Point)
  (h2 : points.card = 2 * n)
  (h3 : ∀ p1 p2 p3 p4 ∈ points, ¬ coplanar p1 p2 p3 p4) 
  (segments : Finset (Point × Point))
  (h4 : segments.card = n^2 + 1) : 
  ∃ (triangle1 triangle2 : (Finset (Point × Point))), 
    triangle1.card = 3 ∧ triangle2.card = 3 ∧ 
    (∃ edge ∈ triangle1, edge ∈ triangle2) := 
sorry

end two_triangles_share_edge_l716_716293


namespace tangent_line_at_P_l716_716849

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 1

def P : ℝ × ℝ := (-1, 3)

theorem tangent_line_at_P :
    ∀ (x y : ℝ), (y = 2*x^2 + 1) →
    (x, y) = P →
    ∃ m b : ℝ, b = -1 ∧ m = -4 ∧ (y = m*x + b) :=
by
  sorry

end tangent_line_at_P_l716_716849


namespace find_x_plus_y_l716_716690

theorem find_x_plus_y (x y : ℝ) 
  (h1 : 16^x = 4^(y + 2)) 
  (h2 : 27^y = 9^(x - 5)) : 
  x + y = 1 := 
sorry

end find_x_plus_y_l716_716690


namespace sum_of_squares_of_roots_l716_716331

theorem sum_of_squares_of_roots : 
  ∀ a b c x1 x2 : ℝ, a ≠ 0 ∧ x1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) 
  ∧ x2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a) ∧ (a = 1 ∧ b = -9 ∧ c = 9) 
  → (x1^2 + x2^2 = 63) := 
by
  intros a b c x1 x2 h1
  cases h1 with h1 h2,
  cases h2 with h2 h3,
  cases h3 with h3 h4,
  cases h4 with h5 h6,
  cases h6 with h6 h7,
  cases h7 with h7 h8,
  sorry

end sum_of_squares_of_roots_l716_716331


namespace translate_sine_to_cosine_l716_716530

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x + Real.pi / 6)

theorem translate_sine_to_cosine :
    ∀ (x : ℝ), (Math.sin (2 * (x + Real.pi / 6) + Real.pi / 6)) = Math.cos (2 * x) :=
by
  intro x
  sorry

end translate_sine_to_cosine_l716_716530


namespace range_of_alpha_l716_716294

open Real

theorem range_of_alpha 
  (α : ℝ) (k : ℤ) :
  (sin α > 0) ∧ (cos α < 0) ∧ (sin α > cos α) →
  (∃ k : ℤ, (2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) ∨ 
  (2 * k * π + (3 * π / 2) < α ∧ α < 2 * k * π + 2 * π)) := 
by 
  sorry

end range_of_alpha_l716_716294


namespace find_zero_of_g_l716_716334

-- Define the problem conditions
def f (a : ℝ) (x : ℝ) : ℝ := a ^ x + a
def g (a : ℝ) (x : ℝ) : ℝ := f a x - 4

-- State the theorem we need to prove
theorem find_zero_of_g (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a 0 = 3) : g a 1 = 0 :=
by
  -- proof omitted
  sorry

end find_zero_of_g_l716_716334


namespace replace_preserve_mean_variance_l716_716404

theorem replace_preserve_mean_variance:
  ∀ (a b c : ℤ), 
    let initial_set := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map (λ x, x : ℤ) in
    let new_set := (initial_set.erase a).++[b, c] in
    let mean (s : List ℤ) := (s.sum : ℚ) / s.length in
    let variance (s : List ℤ) :=
      let m := mean s in
      (s.map (λ x, (x - m) ^ 2)).sum / s.length in
    mean initial_set = 0 ∧ variance initial_set = 10 ∧
    ((mean new_set = 0 ∧ variance new_set = 10) ↔ ((a = -4 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 5))) :=
sorry

end replace_preserve_mean_variance_l716_716404


namespace club_membership_l716_716732

variable (n : ℕ) -- number of boys (and girls initially, as they are equal)
variable me : Prop -- I am a girl

-- Conditions from the problem
axiom h1 : ∀ (b g : ℕ), b = g -- There are the same number of boys as there are girls.
axiom h2 : ∀ (b g : ℕ), (3 * (b + g - 1) / 4 = g) -> b = g -- When one boy is absent, three-fourths of the team are girls.

-- goal
theorem club_membership (b g : ℕ) (H : b = g) :
   -- Ensure gender of the individual and number of boys is 2
   me = ∀ (me : Prop), me → ∃ (b : ℕ), b = 2
   sorry

end club_membership_l716_716732


namespace zero_in_interval_l716_716292

def f (x : ℝ) : ℝ := x^(1/3) - (1/2)^x

theorem zero_in_interval : ∃ x : ℝ, x ∈ set.Ioo (1/3) (1/2) ∧ f x = 0 :=
by
  have h₀ : f 0 < 0 := by
    rw [f, zero_rpow zero_lt_two]
    norm_num
  have h₁ : f (1/3) < 0 := by
    rw [f]
    norm_num
    -- Proof needed here
    sorry
  have h₂ : f (1/2) > 0 := by
    rw [f]
    norm_num
    -- Proof needed here
    sorry
  
  -- Applying the intermediate value theorem
  exact IntermediateValueTheorem h₀ h₁ h₂

end zero_in_interval_l716_716292


namespace total_palm_trees_l716_716593

theorem total_palm_trees (forest_palm_trees : ℕ) (h_forest : forest_palm_trees = 5000) 
    (h_ratio : 3 / 5) : forest_palm_trees + (forest_palm_trees - (h_ratio * forest_palm_trees).to_nat) = 7000 :=
by
  sorry

end total_palm_trees_l716_716593


namespace more_people_needed_to_paint_fence_l716_716269

theorem more_people_needed_to_paint_fence :
  ∀ (n t m t' : ℕ), n = 8 → t = 3 → t' = 2 → (n * t = m * t') → m - n = 4 :=
by
  intros n t m t'
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end more_people_needed_to_paint_fence_l716_716269


namespace regular_polygon_sides_l716_716112

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716112


namespace complex_quadrant_l716_716740

theorem complex_quadrant (z : ℂ) (hz : (3 - 4 * complex.I + z) * complex.I = 2 + complex.I) :
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_quadrant_l716_716740


namespace length_segment_A_A_l716_716881

-- Define the coordinates of points A and A'
def A : (ℝ × ℝ) := (3, -2)
def A' : (ℝ × ℝ) := (-3, -2)

-- Define distance formula
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Problem statement
theorem length_segment_A_A' : dist A A' = 6 := by
  -- the proof will be written here
  sorry

end length_segment_A_A_l716_716881


namespace min_handshakes_35_people_l716_716907

theorem min_handshakes_35_people (n : ℕ) (h1 : n = 35) (h2 : ∀ p : ℕ, p < n → p ≥ 3) : ∃ m : ℕ, m = 51 :=
by
  sorry

end min_handshakes_35_people_l716_716907


namespace notebooks_left_l716_716006

/-!
# Problem: There are 28 notebooks. I gave 1/4 of the 28 books to Yeonju, and 3/7 of the 28 books to Minji. Find how many notebooks are left.

## Conditions
- There are 28 notebooks.
- 1/4 of the 28 books are given to Yeonju.
- 3/7 of the 28 books are given to Minji.

## Goal
- Find how many notebooks are left.
-/

theorem notebooks_left (total_notebooks : ℕ) (yeonju_fraction minji_fraction : ℚ) : 
  total_notebooks = 28 → yeonju_fraction = 1/4 → minji_fraction = 3/7 → 
  (total_notebooks - yeonju_fraction * total_notebooks - minji_fraction * total_notebooks).to_nat = 9 :=
by
  intros h1 h2 h3
  sorry

end notebooks_left_l716_716006


namespace number_of_perfect_square_multiples_21_below_2000_l716_716667

/-- Define n and the condition 21n being a perfect square --/
def is_perfect_square (k : ℕ) : Prop :=
  ∃ m : ℕ, m * m = k

def count_perfect_squares_upto (N : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | n ≤ N ∧ ∃ a : ℕ, n = k * a ∧ is_perfect_square (k * n) }

theorem number_of_perfect_square_multiples_21_below_2000 : 
  count_perfect_squares_upto 2000 21 21 = 9 :=
sorry

end number_of_perfect_square_multiples_21_below_2000_l716_716667


namespace sequence_an_l716_716437

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom S_formula (n : ℕ) (h₁ : n > 0) : S n = 2 * a n - 2

-- Proof goal
theorem sequence_an (n : ℕ) (h₁ : n > 0) : a n = 2 ^ n := by
  sorry

end sequence_an_l716_716437


namespace malcolm_brushes_teeth_l716_716459

theorem malcolm_brushes_teeth :
  (∃ (M : ℕ), M = 180 ∧ (∃ (N : ℕ), N = 90 ∧ (M / N = 2))) :=
by
  sorry

end malcolm_brushes_teeth_l716_716459


namespace solution_set_l716_716994

noncomputable def solve_inequality : Set ℝ :=
  {x | (1 / (x - 1)) >= -1}

theorem solution_set :
  solve_inequality = {x | x ≤ 0} ∪ {x | x > 1} :=
by
  sorry

end solution_set_l716_716994


namespace time_per_bone_l716_716967

theorem time_per_bone (total_hours : ℕ) (total_bones : ℕ) (h1 : total_hours = 1030) (h2 : total_bones = 206) :
  (total_hours / total_bones = 5) :=
by {
  sorry
}

end time_per_bone_l716_716967


namespace regular_polygon_sides_l716_716218

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716218


namespace regular_polygon_sides_l716_716123

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716123


namespace origin_moves_distance_3_75_l716_716597

-- Definitions and conditions
def original_circle_center : ℝ × ℝ := (3, 3)
def original_circle_radius : ℝ := 3
def dilated_circle_center : ℝ × ℝ := (8, 9)
def dilated_circle_radius : ℝ := 5
def origin : ℝ × ℝ := (0, 0)

-- The dilation factor
def dilation_factor : ℝ := dilated_circle_radius / original_circle_radius

-- Calculating the center of dilation
noncomputable def center_of_dilation : ℝ × ℝ :=
  let (x, y) := dilated_circle_center
  let (a, b) := original_circle_center
  ((27:ℝ)/(-8), (39:ℝ)/(-8))

-- Calculating the initial and final distances from the origin to the center of dilation
noncomputable def initial_distance_from_origin : ℝ :=
  let (x, y) := center_of_dilation
  real.sqrt ((x ^ 2) + (y ^ 2))

noncomputable def final_distance_from_origin : ℝ :=
  dilation_factor * initial_distance_from_origin

-- The distance the origin moves
noncomputable def distance_origin_moves : ℝ :=
  final_distance_from_origin - initial_distance_from_origin

-- Theorem to be proved
theorem origin_moves_distance_3_75 :
  distance_origin_moves = (3.75 : ℝ) :=
by
  sorry

end origin_moves_distance_3_75_l716_716597


namespace regular_polygon_sides_l716_716080

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716080


namespace arc_length_eq_one_div_sin_half_l716_716369

theorem arc_length_eq_one_div_sin_half 
  (r : ℝ) (α : ℝ)
  (h1 : α = 1)
  (h2 : r * sin (α / 2) = 1) :
  α * r = 1 / sin (1 / 2) :=
by
  sorry

end arc_length_eq_one_div_sin_half_l716_716369


namespace burger_cost_is_350_l716_716836

noncomputable def cost_of_each_burger (tip steak_cost steak_quantity ice_cream_cost ice_cream_quantity money_left: ℝ) : ℝ :=
(tip - money_left - (steak_cost * steak_quantity + ice_cream_cost * ice_cream_quantity)) / 2

theorem burger_cost_is_350 :
  cost_of_each_burger 99 24 2 2 3 38 = 3.5 :=
by
  sorry

end burger_cost_is_350_l716_716836


namespace replace_one_number_preserves_mean_and_variance_l716_716406

section
open Set

def original_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def new_set (a b c : ℤ) : Set ℤ := 
  if a ∈ original_set then ((original_set.erase a) ∪ {b, c})
  else original_set

def mean (s : Set ℤ) : ℚ := (s.sum id : ℚ) / s.size

def sum_of_squares (s : Set ℤ) : ℚ := ↑(s.sum (λ x, x^2))

theorem replace_one_number_preserves_mean_and_variance :
  ∃ a b c : ℤ, a ∈ original_set ∧ 
    (mean (new_set a b c) = mean original_set) ∧ 
    (sum_of_squares (new_set a b c) = sum_of_squares original_set + 10) :=
  sorry

end

end replace_one_number_preserves_mean_and_variance_l716_716406


namespace bernardo_wins_more_probable_l716_716244

open Nat Finset

namespace BernardoSilviaProblem

noncomputable def factorial (n : ℕ) : ℕ :=
match n with
| 0 => 1
| (nat.succ k) => (nat.succ k) * factorial k

noncomputable def combination (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

def bernardo_prob := (combination 8 2) / (combination 9 3)
#eval bernardo_prob -- Should evaluate to 1/3

def silvia_prob := 3 / 4

def total_prob := (1 / 3) + ((2 / 3) * silvia_prob)
#eval total_prob -- Should evaluate to 5/6

theorem bernardo_wins_more_probable :
  total_prob = 5 / 6 :=
by
  -- Here we state that "total_prob is equal to 5 / 6"
  sorry

end BernardoSilviaProblem

end bernardo_wins_more_probable_l716_716244


namespace pythagorean_numbers_example_l716_716239

theorem pythagorean_numbers_example (a b c : ℕ) (h : a = 6 ∧ b = 8 ∧ c = 10) :
  a^2 + b^2 = c^2 :=
by
  cases h with ha hbc
  cases hbc with hb hc
  rw [ha, hb, hc]
  norm_num
  sorry

end pythagorean_numbers_example_l716_716239


namespace dogs_not_doing_anything_l716_716872

/--
There are 264 dogs in a park. 15% of the dogs are running, 1/4 of them are playing with toys, 
1/6 of them are barking, 10% of them are digging holes, and 12 dogs are competing in an agility course. 
How many dogs are not doing anything?
-/
theorem dogs_not_doing_anything :
  let total_dogs := 264,
      running_dogs := (0.15 * total_dogs).round,
      playing_dogs := (1 / 4 * total_dogs).toNat,
      barking_dogs := (1 / 6 * total_dogs).toNat,
      digging_dogs := (0.10 * total_dogs).round,
      competing_dogs := 12,
      active_dogs := running_dogs + playing_dogs + barking_dogs + digging_dogs + competing_dogs
  in total_dogs - active_dogs = 76 := by {
    sorry
  }

end dogs_not_doing_anything_l716_716872


namespace area_of_right_triangle_with_hypotenuse_and_angle_l716_716934

theorem area_of_right_triangle_with_hypotenuse_and_angle 
  (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 9 * Real.sqrt 3) (h_angle : angle = 30) : 
  ∃ (area : ℝ), area = 364.5 := 
by
  sorry

end area_of_right_triangle_with_hypotenuse_and_angle_l716_716934


namespace logarithmic_equation_solutions_l716_716897

theorem logarithmic_equation_solutions (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1/9) (h3 : x ≠ 1/3) (h4 : x ≠ -1/3) :
  5 * log (x / 9) x + log (9 / x) (x ^ 3) + 8 * log (9 * x^2) (x^2) = 2 ↔ x = 3 ∨ x = sqrt(3) :=
by
  sorry

end logarithmic_equation_solutions_l716_716897


namespace value_of_a_plus_b_l716_716688

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 2) (h3 : a < b) : a + b = -3 := by
  -- Proof goes here
  sorry

end value_of_a_plus_b_l716_716688


namespace rival_awards_l716_716785

theorem rival_awards (S J R : ℕ) (h1 : J = 3 * S) (h2 : S = 4) (h3 : R = 2 * J) : R = 24 := 
by sorry

end rival_awards_l716_716785


namespace area_of_given_right_triangle_l716_716021

variable (a c : ℝ)
def is_right_triangle (a c : ℝ) : Prop := ∃ b : ℝ, a^2 + b^2 = c^2
def area_of_right_triangle (a : ℝ) (b : ℝ) : ℝ := (1 / 2) * a * b

theorem area_of_given_right_triangle (h : is_right_triangle 15 17) : 
  ∃ (b : ℝ), (15^2 + b^2 = 17^2) ∧ (area_of_right_triangle 15 b = 60) :=
by sorry

end area_of_given_right_triangle_l716_716021


namespace find_a_l716_716507

-- Define the quadratic equation
def quadratic_eq (x a : ℝ) : Prop := x^2 - 3 * x - a = 0

-- Given that x = -1 is a root of the equation
axiom root_condition (a : ℝ) : quadratic_eq (-1) a

-- Prove that a = 4
theorem find_a (a : ℝ) (h : root_condition a) : a = 4 :=
by
  sorry

end find_a_l716_716507


namespace traffic_flow_reversal_l716_716750

-- Defining the problem statement
theorem traffic_flow_reversal (n : ℕ) (h : n ≥ 3) :
  ∃ (f : Fin n → Fin n → Prop),
  (∀ i j, i ≠ j → (f i j ∨ f j i)) →
  (∀ i j, ∃ (path : List (Fin n)), path.head = i ∧ path.ilast = j ∧ ∀ (k : Fin path.length - 1), f (path.nth_le k
  sorry

end traffic_flow_reversal_l716_716750


namespace problem1_problem2_l716_716046

-- Problem 1 Statement
theorem problem1 : real.sqrt 4 - (real.sqrt 3 - 1) ^ 0 + 2 ^ (-1 : ℤ) = 1.5 :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x : ℝ) : (1 - 2 * x < 5) ∧ ((x - 2) / 3 ≤ 1) → -2 < x ∧ x ≤ 5 :=
by
  sorry

end problem1_problem2_l716_716046


namespace regular_polygon_sides_l716_716111

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716111


namespace polynomials_equiv_l716_716660

theorem polynomials_equiv (f g : polynomial ℝ) :
  (∀ x : ℝ, (x^2 + x + 1) * f.eval (x^2 - x + 1) = (x^2 - x + 1) * g.eval (x^2 + x + 1)) →
  (∃ k : ℝ, f = polynomial.X * polynomial.C k ∧ g = polynomial.X * polynomial.C k) :=
by
  sorry

end polynomials_equiv_l716_716660


namespace minimum_quotient_l716_716995

noncomputable def digit (n : ℕ) (i : ℕ) : ℕ := (n / 10^i) % 10

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  let k := nat.log 10 n
  finset.sum (finset.range (k + 1)) (λ i, (digit n i)^2)

theorem minimum_quotient (n : ℕ) (hn : n > 0) :
  n / (sum_of_squares_of_digits n) ≥ 1 / 9 :=
sorry

end minimum_quotient_l716_716995


namespace right_triangle_area_l716_716016

def hypotenuse := 17
def leg1 := 15
def leg2 := 8
def area := (1 / 2:ℝ) * leg1 * leg2 

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hl1 : a = 15) (hl2 : c = 17) :
  area = 60 := by 
  sorry

end right_triangle_area_l716_716016


namespace smallest_pos_int_four_odd_div_eight_even_div_l716_716543

def is_odd (n : ℕ) : Prop := n % 2 = 1

def odd_divisors_count (n : ℕ) : ℕ := (finset.filter is_odd (finset.divisors n)).card

def even_divisors_count (n : ℕ) : ℕ := (finset.filter (λ d, ¬ is_odd d) (finset.divisors n)).card

theorem smallest_pos_int_four_odd_div_eight_even_div : 
  ∃ n : ℕ, n = 60 ∧ 1 < n ∧ odd_divisors_count n = 4 ∧ even_divisors_count n = 8 :=
by
  sorry

end smallest_pos_int_four_odd_div_eight_even_div_l716_716543


namespace shift_parabola_two_units_right_l716_716954

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end shift_parabola_two_units_right_l716_716954


namespace mean_and_variance_unchanged_l716_716398

noncomputable def initial_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
noncomputable def replaced_set_1 : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 5, -1}
noncomputable def replaced_set_2 : Set ℤ := {-5, -3, -2, -1, 0, 1, 2, 3, 4, -5, 1}

noncomputable def mean (s : Set ℤ) : ℚ :=
  (∑ x in s.to_finset, x) / s.to_finset.card

noncomputable def variance (s : Set ℤ) : ℚ :=
  let μ := mean s
  (∑ x in s.to_finset, (x : ℚ) ^ 2) / s.to_finset.card - μ ^ 2

theorem mean_and_variance_unchanged :
  mean initial_set = 0 ∧ variance initial_set = 10 ∧
  (mean replaced_set_1 = 0 ∧ variance replaced_set_1 = 10 ∨
   mean replaced_set_2 = 0 ∧ variance replaced_set_2 = 10) := by
  sorry

end mean_and_variance_unchanged_l716_716398


namespace servings_in_box_l716_716586

theorem servings_in_box (total_cereal : ℕ) (serving_size : ℕ) (total_cereal_eq : total_cereal = 18) (serving_size_eq : serving_size = 2) :
  total_cereal / serving_size = 9 :=
by
  sorry

end servings_in_box_l716_716586


namespace area_of_given_right_triangle_l716_716022

variable (a c : ℝ)
def is_right_triangle (a c : ℝ) : Prop := ∃ b : ℝ, a^2 + b^2 = c^2
def area_of_right_triangle (a : ℝ) (b : ℝ) : ℝ := (1 / 2) * a * b

theorem area_of_given_right_triangle (h : is_right_triangle 15 17) : 
  ∃ (b : ℝ), (15^2 + b^2 = 17^2) ∧ (area_of_right_triangle 15 b = 60) :=
by sorry

end area_of_given_right_triangle_l716_716022


namespace regular_polygon_sides_l716_716217

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716217


namespace part1_part2_part3_l716_716816

variable {x : ℝ}

def A := {x : ℝ | x^2 + 3 * x - 4 > 0}
def B := {x : ℝ | x^2 - x - 6 < 0}
def C_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem part1 : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 3} := sorry

theorem part2 : (C_R (A ∩ B)) = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := sorry

theorem part3 : (A ∪ (C_R B)) = {x : ℝ | x ≤ -2 ∨ x > 1} := sorry

end part1_part2_part3_l716_716816


namespace final_number_is_odd_l716_716469

theorem final_number_is_odd : 
  ∃ (n : ℤ), n % 2 = 1 ∧ n ≥ 1 ∧ n < 1024 := sorry

end final_number_is_odd_l716_716469


namespace tangent_line_equation_normal_line_equation_l716_716290

noncomputable def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos t, Real.sin t)

noncomputable def t0 : ℝ := Real.pi / 3

theorem tangent_line_equation :
  let (x0, y0) := curve t0
  y - y0 = (-1/3) * (x - x0) ↔
  y = - (1/3) * x + 2 * sqrt 3 / 3 := sorry

theorem normal_line_equation :
  let (x0, y0) := curve t0
  y - y0 = 3 * (x - x0) ↔
  y = 3 * x - sqrt 3 := sorry

end tangent_line_equation_normal_line_equation_l716_716290


namespace find_a_l716_716718

noncomputable def a_value (a : ℝ) : Prop :=
  (∀ (f : ℝ → ℝ), (∀ x, f x = a * Real.log x) → (∀ x, deriv f x = a / x) → deriv (λ x, a * Real.log x) 2 = 2 → a = 4)

theorem find_a : a_value 4 :=
by
  sorry

end find_a_l716_716718


namespace interval_is_trap_l716_716560

theorem interval_is_trap {X : ℕ → ℝ} (a : ℝ) 
  (h : ∀ ε > 0, ∃ k : ℕ, ∀ n > k, |X n - a| < ε) :
  ∀ ε > 0, ∃ k : ℕ, ∀ n > k, X n ∈ set.Icc (a - ε) (a + ε) :=
by
  sorry

end interval_is_trap_l716_716560


namespace profit_is_correct_l716_716564

-- Definitions of the conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def price_per_set : ℕ := 50
def sets_sold : ℕ := 500

-- Derived calculations
def revenue (sets_sold : ℕ) (price_per_set : ℕ) : ℕ :=
  sets_sold * price_per_set

def manufacturing_costs (initial_outlay : ℕ) (cost_per_set : ℕ) (sets_sold : ℕ) : ℕ :=
  initial_outlay + (cost_per_set * sets_sold)

def profit (revenue : ℕ) (manufacturing_costs : ℕ) : ℕ :=
  revenue - manufacturing_costs

-- Theorem stating the problem
theorem profit_is_correct : 
  profit (revenue sets_sold price_per_set) (manufacturing_costs initial_outlay cost_per_set sets_sold) = 5000 :=
by
  sorry

end profit_is_correct_l716_716564


namespace polygon_diagonals_excluding_vertex_l716_716038

theorem polygon_diagonals_excluding_vertex (n : ℕ) (h : n = 15) :
  (n * (n - 3)) / 2 - (n - 3) = 78 :=
by 
  rw h
  have h1 : (15 * (15 - 3)) / 2 = 90 := by norm_num
  have h2 : 15 - 3 = 12 := by norm_num
  rw [h1, h2]
  norm_num
  exact 78

end polygon_diagonals_excluding_vertex_l716_716038


namespace regular_polygon_sides_l716_716141

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716141


namespace median_eq_AC_median_perp_bisector_eq_AC_l716_716711

-- Defining the points A, B, and C
def pointA : ℝ × ℝ := (0, 4)
def pointB : ℝ × ℝ := (-2, 6)
def pointC : ℝ × ℝ := (-8, 0)

-- Problem 1: Median to side AC
theorem median_eq_AC_median : 
  let D := ((fst pointA + fst pointC) / 2, (snd pointA + snd pointC) / 2) in
  D = (-4, 2) ∧  -- Midpoint D of AC
  let k := (snd pointB - snd D) / (fst pointB - fst D) in
  k = 2 ∧  -- Slope of median BD
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = 10 ∧ a * x + b * y + c = 0 := // Equation of the median
begin
  sorry,
end

-- Problem 2: Perpendicular bisector to side AC
theorem perp_bisector_eq_AC :
  let D := ((fst pointA + fst pointC) / 2, (snd pointA + snd pointC) / 2) in
  D = (-4, 2) ∧  -- Midpoint D of AC
  let k := (snd pointC - snd pointA) / (fst pointC - fst pointA) in
  k = 1 / 2 ∧  -- Slope of AC
  let k_perp := -1 / k in
  k_perp = -2 ∧  -- Slope of the perpendicular bisector
  ∃ a b c : ℝ, a = 2 ∧ b = 1 ∧ c = 6 ∧ a * x + b * y + c = 0 := // Equation of the perpendicular bisector
begin
  sorry,
end

end median_eq_AC_median_perp_bisector_eq_AC_l716_716711


namespace find_coefficients_l716_716277

noncomputable def f (a b c : ℚ) : Polynomial ℚ := Polynomial.X^4 - Polynomial.X^3 + a * Polynomial.X^2 + b * Polynomial.X + c
noncomputable def varphi : Polynomial ℚ := Polynomial.X^3 - 2 * Polynomial.X^2 - 5 * Polynomial.X + 6

theorem find_coefficients (a b c : ℚ) :
  (f a b c) % varphi = 0 ↔ (a = -7 ∧ b = 1 ∧ c = 6) :=
by sorry

end find_coefficients_l716_716277


namespace does_not_uniquely_determine_equilateral_l716_716896

def equilateral_triangle (a b c : ℕ) : Prop :=
a = b ∧ b = c

def right_triangle (a b c : ℕ) : Prop :=
a^2 + b^2 = c^2

def isosceles_triangle (a b c : ℕ) : Prop :=
a = b ∨ b = c ∨ a = c

def scalene_triangle (a b c : ℕ) : Prop :=
a ≠ b ∧ b ≠ c ∧ a ≠ c

def circumscribed_circle_radius (a b c r : ℕ) : Prop :=
r = a * b * c / (4 * (a * b * c))

def angle_condition (α β γ : ℕ) (t : ℕ → ℕ → ℕ → Prop) : Prop :=
∃ (a b c : ℕ), t a b c ∧ α + β + γ = 180

theorem does_not_uniquely_determine_equilateral :
  ¬ ∃ (α β : ℕ), equilateral_triangle α β β ∧ α + β = 120 :=
sorry

end does_not_uniquely_determine_equilateral_l716_716896


namespace tangents_to_circle_l716_716279

theorem tangents_to_circle :
  ∀ P : ℝ × ℝ, P = (-1, 0) ∨ P = (-1, 2) →
  ∃ l : ℝ → ℝ → Prop, (P = (-1, 0) → l = (λ x y, x = -1)) ∧
                      (P = (-1, 2) → (l = (λ x y, x = -1) ∨ l = (λ x y, 3 * x + 4 * y - 5 = 0))) :=
by sorry

end tangents_to_circle_l716_716279


namespace regular_polygon_sides_l716_716116

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716116


namespace abscissa_of_point_P_l716_716321

open Real

noncomputable def hyperbola_abscissa (x y : ℝ) : Prop :=
  (x^2 - y^2 = 4) ∧
  (x > 0) ∧
  ((x + 2 * sqrt 2) * (x - 2 * sqrt 2) = -y^2)

theorem abscissa_of_point_P :
  ∃ (x y : ℝ), hyperbola_abscissa x y ∧ x = sqrt 6 := by
  sorry

end abscissa_of_point_P_l716_716321


namespace find_number_l716_716282

theorem find_number (x : ℝ) (h : x - (3/5) * x = 50) : x = 125 := by
  sorry

end find_number_l716_716282


namespace regular_polygon_sides_l716_716110

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716110


namespace regular_polygon_sides_l716_716143

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716143


namespace james_older_brother_is_16_l716_716788

variables (John James James_older_brother : ℕ)

-- Given conditions
def current_age_john : ℕ := 39
def three_years_ago_john (caj : ℕ) : ℕ := caj - 3
def twice_as_old_condition (ja : ℕ) (james_age_in_6_years : ℕ) : Prop :=
  ja = 2 * james_age_in_6_years
def james_age_in_6_years (jc : ℕ) : ℕ := jc + 6
def james_older_brother_age (jc : ℕ) : ℕ := jc + 4

-- Theorem to be proved
theorem james_older_brother_is_16
  (H1 : current_age_john = John)
  (H2 : three_years_ago_john current_age_john = 36)
  (H3 : twice_as_old_condition 36 (james_age_in_6_years James))
  (H4 : james_older_brother_age James = James_older_brother) :
  James_older_brother = 16 := sorry

end james_older_brother_is_16_l716_716788


namespace total_students_l716_716535

theorem total_students (total_students_with_brown_eyes total_students_with_black_hair: ℕ)
    (h1: ∀ (total_students : ℕ), (2 * total_students_with_brown_eyes) = 3 * total_students)
    (h2: (2 * total_students_with_black_hair) = total_students_with_brown_eyes)
    (h3: total_students_with_black_hair = 6) : 
    ∃ total_students : ℕ, total_students = 18 :=
by
  sorry

end total_students_l716_716535


namespace possible_values_of_C_l716_716977

-- Definitions based on conditions
def sum_digits_9_1_A_4_3_B_2 (A B : ℕ) : ℕ := 9 + 1 + A + 4 + 3 + B + 2
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Conditions as hypotheses
variables (A B C : ℕ)
hypothesis h1 : divisible_by_3 (sum_digits_9_1_A_4_3_B_2 A B)
hypothesis h2 : divisible_by_5 (C)

-- Proof statement
theorem possible_values_of_C : C = 0 ∨ C = 5 :=
sorry

end possible_values_of_C_l716_716977


namespace integer_root_b_l716_716317

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end integer_root_b_l716_716317


namespace problem1_problem2_l716_716047

-- Problem 1 Statement
theorem problem1 : real.sqrt 4 - (real.sqrt 3 - 1) ^ 0 + 2 ^ (-1 : ℤ) = 1.5 :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x : ℝ) : (1 - 2 * x < 5) ∧ ((x - 2) / 3 ≤ 1) → -2 < x ∧ x ≤ 5 :=
by
  sorry

end problem1_problem2_l716_716047


namespace winning_strategy_for_player_A_l716_716472

def player_has_winning_strategy (A B : Type) :=
  ∀ (piles : list ℕ), piles = [100, 200, 300] →
  ∃ has_strategy_for_A : Prop, has_strategy_for_A

theorem winning_strategy_for_player_A : player_has_winning_strategy A B :=
by
  intros piles hpiles_eq
  use true
  sorry

end winning_strategy_for_player_A_l716_716472


namespace find_k_l716_716312

def A (a b : ℤ) : Prop := 3 * a + b - 2 = 0
def B (a b : ℤ) (k : ℤ) : Prop := k * (a^2 - a + 1) - b = 0

theorem find_k (k : ℤ) (h : ∃ a b : ℤ, A a b ∧ B a b k ∧ a > 0) : k = -1 ∨ k = 2 :=
by
  sorry

end find_k_l716_716312


namespace distinct_paths_count_l716_716065

/-- The number of distinct paths a particle can take from (0, 0) to (6, 6),
following the given rules, is 183. -/
theorem distinct_paths_count : 
  let allowed_moves := λ (a b : ℕ × ℕ), (a.1 + 1, a.2) = b ∨ (a.1, a.2 + 1) = b ∨ (a.1 + 1, a.2 + 1) = b
  let cannot_include_right_angle_turns := sorry
  let valid_diagonal_moves := sorry
  number_of_paths (0, 0) (6, 6) allowed_moves cannot_include_right_angle_turns valid_diagonal_moves = 183 := 
sorry

end distinct_paths_count_l716_716065


namespace maximum_value_of_sums_of_cubes_l716_716803

theorem maximum_value_of_sums_of_cubes 
  (a b c d e : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 9) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 27 :=
sorry

end maximum_value_of_sums_of_cubes_l716_716803


namespace remainder_of_large_power_l716_716671

theorem remainder_of_large_power :
  (2^(2^(2^2))) % 500 = 36 :=
sorry

end remainder_of_large_power_l716_716671


namespace find_x_y_z_sum_l716_716886

theorem find_x_y_z_sum () : ∃ x y z : ℕ, x + y + z = 182 ∧
  ∃ n, let n := x - y * Real.sqrt (z) in
  0 < x ∧ 0 < y ∧ 0 < z ∧
  ¬ (∃ p: ℕ, p.prime ∧ p * p ∣ z) ∧ 
  ((120 - n)^2 = 7200) ∧
  (x + y + z = 182) :=
by
  sorry

end find_x_y_z_sum_l716_716886


namespace pizza_slices_left_l716_716950

theorem pizza_slices_left (total_slices : ℕ) (angeli_slices : ℚ) (marlon_slices : ℚ) 
  (H1 : total_slices = 8) (H2 : angeli_slices = 3/2) (H3 : marlon_slices = 3/2) :
  total_slices - (angeli_slices + marlon_slices) = 5 :=
by
  sorry

end pizza_slices_left_l716_716950


namespace girls_count_l716_716512

-- Define the constants according to the conditions
def boys_on_team : ℕ := 28
def groups : ℕ := 8
def members_per_group : ℕ := 4

-- Calculate the total number of members
def total_members : ℕ := groups * members_per_group

-- Calculate the number of girls by subtracting the number of boys from the total members
def girls_on_team : ℕ := total_members - boys_on_team

-- The proof statement: prove that the number of girls on the team is 4
theorem girls_count : girls_on_team = 4 := by
  -- Skip the proof, completing the statement
  sorry

end girls_count_l716_716512


namespace regular_polygon_sides_l716_716136

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l716_716136


namespace regular_polygon_sides_l716_716151

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716151


namespace trajectory_is_parabola_l716_716391

-- Define the coordinates of the points in the cube
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

-- Define the lines BC and C1D1 based on points in the Cartesian space
def LineBC : set Point3D := {P | P.x = 1 ∧ P.y ∈ set.Icc 0 1 ∧ P.z = 0}
def LineC1D1 : set Point3D := {P | P.x = 1 ∧ P.y = 1 ∧ P.z ∈ set.Icc 0 1}

-- Definition of distance in the 3D space
def distance (P Q : Point3D) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Define a point P on the side BB1C1C
def OnSideBB1C1C (P : Point3D) : Prop :=
  P.x = 1 ∧ P.y ∈ set.Icc 0 1 ∧ P.z ∈ set.Icc 0 1

-- Define equal distance condition for the problem
def EqualDistanceToLines (P : Point3D) : Prop :=
  ∃ (P_BC P_C1D1 : Point3D),
  (P_BC ∈ LineBC ∧ P_C1D1 ∈ LineC1D1) ∧
  (distance P P_BC = distance P P_C1D1)

-- Prove the trajectory is a parabola
theorem trajectory_is_parabola :
  ∀ P : Point3D, OnSideBB1C1C P → EqualDistanceToLines P → (∃ a b : ℝ, P.y = a * P.z^2 + b) :=
by
  sorry

end trajectory_is_parabola_l716_716391


namespace triangle_perpendicular_division_l716_716870

variable (a b c : ℝ)
variable (b_gt_c : b > c)
variable (triangle : True)

theorem triangle_perpendicular_division (a b c : ℝ) (b_gt_c : b > c) :
  let CK := (1 / 2) * Real.sqrt (a^2 + b^2 - c^2)
  CK = (1 / 2) * Real.sqrt (a^2 + b^2 - c^2) :=
by
  sorry

end triangle_perpendicular_division_l716_716870


namespace sum_of_even_factors_420_l716_716247

def sum_even_factors (n : ℕ) : ℕ :=
  if n ≠ 420 then 0
  else 
    let even_factors_sum :=
      (2 + 4) * (1 + 3) * (1 + 5) * (1 + 7)
    even_factors_sum

theorem sum_of_even_factors_420 : sum_even_factors 420 = 1152 :=
by {
  -- Proof skipped
  sorry
}

end sum_of_even_factors_420_l716_716247


namespace vector_magnitude_problem_l716_716349

open Real

noncomputable def unit_vector (v : ℝ × ℝ) : Prop :=
  ∥v∥ = 1

noncomputable def angle_120_degrees (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = -1 / 2

theorem vector_magnitude_problem (a b : ℝ × ℝ)
  (ha : unit_vector a) (hb : unit_vector b) (h_angle : angle_120_degrees a b) :
  ∥(2 * a.1 - b.1, 2 * a.2 - b.2)∥ = Real.sqrt 7 :=
by
  sorry

end vector_magnitude_problem_l716_716349


namespace other_x_intercept_l716_716624

-- Definition of the two foci
def f1 : ℝ × ℝ := (0, 2)
def f2 : ℝ × ℝ := (3, 0)

-- One x-intercept is given as
def intercept1 : ℝ × ℝ := (0, 0)

-- We need to prove the other x-intercept is (15/4, 0)
theorem other_x_intercept : ∃ x : ℝ, (x, 0) = (15/4, 0) ∧
  (dist (x, 0) f1 + dist (x, 0) f2 = dist intercept1 f1 + dist intercept1 f2) :=
by
  sorry

end other_x_intercept_l716_716624


namespace solve_system_of_equations_l716_716487

/-- Solve the system of equations:
(x^2 * y^4) ^ - (Real.log x) = y ^ (Real.log (y / x^7))
y^2 - x * y - 2 * x^2 + 8 * x - 4 * y = 0
Given that (x, y) = (2, 2) or (2, 4) or ((sqrt(17) - 1) / 2, (9 - sqrt(17)) / 2)
--/
theorem solve_system_of_equations (x y : ℝ) :
  (x ≠ 0 ∧ y ≠ 0) ∧
  ( (x^2 * y^4) ^ - (Real.log x) = y ^ (Real.log (y / x^7)) ) ∧
  ( y^2 - x * y - 2 * x^2 + 8 * x - 4 * y = 0 ) →
  ( (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = (Real.sqrt 17 - 1) / 2 ∧ y = (9 - Real.sqrt 17) / 2) ) :=
by sorry

end solve_system_of_equations_l716_716487


namespace regular_polygon_sides_l716_716177

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716177


namespace salad_bar_problem_l716_716632

theorem salad_bar_problem : 
  ∃ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
  mushrooms = 3 ∧
  cherry_tomatoes = 2 * mushrooms ∧
  pickles = 4 * cherry_tomatoes ∧
  (1 / 3) * bacon_bits = 32 ∧
  (red_bacon_bits = 32) ∧
  let ratio := bacon_bits / pickles in
  ratio = 4 :=
sorry

end salad_bar_problem_l716_716632


namespace students_not_make_cut_l716_716875

theorem students_not_make_cut (girls boys called_back : ℕ) (h1 : girls = 9) (h2 : boys = 14) (h3 : called_back = 2) : 
  (girls + boys - called_back = 21) :=
by simp [h1, h2, h3]; sorry

end students_not_make_cut_l716_716875


namespace reasonable_survey_method_l716_716554

-- Definitions of the conditions
def OptionA : Prop := "Comprehensive survey for the lifespan of light bulbs"
def OptionB : Prop := "Comprehensive survey for preservatives in bagged food"
def OptionC : Prop := "Random sampling survey for quality of equipment parts of 'Tianwen-1'"
def OptionD : Prop := "Random sampling survey for air quality in Fuzhou"

-- The theorem stating that Option D is the most reasonable choice given the conditions
theorem reasonable_survey_method : OptionD :=
by
  sorry

end reasonable_survey_method_l716_716554


namespace regular_polygon_sides_l716_716195

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716195


namespace sum_of_num_and_den_of_0_45_l716_716636

theorem sum_of_num_and_den_of_0_45 : 
  let x := 0.454545... in
  let frac_x := (45 : ℕ) / (99 : ℕ) in
  let num := 5 in
  let denom := 11 in
  let sum := num + denom in
  sum = 16 :=
by sorry

end sum_of_num_and_den_of_0_45_l716_716636


namespace maintain_mean_and_variance_l716_716415

def initial_set : Finset ℤ :=
  {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset ℤ) : ℚ :=
  (s.sum id : ℚ) / s.card

def variance (s : Finset ℤ) : ℚ :=
  (s.sum (λ x, (x * x : ℚ))) / (s.card : ℚ) - (mean s)^2

theorem maintain_mean_and_variance :
  ∃ (a b c : ℤ), a ∈ initial_set ∧
                 b ∉ initial_set ∧ 
                 c ∉ initial_set ∧ 
                 mean initial_set = mean (initial_set.erase a ∪ {b, c}) ∧
                 variance initial_set = variance (initial_set.erase a ∪ {b, c})
  :=
begin
  sorry
end

end maintain_mean_and_variance_l716_716415


namespace max_catch_up_distance_l716_716621

/-- 
  Suppose Alex and Max are running a race on a road that is 5000 feet long.
  Initially, they are at the same position. Then the race progresses as follows:
  1. Alex gets ahead by 300 feet.
  2. Max then gets ahead by 170 feet (reducing Alex's lead).
  3. Afterward, Alex manages to increase his lead by 440 feet.
  Prove that the number of feet left for Max to catch up to Alex is 4430 feet.
-/
theorem max_catch_up_distance (initial := 200) (total_distance := 5000)
  (alex_ahead1 := 300) (max_ahead := 170) (alex_ahead2 := 440) :
  let final_lead := alex_ahead1 - max_ahead + alex_ahead2 in
  total_distance - final_lead = 4430 :=
by
  let final_lead := alex_ahead1 - max_ahead + alex_ahead2
  calc
  total_distance - final_lead = 5000 - final_lead : by rfl
                         ... = 5000 - (alex_ahead1 - max_ahead + alex_ahead2) : by rfl
                         ... = 5000 - (300 - 170 + 440) : by rfl
                         ... = 5000 - 570 : by rfl
                         ... = 4430 : by rfl

end max_catch_up_distance_l716_716621


namespace regular_polygon_sides_l716_716131

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716131


namespace quadrilateral_ABCD_area_l716_716832

noncomputable def quadrilateral_area (AB BC AC CD AE BD: ℝ) (E : ℝ × ℝ) : ℝ :=
  let area_ABC := (1 / 2) * AB * BC
  let area_ACD := (1 / 2) * AC * CD
  area_ABC + area_ACD

theorem quadrilateral_ABCD_area :
  ∃ (AB BC BD : ℝ) (E : ℝ × ℝ), 
  angle ABC = 90 ∧ angle ACD = 90 ∧ AC = 25 ∧ CD = 40 ∧ AE = 7 ∧ 
  quadrilateral_area AB BC 25 40 7 BD E = 656.25 :=
  sorry

end quadrilateral_ABCD_area_l716_716832


namespace number_of_cows_l716_716751

noncomputable def cows_eating_husk (total_bags : ℕ) (days : ℕ) (one_cow_bags_per_days : ℚ) : ℕ :=
  total_bags / days * one_cow_bags_per_days

theorem number_of_cows (total_bags : ℕ) (days : ℕ) (one_cow_bags_per_days : ℚ) 
  (h1 : total_bags = 40) (h2 : days = 40) (h3 : one_cow_bags_per_days = 1/40) :
  cows_eating_husk total_bags days one_cow_bags_per_days = 40 := 
by
  simp [cows_eating_husk, h1, h2, h3]
  norm_num
  sorry

end number_of_cows_l716_716751


namespace number_exceeds_16_percent_by_21_l716_716858

theorem number_exceeds_16_percent_by_21 :
  ∃ x : ℝ, x = 0.16 * x + 21 ∧ x = 25 :=
begin
  sorry
end

end number_exceeds_16_percent_by_21_l716_716858


namespace regular_polygon_sides_l716_716068

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716068


namespace regular_polygon_sides_l716_716122

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716122


namespace intersection_A_B_l716_716433

-- Definition of set A
def A : Set ℝ := { x | x ≤ 3 }

-- Definition of set B
def B : Set ℝ := {2, 3, 4, 5}

-- Proof statement
theorem intersection_A_B :
  A ∩ B = {2, 3} :=
sorry

end intersection_A_B_l716_716433


namespace points_C_characterization_l716_716301

-- Given definitions from the problem conditions
variable (A B C D F : Point)
variable (segment_AB : LineSegment A B)
variable (median_AD : Median A C D)
variable (altitude_BF : Altitude B A F)

-- Condition: median of triangle ABC from A is equal to the altitude from B.
variable (h : length median_AD = length altitude_BF)

-- The final theorem statement
theorem points_C_characterization :
  ∃ (S1 S2 : Circle), 
    ∀ (C : Point), 
      (on_circle C S1 ∨ on_circle C S2) ∧ ¬(C ∈ S1 ∧ C ∈ S2) :=
sorry

end points_C_characterization_l716_716301


namespace greatest_integer_difference_l716_716562

theorem greatest_integer_difference (x y : ℝ) 
  (hx1 : 3 < x) (hx2 : x < 6) (hy1 : 6 < y) (hy2 : y < 10) :
  ∃ n : ℕ, n = 4 ∧ (y - x).natAbs = n := 
sorry

end greatest_integer_difference_l716_716562


namespace problem1_proof_l716_716720

noncomputable def problem1 (α β : ℝ) (f : ℝ → ℝ) :=
  (f = λ x, x^2 + 4 * x * Real.sin(α) + (2 / 7) * Real.tan(α)) ∧
  (0 < α ∧ α < Real.pi / 4) ∧
  (∃ x, f x = 0 ∧ ∀ y ≠ x, f y ≠ 0) ∧
  (2 * Real.cos(β)^2 = (3 / 14) + Real.sin(β)) ∧
  (β / 2 < Real.pi / 2 ∧ β < Real.pi)

theorem problem1_proof (α β : ℝ) (f : ℝ → ℝ) :
  problem1 α β f →
  Real.sin(2 * α) = 1 / 7 ∧
  β - 2 * α = 2 * Real.pi / 3 :=
by
  sorry

end problem1_proof_l716_716720


namespace cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l716_716259

-- Definition of size-n tromino
def tromino_area (n : ℕ) := (4 * 4 * n - 1)

-- Problem (a): Can a size-5 tromino be tiled by size-1 trominos
theorem cannot_tile_size5_with_size1_trominos :
  ¬ (∃ (count : ℕ), count * 3 = tromino_area 5) :=
by sorry

-- Problem (b): Can a size-2013 tromino be tiled by size-1 trominos
theorem can_tile_size2013_with_size1_trominos :
  ∃ (count : ℕ), count * 3 = tromino_area 2013 :=
by sorry

end cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l716_716259


namespace area_of_triangle_QDA_l716_716754

theorem area_of_triangle_QDA (s : ℝ) (A B C D P Q : ℝ × ℝ)
  (h_square : (A = (0, 0)) ∧ (B = (s, 0)) ∧ (C = (s, s)) ∧ (D = (0, s)))
  (h_P_on_BC : ∃ k : ℝ, k > 0 ∧ k < 1 ∧ P = ((1 - k) • B + k • C))
  (h_ratio : ∃ k : ℝ, k = 1/4 ∧ P = (s, s * k))
  (h_midpoint_Q : Q = ((s + 0) / 2, s))
  (h_area_PCQ : 1/2 * abs (P.1 * Q.2 + Q.1 * C.2 + C.1 * P.2 - P.2 * Q.1 - Q.2 * C.1 - C.2 * P.1) = 5) : 
  let area_Triangle (X Y Z : ℝ × ℝ) := 1/2 * abs (X.1 * Y.2 + Y.1 * Z.2 + Z.1 * X.2 - X.2 * Y.1 - Y.2 * Z.1 - Z.2 * X.1) in
  area_Triangle Q D A = 20 := 
sorry

end area_of_triangle_QDA_l716_716754


namespace poly_identity_l716_716809

noncomputable def f (x : ℤ) : ℤ :=
  a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

noncomputable def g (x : ℤ) : ℤ :=
  b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0

noncomputable def h (x : ℤ) : ℤ :=
  c_2 * x^2 + c_1 * x + c_0

theorem poly_identity (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) 
                      (b_3 b_2 b_1 b_0 : ℤ) 
                      (c_2 c_1 c_0 : ℤ)
                      (H1: |a_1| ≤ 4) 
                      (H2 : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1) 
                      (H3 : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1) 
                      (H4 : f 10 = g 10 * h 10)
                      : ∀ x, f x = g x * h x := 
  sorry

end poly_identity_l716_716809


namespace measure_angle_BAC_l716_716768

-- Define the elements in the problem
def triangle (A B C : Type) := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

-- Define the lengths and angles
variables {A B C X Y : Type}

-- Define the conditions given in the problem
def conditions (AX XY YB BC : ℝ) (angleABC : ℝ) : Prop :=
  AX = XY ∧ XY = YB ∧ YB = BC ∧ angleABC = 100

-- The Lean 4 statement (proof outline is not required)
theorem measure_angle_BAC {A B C X Y : Type} (hT : triangle A B C)
  (AX XY YB BC : ℝ) (angleABC : ℝ) (hC : conditions AX XY YB BC angleABC) :
  ∃ (t : ℝ), t = 25 :=
sorry
 
end measure_angle_BAC_l716_716768


namespace triangle_DEF_perimeter_l716_716852

variables {D E F P Q R : Type}
variables [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P]
variables [MetricSpace Q] [MetricSpace R]

noncomputable def radius : ℝ := 15
noncomputable def DP : ℝ := 19
noncomputable def PE : ℝ := 25

theorem triangle_DEF_perimeter
  (D E F P Q R : Type)
  [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  (radius: ℝ) (DP: ℝ) (PE: ℝ)
  (h_radius: radius = 15)
  (h_DP: DP = 19)
  (h_PE: PE = 25)
  : let x := 83.6 in 2 * (44 + x) = 255.2 :=
by
  sorry

end triangle_DEF_perimeter_l716_716852


namespace problem_solution_l716_716304

def sequencea (a : ℕ → ℕ) := a 1 = 3 ∧ ∀ n ≥ 2, a n + a (n - 1) = 4 * n

def sequenceb (a b : ℕ → ℚ) := ∀ n, b n = a n / 2 ^ n

noncomputable def Sn (b : ℕ → ℚ) (S : ℕ → ℚ) := ∀ n, S n = (finset.range n).sum (λ k, b (k + 1))

theorem problem_solution (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ) :
  sequencea a →
  sequenceb a b →
  Sn b S →
  (∀ k, odd k → ∃ d, ∀ n, a (2*n + 1) = a 1 + n * d) ∧
  (∀ k, even k → ∃ d, ∀ n, a (2*n) = a 2 + n * d) ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = 5 - (2 * n + 5) / 2 ^ n) :=
by
  sorry

end problem_solution_l716_716304


namespace polynomial_coeff_sum_l716_716362

noncomputable def polynomial :=
  (5 * x - 4)^5

def coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :=
  a_0 + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5

theorem polynomial_coeff_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  polynomial = coefficients a_0 a_1 a_2 a_3 a_4 a_5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 25 :=
  by
    sorry

end polynomial_coeff_sum_l716_716362


namespace triangle_area_l716_716018

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end triangle_area_l716_716018


namespace shifted_parabola_l716_716955

theorem shifted_parabola (x : ℝ) : 
  (let y := x^2 in (let x := x + 2 in y)) = (x - 2)^2 := sorry

end shifted_parabola_l716_716955


namespace sum_of_five_digit_binary_numbers_l716_716438

theorem sum_of_five_digit_binary_numbers :
  let T := { x : ℕ | ∃ (n : ℕ), x = nat.binary_num n 5 ∧ x < 2^5 ∧ x >= 2^4 } in
  (Σ (x : ℕ) in T, x) = 0b11111000 :=
by
  let T := { x : ℕ | ∃ (n : ℕ), x = nat.binary_num n 5 ∧ x < 2^5 ∧ x >= 2^4 }
  have T_cardinality : T.card = 16 
  -- proof goes here
  sorry
  calc (Σ (x : ℕ) in T, x)
      = Σ (x : ℕ) in T, x -- here will be the actual decomposition of the sum
     ... = 0b11111000 -- final value

end sum_of_five_digit_binary_numbers_l716_716438


namespace range_of_a_l716_716026

theorem range_of_a {a : ℝ} (h : ∀ x ∈ set.Icc (-2 : ℝ) 1, a * x^3 - x^2 + 4 * x + 3 ≥ 0) : -6 ≤ a ∧ a ≤ -2 :=
sorry

end range_of_a_l716_716026


namespace businessmen_neither_coffee_nor_tea_l716_716630

theorem businessmen_neither_coffee_nor_tea :
  ∀ (total_count coffee tea both neither : ℕ),
    total_count = 30 →
    coffee = 15 →
    tea = 13 →
    both = 6 →
    neither = total_count - (coffee + tea - both) →
    neither = 8 := 
by
  intros total_count coffee tea both neither ht hc ht2 hb hn
  rw [ht, hc, ht2, hb] at hn
  simp at hn
  exact hn

end businessmen_neither_coffee_nor_tea_l716_716630


namespace calculate_expression_l716_716248

theorem calculate_expression :
  (Real.sqrt 81) + (Real.cbrt (-27)) + (Real.sqrt ((-2)^2)) + |(Real.sqrt 3) - 2| = 10 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l716_716248


namespace min_phi_semi_circle_l716_716473

noncomputable def minimum_phi : Real :=
  π - Real.arctan(5 * Real.sqrt 6 / 12)

theorem min_phi_semi_circle
  (z : ℂ) (x y : ℝ)
  (hz : z = x + yi)
  (hx : x^2 + y^2 = 1)
  (hy : y > 0) :
  ∃ φ : ℝ, φ = Complex.arg((z - 2) / (z + 3)) ∧ φ = minimum_phi := 
sorry

end min_phi_semi_circle_l716_716473


namespace train_relative_speed_l716_716880

-- Definitions of given conditions
def initialDistance : ℝ := 13
def speedTrainA : ℝ := 37
def speedTrainB : ℝ := 43

-- Definition of the relative speed
def relativeSpeed : ℝ := speedTrainB - speedTrainA

-- Theorem to prove the relative speed
theorem train_relative_speed
  (h1 : initialDistance = 13)
  (h2 : speedTrainA = 37)
  (h3 : speedTrainB = 43) :
  relativeSpeed = 6 := by
  -- Placeholder for the actual proof
  sorry

end train_relative_speed_l716_716880


namespace train_length_is_170_meters_l716_716946

-- Definition of the conditions
def speed_km_per_hr := 45
def time_seconds := 30
def bridge_length_meters := 205

-- Convert speed from km/hr to m/s
def speed_m_per_s : ℝ := (speed_km_per_hr : ℝ) * 1000 / 3600

-- The total distance covered by the train in 30 seconds
def total_distance_m : ℝ := speed_m_per_s * (time_seconds : ℝ)

-- The length of the train
def length_of_train_m : ℝ := total_distance_m - (bridge_length_meters : ℝ)

-- The theorem we need to prove
theorem train_length_is_170_meters : 
  length_of_train_m = 170 := by
    sorry

end train_length_is_170_meters_l716_716946


namespace regular_polygon_sides_l716_716206

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716206


namespace probability_adjacent_vertices_decagon_l716_716877

theorem probability_adjacent_vertices_decagon :
  ∀ (decagon : Finset ℕ),
    decagon.card = 10 →
    let choices := decagon.powerset.filter (λ s, s.card = 3) in
    let total_ways := choices.card in
    let successful_ways := (decagon.filter (λ i, (decagon.image (λ x, x + 1) ∩ decagon.image (λ x, x + 2)))).card in
    successful_ways / total_ways = 1 / 12 :=
by {
  intros,
  sorry
}

end probability_adjacent_vertices_decagon_l716_716877


namespace solution_set_l716_716276

theorem solution_set (x : ℝ) : (x + 1 = |x + 3| - |x - 1|) ↔ (x = 3 ∨ x = -1 ∨ x = -5) :=
by
  sorry

end solution_set_l716_716276


namespace replace_one_number_preserves_mean_and_variance_l716_716405

section
open Set

def original_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def new_set (a b c : ℤ) : Set ℤ := 
  if a ∈ original_set then ((original_set.erase a) ∪ {b, c})
  else original_set

def mean (s : Set ℤ) : ℚ := (s.sum id : ℚ) / s.size

def sum_of_squares (s : Set ℤ) : ℚ := ↑(s.sum (λ x, x^2))

theorem replace_one_number_preserves_mean_and_variance :
  ∃ a b c : ℤ, a ∈ original_set ∧ 
    (mean (new_set a b c) = mean original_set) ∧ 
    (sum_of_squares (new_set a b c) = sum_of_squares original_set + 10) :=
  sorry

end

end replace_one_number_preserves_mean_and_variance_l716_716405


namespace greatest_integer_inequality_l716_716286

def greatest_integer (x : ℝ) : ℤ := floor x

theorem greatest_integer_inequality (x y : ℝ) :
  greatest_integer x > greatest_integer y → x > y ∧ ¬(x > y → greatest_integer x > greatest_integer y) :=
by
  split
  · intro h
    have h₁ : (x : ℤ) ≥ greatest_integer x := floor_le (by exact le_of_eq rfl)
    have h₂ : greatest_integer y ≤ y := le_of_lt (floor_lt y)
    exact lt_trans (trans h₁ h) h₂
  · intro h
    use (1.2 : ℝ)
    use (1.1 : ℝ)
    simp only [greatest_integer]
    norm_cast
    exact dec_trivial
sorry

end greatest_integer_inequality_l716_716286


namespace conditional_prob_B_given_A_l716_716291

-- First, define the set of numbers and events A and B
def num_set := {1, 2, 3, 4, 5}

def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0
def event_B (x y : ℕ) : Prop := (x % 2 = 0) ∧ (y % 2 = 0)

-- Define the probability calculation for event_A and event_B
def P_event_A : ℚ :=
  (nat.choose 3 2 + nat.choose 2 2) / nat.choose 5 2

def P_event_AB : ℚ :=
  nat.choose 2 2 / nat.choose 5 2

-- Define the conditional probability P(B|A)
def P_B_given_A : ℚ :=
  P_event_AB / P_event_A

theorem conditional_prob_B_given_A : P_B_given_A = 1 / 4 :=
by
  -- The proof would be placed here, but it's omitted based on the requirement
  sorry

end conditional_prob_B_given_A_l716_716291


namespace sin_mono_increasing_translated_l716_716010

theorem sin_mono_increasing_translated :
  ∀ x : ℝ, 
  ∃ f : ℝ → ℝ, 
    (∀ x, f x = sin (2 * (x - π / 10) + π / 5)) →
    (∀ a b : ℝ, (3 * π / 4 ≤ a ∧ a ≤ b ∧ b ≤ 5 * π / 4) → 
      f a ≤ f b) :=
by
  sorry

end sin_mono_increasing_translated_l716_716010


namespace solve_for_y_l716_716484

theorem solve_for_y (y : ℝ) : 7 - y = 4 → y = 3 :=
by
  sorry

end solve_for_y_l716_716484


namespace sum_of_values_l716_716450

def f (x : ℝ) : ℝ :=
if x < 3 then x^2 + 5*x + 6 else 3*x + 1

theorem sum_of_values (h : f 20) : -7 + 2 + 19 / 3 = 4 / 3 :=
sorry

end sum_of_values_l716_716450


namespace sector_angle_in_degrees_l716_716951

theorem sector_angle_in_degrees (c : ℝ) : 
  let θ := c * (180 / Real.pi) in θ ≈ 57 :=
by
  sorry

end sector_angle_in_degrees_l716_716951


namespace max_squares_covered_by_card_l716_716922

noncomputable def card_max_cover (width height : ℝ) : ℕ :=
  let diagonal := Real.sqrt (width ^ 2 + height ^ 2)
  in if diagonal < 2 then 4 else 0

theorem max_squares_covered_by_card : card_max_cover 1.8 0.8 = 4 := by
  sorry

end max_squares_covered_by_card_l716_716922


namespace point_in_fourth_quadrant_l716_716570

def complex_point_quadrant (z : ℂ) : ℕ :=
  if 0 < z.re ∧ 0 < z.im then 1
  else if z.re < 0 ∧ 0 < z.im then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if 0 < z.re ∧ z.im < 0 then 4
  else 0  -- this means the point lies on the axes which is not considered in the options

theorem point_in_fourth_quadrant : complex_point_quadrant ((2 - 1 * complex.i) * (2 - 1 * complex.i)) = 4 := sorry

end point_in_fourth_quadrant_l716_716570


namespace find_parabola_eq_line_passes_fixed_point_l716_716759

-- Definitions for conditions
def parabola_eq (p : ℝ) : Prop := p > 0 ∧ equation_parabola = (fun y => sqrt (2 * p * x)) -- Using parametric y^2 = 2px

def satisfies_distance (center : ℝ × ℝ) : Prop := center.1 - (-(p / 2)) = 3 / 2

def point_on_parabola (M : ℝ × ℝ) : Prop := M.1 * M.2 = 4 * M.1 -- Derived from y^2 = 4x and a given point

-- Theorems to prove
theorem find_parabola_eq (p : ℝ) : 
  parabola_eq p → satisfies_distance (p / 4, 0) → equation_parabola = (fun y => sqrt (4 * x)) := 
sorry

theorem line_passes_fixed_point (M DE : ℝ × ℝ) : 
  point_on_parabola M → (MD ⊥ ME) → passes_fixed_point DE (8, -4) :=
sorry

end find_parabola_eq_line_passes_fixed_point_l716_716759


namespace remainder_when_divided_by_11_l716_716901

theorem remainder_when_divided_by_11 (N : ℕ)
  (h₁ : N = 5 * 5 + 0) :
  N % 11 = 3 := 
sorry

end remainder_when_divided_by_11_l716_716901


namespace ab_divisible_by_six_l716_716807

def last_digit (n : ℕ) : ℕ :=
  (2 ^ n) % 10

def b_value (n : ℕ) (a : ℕ) : ℕ :=
  2 ^ n - a

theorem ab_divisible_by_six (n : ℕ) (h : n > 3) :
  let a := last_digit n
  let b := b_value n a
  ∃ k : ℕ, ab = 6 * k :=
by
  sorry

end ab_divisible_by_six_l716_716807


namespace opposite_sides_of_plane_l716_716696

noncomputable def point : Type := ℝ × ℝ × ℝ

def tetrahedron (S A B C : point) : Prop :=
  -- Mutually perpendicular edges SA, SB, SC
  let (sx, sy, sz) := S in
  let (ax, ay, az) := A in
  let (bx, by, bz) := B in
  let (cx, cy, cz) := C in
  ax > 0 ∧ ay = 0 ∧ az = 0 ∧
  bx = 0 ∧ by > 0 ∧ bz = 0 ∧
  cx = 0 ∧ cy = 0 ∧ cz > 0

def plane_equation (A B C : point) (x y z : ℝ) : ℝ :=
  let (ax, ay, az) := A in
  let (bx, by, bz) := B in
  let (cx, cy, cz) := C in
  (x / ax) + (y / by) + (z / cz)

def circumcenter (A B C : point) : point :=
  let (ax, ay, az) := A in
  let (bx, by, bz) := B in
  let (cx, cy, cz) := C in
  (ax / 2, by / 2, cz / 2)

theorem opposite_sides_of_plane (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  let S : point := (0, 0, 0)
  let A : point := (a, 0, 0)
  let B : point := (0, b, 0)
  let C : point := (0, 0, c)
  let O : point := circumcenter A B C in
  plane_equation A B C O > 1 ∧ plane_equation A B C S < 0 :=
by
  sorry

end opposite_sides_of_plane_l716_716696


namespace wrongly_written_height_is_176_l716_716494

-- Definitions and given conditions
def average_height_incorrect := 182
def average_height_correct := 180
def num_boys := 35
def actual_height := 106

-- The difference in total height due to the error
def total_height_incorrect := num_boys * average_height_incorrect
def total_height_correct := num_boys * average_height_correct
def height_difference := total_height_incorrect - total_height_correct

-- The wrongly written height
def wrongly_written_height := actual_height + height_difference

-- Proof statement
theorem wrongly_written_height_is_176 : wrongly_written_height = 176 := by
  sorry

end wrongly_written_height_is_176_l716_716494


namespace regular_polygon_sides_l716_716127

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716127


namespace proof_problem_l716_716723

open Set Real

noncomputable theory

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | -5 ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ 1}
def B : Set ℝ := {x : ℝ | log 2 (2 - x) ≤ 1}

theorem proof_problem :
  (A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1}) ∧
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 2}) ∧
  (compl B ∪ A = {x : ℝ | x ≤ 1 ∨ x ≥ 2}) :=
begin
  sorry
end

end proof_problem_l716_716723


namespace portion_to_joe_and_darcy_eq_half_l716_716821

open Int

noncomputable def portion_given_to_joe_and_darcy : ℚ := 
let total_slices := 8
let portion_to_carl := 1 / 4
let slices_to_carl := portion_to_carl * total_slices
let slices_left := 2
let slices_given_to_joe_and_darcy := total_slices - slices_to_carl - slices_left
let portion_to_joe_and_darcy := slices_given_to_joe_and_darcy / total_slices
portion_to_joe_and_darcy

theorem portion_to_joe_and_darcy_eq_half :
  portion_given_to_joe_and_darcy = 1 / 2 :=
sorry

end portion_to_joe_and_darcy_eq_half_l716_716821


namespace probability_of_two_red_shoes_is_1_3_l716_716520

noncomputable theory
open_locale classical

def num_red_shoes := 6
def num_green_shoes := 4
def total_shoes := num_red_shoes + num_green_shoes

def comb (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

def total_ways_to_draw_two_shoes := comb total_shoes 2
def ways_to_draw_two_red_shoes := comb num_red_shoes 2

def probability_two_red_shoes :=
  ways_to_draw_two_red_shoes / total_ways_to_draw_two_shoes

theorem probability_of_two_red_shoes_is_1_3 :
  probability_two_red_shoes = 1 / 3 :=
by
  sorry

end probability_of_two_red_shoes_is_1_3_l716_716520


namespace regular_polygon_sides_l716_716165

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716165


namespace number_of_alligators_l716_716378

theorem number_of_alligators (A : ℕ) 
  (num_snakes : ℕ := 18) 
  (total_eyes : ℕ := 56) 
  (eyes_per_snake : ℕ := 2) 
  (eyes_per_alligator : ℕ := 2) 
  (snakes_eyes : ℕ := num_snakes * eyes_per_snake) 
  (alligators_eyes : ℕ := A * eyes_per_alligator) 
  (total_animals_eyes : ℕ := snakes_eyes + alligators_eyes) 
  (total_eyes_eq : total_animals_eyes = total_eyes) 
: A = 10 :=
by 
  sorry

end number_of_alligators_l716_716378


namespace number_of_ones_minus_zeros_in_253_binary_l716_716983

def binary_representation (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | value => (binary_representation (value / 2)) * 10 + (value % 2)

-- The binary representation of 253 is 11111101
theorem number_of_ones_minus_zeros_in_253_binary : 
  let x := 1
  let y := 7
  y - x = 6 :=
by
  -- Definitions for x, y
  let x := 1
  let y := 7
  -- Required proof
  sorry

end number_of_ones_minus_zeros_in_253_binary_l716_716983


namespace regular_polygon_sides_l716_716083

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716083


namespace avg_multiples_of_10_l716_716566

-- Step 1: Defining the conditions
def sequence : List ℤ := List.range' 10 400 (λ n => 10 * n)

-- Step 2: Defining the question (arithmetic mean)
def average (l : List ℤ) : ℤ := l.sum / l.length

-- Step 3: The proof problem
theorem avg_multiples_of_10 (h : sequence.all (λ x => x % 10 = 0) ∧
                              sequence.head = 10 ∧ 
                              sequence.last = 400) : 
    average sequence = 205 := 
sorry

end avg_multiples_of_10_l716_716566


namespace solve_inequality_l716_716486

open Set

-- Define a predicate for the inequality solution sets
def inequality_solution_set (k : ℝ) : Set ℝ :=
  if h : k = 0 then {x | x < 1}
  else if h : 0 < k ∧ k < 2 then {x | x < 1 ∨ x > 2 / k}
  else if h : k = 2 then {x | True} \ {1}
  else if h : k > 2 then {x | x < 2 / k ∨ x > 1}
  else {x | 2 / k < x ∧ x < 1}

-- The statement of the proof
theorem solve_inequality (k : ℝ) :
  ∀ x : ℝ, k * x^2 - (k + 2) * x + 2 < 0 ↔ x ∈ inequality_solution_set k :=
by
  sorry

end solve_inequality_l716_716486


namespace distance_between_polar_points_A_B_is_4_l716_716764

noncomputable def distance_between_points 
  (r1 θ1 r2 θ2 : ℝ) : ℝ :=
  real.sqrt ((r1 * cos θ1 - r2 * cos θ2)^2 + (r1 * sin θ1 - r2 * sin θ2)^2)

theorem distance_between_polar_points_A_B_is_4 :
  distance_between_points 3 (5 * real.pi / 3) 1 (2 * real.pi / 3) = 4 :=
by 
  sorry

end distance_between_polar_points_A_B_is_4_l716_716764


namespace not_possible_total_l716_716767

-- Definitions
variables (d r : ℕ)

-- Theorem to prove that 58 cannot be expressed as 26d + 3r
theorem not_possible_total : ¬∃ (d r : ℕ), 26 * d + 3 * r = 58 :=
sorry

end not_possible_total_l716_716767


namespace area_triangle_PQR_l716_716923

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) := { point : ℝ × ℝ | (point.1 - center.1) ^ 2 + (point.2 - center.2) ^ 2 = radius ^ 2 }

noncomputable def Triangle.area (A B C : ℝ × ℝ) : ℝ :=
abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Define specific circles
def circle1 : set (ℝ × ℝ) := circle (0, 7) 2
def circle2 : set (ℝ × ℝ) := circle (0, 3) 3

-- Define points P, Q, R, and assert conditions
axiom P : ℝ × ℝ := (0, 10)
axiom Q : ℝ × ℝ := (-3.6, 0)
axiom R : ℝ × ℝ := (3.6, 0)

-- The main theorem to prove "Area of triangle PQR is 12.38√21"
theorem area_triangle_PQR : Triangle.area P Q R = 12.38 * real.sqrt 21 :=
sorry

end area_triangle_PQR_l716_716923


namespace tanya_bought_11_pears_l716_716492

variable (P : ℕ)

-- Define the given conditions about the number of different fruits Tanya bought
def apples : ℕ := 4
def pineapples : ℕ := 2
def basket_of_plums : ℕ := 1

-- Define the total number of fruits initially and the remaining fruits
def initial_fruit_total : ℕ := 18
def remaining_fruit_total : ℕ := 9
def half_fell_out_of_bag : ℕ := remaining_fruit_total * 2

-- The main theorem to prove
theorem tanya_bought_11_pears (h : P + apples + pineapples + basket_of_plums = initial_fruit_total) : P = 11 := by
  -- providing a placeholder for the proof
  sorry

end tanya_bought_11_pears_l716_716492


namespace maintain_mean_and_variance_l716_716418

def initial_set : Finset ℤ :=
  {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset ℤ) : ℚ :=
  (s.sum id : ℚ) / s.card

def variance (s : Finset ℤ) : ℚ :=
  (s.sum (λ x, (x * x : ℚ))) / (s.card : ℚ) - (mean s)^2

theorem maintain_mean_and_variance :
  ∃ (a b c : ℤ), a ∈ initial_set ∧
                 b ∉ initial_set ∧ 
                 c ∉ initial_set ∧ 
                 mean initial_set = mean (initial_set.erase a ∪ {b, c}) ∧
                 variance initial_set = variance (initial_set.erase a ∪ {b, c})
  :=
begin
  sorry
end

end maintain_mean_and_variance_l716_716418


namespace projectile_reaches_64_feet_at_t_l716_716848

theorem projectile_reaches_64_feet_at_t (t : ℝ) : 
    (y = -16 * t ^ 2 + 100 * t) ∧ (y = 64) → t ≈ 0.7 :=
by
  sorry

end projectile_reaches_64_feet_at_t_l716_716848


namespace train_length_is_170_meters_l716_716947

-- Definition of the conditions
def speed_km_per_hr := 45
def time_seconds := 30
def bridge_length_meters := 205

-- Convert speed from km/hr to m/s
def speed_m_per_s : ℝ := (speed_km_per_hr : ℝ) * 1000 / 3600

-- The total distance covered by the train in 30 seconds
def total_distance_m : ℝ := speed_m_per_s * (time_seconds : ℝ)

-- The length of the train
def length_of_train_m : ℝ := total_distance_m - (bridge_length_meters : ℝ)

-- The theorem we need to prove
theorem train_length_is_170_meters : 
  length_of_train_m = 170 := by
    sorry

end train_length_is_170_meters_l716_716947


namespace train_crossing_time_l716_716611

theorem train_crossing_time
  (length : ℝ) (speed : ℝ) (time : ℝ)
  (h1 : length = 100) (h2 : speed = 30.000000000000004) :
  time = length / speed :=
by
  sorry

end train_crossing_time_l716_716611


namespace count_numbers_with_3_digit_between_200_and_499_l716_716359

def contains_digit_3 (n : ℕ) : Prop := 
  n.to_digits 10 contains 3

theorem count_numbers_with_3_digit_between_200_and_499 : 
  (Finset.filter contains_digit_3 (Finset.Ico 200 500)).card = 138 := 
by 
  sorry

end count_numbers_with_3_digit_between_200_and_499_l716_716359


namespace volume_tetrahedron_l716_716708

variables {A B C A1 B1 C1 D : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [AddCommGroup A1] [AddCommGroup B1] [AddCommGroup C1] [AddCommGroup D]
variables (segBC : ℝ) (segA1B1 : ℝ) (midD : ℝ)
variables (vol : ℝ)

-- Define the conditions
def baseEdgeLength := segBC = 2
def lateralEdgeLength := segA1B1 = √3
def midpointBD := midD = 1

-- Main theorem to prove
theorem volume_tetrahedron (h1 : baseEdgeLength segBC) (h2 : lateralEdgeLength segA1B1) (h3 : midpointBD midD) : vol = 1 :=
sorry

end volume_tetrahedron_l716_716708


namespace sin_probability_interval_l716_716930

theorem sin_probability_interval :
  let I : set ℝ := set.Icc (-1 : ℝ) (1 : ℝ)
  let J : set ℝ := set.Icc (-1 / 2) (real.sqrt 2 / 2)
  let range_x : set ℝ := { x : ℝ | (sin (real.pi * x / 4)) ∈ J }
  set.density I range_x = 5 / 6 :=
by
  -- Here we assume that set.density is the appropriate formalization of the desired probability.
  sorry

end sin_probability_interval_l716_716930


namespace product_of_large_numbers_has_38_digits_l716_716644

theorem product_of_large_numbers_has_38_digits :
  ∀ (a b : ℕ), 
    a = 1002000000000000000 ∧ b = 999999999999999999 → 
    (nat.log10 (a * b) + 1) = 38 :=
begin
  intros a b h,
  cases h with ha hb,
  sorry
end

end product_of_large_numbers_has_38_digits_l716_716644


namespace regular_polygon_sides_l716_716087

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716087


namespace find_initial_amount_l716_716423

noncomputable def initial_amount_in_piggy_bank (weekly_allowance : ℕ) (given_after_weeks : ℕ) (weeks : ℕ) : ℕ :=
  let added_amount := (weeks * (weekly_allowance / 2))
  given_after_weeks - added_amount

theorem find_initial_amount :
  ∀ (initial_amount : ℕ), 
    ∀ (weekly_allowance : ℕ), 
      ∀ (given_after_weeks : ℕ), 
        ∀ (weeks : ℕ),
          weekly_allowance = 10 →
          given_after_weeks = 83 →
          weeks = 8 →
          initial_amount_in_piggy_bank weekly_allowance given_after_weeks weeks = initial_amount :=
begin
  assume initial_amount weekly_allowance given_after_weeks weeks,
  assume weekly_allowance_eq given_after_weeks_eq weeks_eq,
  have half_allowance : ℕ := weekly_allowance / 2,
  have added_amount := weeks * half_allowance,
  have init_amount := given_after_weeks - added_amount,
  rw [weekly_allowance_eq, given_after_weeks_eq, weeks_eq],
  have : added_amount = 40 := rfl,
  have : init_amount = 43 := rfl,
  have : initial_amount = 43 := rfl,
  sorry
end

end find_initial_amount_l716_716423


namespace regular_polygon_sides_l716_716069

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716069


namespace factorize_1_factorize_2_factorize_3_l716_716273

-- Problem 1: Factorize 3a^3 - 6a^2 + 3a
theorem factorize_1 (a : ℝ) : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
sorry

-- Problem 2: Factorize a^2(x - y) + b^2(y - x)
theorem factorize_2 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
sorry

-- Problem 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factorize_3 (a b : ℝ) : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
sorry

end factorize_1_factorize_2_factorize_3_l716_716273


namespace inverse_proportion_inequality_l716_716313

theorem inverse_proportion_inequality {x1 x2 : ℝ} (h1 : x1 > x2) (h2 : x2 > 0) : 
    -3 / x1 > -3 / x2 := 
by 
  sorry

end inverse_proportion_inequality_l716_716313


namespace rectangle_area_in_square_l716_716229

theorem rectangle_area_in_square :
  ∃ (area : ℝ), (side : ℝ) (leg : ℝ),
  side = 4 ∧
  leg = (side * real.sqrt 2) / 2 ∧
  area = (leg * leg) ∧
  area = 8 :=
by {
  sorry
}

end rectangle_area_in_square_l716_716229


namespace nth_equation_holds_l716_716466

theorem nth_equation_holds (n : ℕ) : 
  let eq_fst := (4^2 - 1^2 - 9) / 2 = 3
  let eq_snd := (5^2 - 2^2 - 9) / 2 = 6
  let eq_trd := (6^2 - 3^2 - 9) / 2 = 9
  let eq_fourth := (7^2 - 4^2 - 9) / 2 = 12
  in (eq_fst ∧ eq_snd ∧ eq_trd ∧ eq_fourth) →
     ( (n+3)^2 - n^2 - 9 ) / 2 = 3 * n := by
  intros
  sorry

end nth_equation_holds_l716_716466


namespace ratio_side_length_to_perimeter_l716_716938

theorem ratio_side_length_to_perimeter (side_length : ℝ) (perimeter : ℝ) (h_side : side_length = 15) (h_perimeter : perimeter = 4 * side_length) : side_length / perimeter = 1 / 4 :=
by
  rw [h_side, h_perimeter]
  norm_num
  simp


end ratio_side_length_to_perimeter_l716_716938


namespace regular_polygon_sides_l716_716202

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716202


namespace appropriate_sampling_methods_l716_716748

structure Region :=
  (total_households : ℕ)
  (farmer_households : ℕ)
  (worker_households : ℕ)
  (sample_size : ℕ)

theorem appropriate_sampling_methods (r : Region) 
  (h_total: r.total_households = 2004)
  (h_farmers: r.farmer_households = 1600)
  (h_workers: r.worker_households = 303)
  (h_sample: r.sample_size = 40) :
  ("Simple random sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Systematic sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Stratified sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) :=
by
  sorry

end appropriate_sampling_methods_l716_716748


namespace minimum_value_of_v_l716_716337

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

noncomputable def g (x u v : ℝ) : ℝ := (x - u)^3 - 3 * (x - u) - v

theorem minimum_value_of_v (u v : ℝ) (h_pos_u : u > 0) :
  ∀ u > 0, ∀ x : ℝ, f x = g x u v → v ≥ 4 :=
by
  sorry

end minimum_value_of_v_l716_716337


namespace measure_of_angle_Q_l716_716354

theorem measure_of_angle_Q (a b c d e Q : ℝ)
  (ha : a = 138) (hb : b = 85) (hc : c = 130) (hd : d = 120) (he : e = 95)
  (h_hex : a + b + c + d + e + Q = 720) : 
  Q = 152 :=
by
  rw [ha, hb, hc, hd, he] at h_hex
  linarith

end measure_of_angle_Q_l716_716354


namespace problem_general_term_problem_max_sum_l716_716710

noncomputable def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def a_n_term (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  a 0 + n * (-2)

theorem problem_general_term
  (a : ℕ → ℤ)
  (h₁ : arithmetic_seq a (-2))
  (h₂ : a 3 = a 2 + a 5) :
  ∀ n, a n = 8 - 2 * n := sorry

noncomputable def sum_seq (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a i

theorem problem_max_sum
  (a : ℕ → ℤ)
  (h₁ : arithmetic_seq a (-2))
  (h₂ : a 3 = a 2 + a 5) :
  (∃ n, sum_seq a n = 12) := sorry

end problem_general_term_problem_max_sum_l716_716710


namespace ratio_of_2_pointers_l716_716461

theorem ratio_of_2_pointers 
  (M2P : ℕ := 25) 
  (M3P : ℕ := 8) 
  (MFT : ℕ := 10) 
  (TotalPoints : ℕ := 201) : 
  let 
    MarkPoints := 2 * M2P + 3 * M3P + MFT, 
    OpponentPoints := 2 * a * M2P + (3 * (M3P / 2)) + (MFT / 2) 
  in 
    MarkPoints + OpponentPoints = TotalPoints → a = 2 :=
by 
  let MarkPoints := 2 * M2P + 3 * M3P + MFT;
  let OpponentPoints := 2 * b * M2P + (3 * (M3P / 2)) + (MFT / 2);
  assume h : MarkPoints + OpponentPoints = TotalPoints;
  have : b = 2 := sorry;
  exact this

end ratio_of_2_pointers_l716_716461


namespace find_number_l716_716590

theorem find_number (n : ℝ) (h : n / 0.04 = 400.90000000000003) : n = 16.036 := 
by
  sorry

end find_number_l716_716590


namespace regular_polygon_sides_l716_716191

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716191


namespace expected_stand_ups_expected_never_stand_ups_l716_716464

open Classical

-- a) Expected number of times someone will stand up
theorem expected_stand_ups (n : ℕ) : 
  (n * (n - 1)) / 4 = (∑ i in finset.range n, ∑ j in finset.range i, 1 / 2) :=
sorry

-- b) Expected number of people who will never have to stand up
theorem expected_never_stand_ups (n : ℕ) : 
  (1 + (∑ i in finset.range (n - 1), 1 / (i + 1))) = (∑ i in finset.range n, 1 / (i + 1)) :=
sorry

end expected_stand_ups_expected_never_stand_ups_l716_716464


namespace longest_side_length_is_sqrt_10_l716_716253

noncomputable def longest_side_of_quadrilateral : ℝ :=
  let x := ℝ
  let y := ℝ
  let vertices := set.univ.filter(fun (p : ℝ × ℝ) => 
    let (x, y) := p in
    (x + 2 * y <= 6) ∧ (x + y >= 1) ∧ (x >= 0) ∧ (y >= 0))
  in
  let distances := set.to_finset vertices.to_finset.filter_map(fun (v1 : ℝ × ℝ) =>
    vertices.to_finset.to_list.map(fun (v2 : ℝ × ℝ) => 
      if v1 != v2 then some (dist v1 v2) else none))
  in
  sorry

theorem longest_side_length_is_sqrt_10 :
  longest_side_of_quadrilateral = real.sqrt 10 :=
sorry

end longest_side_length_is_sqrt_10_l716_716253


namespace finite_cut_out_squares_l716_716824

theorem finite_cut_out_squares (n : ℕ) :
  ∃ (K : set (set (ℕ × ℕ))),
    (∀ k ∈ K, ∃ N : ℕ, N > 0 ∧ ∀ x y, x < N ∧ y < N → (x, y) ∈ k) ∧
    (∀ k ∈ K, (∃ B : set (ℕ × ℕ), card B = n ∧ B ⊆ k) ∧
      (1/5 * (card k : ℝ) ≤ card B ∧ card B ≤ 4/5 * (card k : ℝ))) := sorry

end finite_cut_out_squares_l716_716824


namespace nine_point_circle_l716_716421

open_locale euclidean_geometry

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def altitude_foot (A B C : Point) : Point := sorry

theorem nine_point_circle (A B C : Point) :
  let H := orthocenter A B C,
      L := midpoint B C,
      M := midpoint C A,
      N := midpoint A B,
      D := altitude_foot A B C,
      E := altitude_foot B C A,
      F := altitude_foot C A B,
      P := midpoint H A,
      Q := midpoint H B,
      R := midpoint H C
  in is_cyclic {L, M, N, D, E, F, P, Q, R} :=
by {
  sorry
}

end nine_point_circle_l716_716421


namespace maclaurin_series_ex_maclaurin_series_sin_maclaurin_series_cos_l716_716657

noncomputable theory

open Real

theorem maclaurin_series_ex (x : ℝ) :
  (∀ n : ℕ, (∂^[n] (λ x:ℝ, exp x) 0 = 1)) ∧
  (∀ n : ℕ, exp x = ∑' n, x^n / Nat.factorial n) := 
sorry

theorem maclaurin_series_sin (x : ℝ) :
  (∀ n, (deriv^[n] sin 0 = if even n then 0 else (-1)^((n-1)/2))) ∧
  (sin x = ∑' n, (-1)^n * x^(2*n+1) / Nat.factorial (2*n+1)) :=
sorry

theorem maclaurin_series_cos (x : ℝ) :
  (∀ n, (deriv^[n] cos 0 = if even n then (-1)^(n/2) else 0)) ∧
  (cos x = ∑' n, (-1)^n * x^(2*n) / Nat.factorial (2*n)) := 
sorry

end maclaurin_series_ex_maclaurin_series_sin_maclaurin_series_cos_l716_716657


namespace mass_percentage_of_Cl_in_NaOCl_l716_716664

theorem mass_percentage_of_Cl_in_NaOCl :
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  100 * (Cl_mass / NaOCl_mass) = 47.6 := 
by
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  sorry

end mass_percentage_of_Cl_in_NaOCl_l716_716664


namespace regular_polygon_sides_l716_716100

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716100


namespace gcf_of_terms_l716_716663

theorem gcf_of_terms (x y : ℝ) : 
  let term1 := 9 * x^3 * y^2 in
  let term2 := 12 * x^2 * y^3 in
  let gcf := 3 * x^2 * y^2 in
  gcf = greatest_common_factor [term1, term2] :=
sorry

end gcf_of_terms_l716_716663


namespace sum_expression_l716_716306

noncomputable def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -1 else -2 * 3^(n-2)

noncomputable def sum_S (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i, sequence_a (i + 1))

theorem sum_expression (n : ℕ) : sum_S n = -3^(n-1) := by
  sorry

end sum_expression_l716_716306


namespace cyclic_quadrilateral_l716_716536

variables (A B C K L M : Point) (l : Line)

-- Assume triangle ABC
variables (h_triangle_abc : Triangle A B C)

-- Assume point M is given
variables (h_m : ∃ (m_line : Line), lies_on M m_line)

-- Assume l intersects sides AC and BC at points K and L respectively
variables (h_intersects_ac : Intersects l (Segment A C) K)
variables (h_intersects_bc : Intersects l (Segment B C) L)

-- Assume quadrilateral AKLB is cyclic
theorem cyclic_quadrilateral (h_cyclic : CyclicQuadrilateral A K L B) : 
  CyclicQuadrilateral A K L B :=
sorry

end cyclic_quadrilateral_l716_716536


namespace directional_derivative_at_M0_l716_716044

variables {α : Type*} [RealField α]
variables {x y z x0 y0 z0 a : α}
variables {u : α → α → α → α} {φ : α → α → α → α}
variables {τ : α × α × α} -- τ is the unit tangent vector

-- Conditions:
def condition1 : u x y z = a := sorry
def condition2 : φ x y z = 0 := sorry
def M0 : α × α × α := (x0, y0, z0)
def unit_tangent_vector : α × α × α := τ

-- Gradient of u
def grad_u (u : α → α → α → α) (p : α × α × α) : α × α × α :=
  (partial_deriv (u p.1 p.2) p.1, partial_deriv (u p.1 p.2) p.2, partial_deriv (u p.1 p.2) p.3)

-- Directional Derivative
noncomputable def directional_derivative (u : α → α → α → α) (τ : α × α × α) (p : α × α × α) : α :=
  dot_product (grad_u u p) τ

-- Problem Statement
theorem directional_derivative_at_M0 : directional_derivative u τ M0 ≠ 0 :=
sorry

end directional_derivative_at_M0_l716_716044


namespace total_students_l716_716534

theorem total_students (total_students_with_brown_eyes total_students_with_black_hair: ℕ)
    (h1: ∀ (total_students : ℕ), (2 * total_students_with_brown_eyes) = 3 * total_students)
    (h2: (2 * total_students_with_black_hair) = total_students_with_brown_eyes)
    (h3: total_students_with_black_hair = 6) : 
    ∃ total_students : ℕ, total_students = 18 :=
by
  sorry

end total_students_l716_716534


namespace distinct_lines_intersection_points_sum_l716_716676

theorem distinct_lines_intersection_points_sum 
  (lines : Finset (Set ℝ × Set ℝ))
  (h_distinct : ∀ (l1 l2 : Set ℝ × Set ℝ), l1 ∈ lines → l2 ∈ lines → l1 ≠ l2)
  (h_lines_count : lines.card = 5) :
  ∑ N in {N | ∃ (intersection_points : Finset (ℝ × ℝ)), intersection_points.card = N ∧ 
          ∀ (p ∈ intersection_points), ∃ (l1 l2 : Set ℝ × Set ℝ),
          l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ p ∈ (l1 ∩ l2)}, id N = 37 :=
sorry

end distinct_lines_intersection_points_sum_l716_716676


namespace regular_polygon_sides_l716_716196

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716196


namespace sin_graph_shift_l716_716500

theorem sin_graph_shift (x : ℝ) : 
  (sin (2 * (x + π / 6))) = sin (2 * x + π / 3) := 
by 
  sorry

end sin_graph_shift_l716_716500


namespace area_of_given_right_triangle_l716_716020

variable (a c : ℝ)
def is_right_triangle (a c : ℝ) : Prop := ∃ b : ℝ, a^2 + b^2 = c^2
def area_of_right_triangle (a : ℝ) (b : ℝ) : ℝ := (1 / 2) * a * b

theorem area_of_given_right_triangle (h : is_right_triangle 15 17) : 
  ∃ (b : ℝ), (15^2 + b^2 = 17^2) ∧ (area_of_right_triangle 15 b = 60) :=
by sorry

end area_of_given_right_triangle_l716_716020


namespace regular_polygon_sides_l716_716120

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716120


namespace regular_polygon_sides_l716_716204

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716204


namespace function_is_constant_and_straight_line_l716_716743

-- Define a function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition: The derivative of f is 0 everywhere
axiom derivative_zero_everywhere : ∀ x, deriv f x = 0

-- Conclusion: f is a constant function
theorem function_is_constant_and_straight_line : ∃ C : ℝ, ∀ x, f x = C := by
  sorry

end function_is_constant_and_straight_line_l716_716743


namespace ladder_rectangle_count_l716_716600

open Finset

def count_rectangles_in_ladder (n : ℕ) : ℕ :=
  ∑ i in range n, (choose (11 - i) 2) * (choose (11 - i) 2)

theorem ladder_rectangle_count :
  count_rectangles_in_ladder 10 = 715 :=
sorry

end ladder_rectangle_count_l716_716600


namespace parallel_perpendicular_l716_716364

variable {m n : Line} {α : Plane}

theorem parallel_perpendicular (h1 : m ∥ n) (h2 : n ⊥ α) : m ⊥ α :=
sorry

end parallel_perpendicular_l716_716364


namespace replace_preserve_mean_variance_l716_716402

theorem replace_preserve_mean_variance:
  ∀ (a b c : ℤ), 
    let initial_set := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map (λ x, x : ℤ) in
    let new_set := (initial_set.erase a).++[b, c] in
    let mean (s : List ℤ) := (s.sum : ℚ) / s.length in
    let variance (s : List ℤ) :=
      let m := mean s in
      (s.map (λ x, (x - m) ^ 2)).sum / s.length in
    mean initial_set = 0 ∧ variance initial_set = 10 ∧
    ((mean new_set = 0 ∧ variance new_set = 10) ↔ ((a = -4 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 5))) :=
sorry

end replace_preserve_mean_variance_l716_716402


namespace regular_polygon_sides_l716_716101

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716101


namespace speed_of_current_l716_716867

noncomputable def distance_km : ℝ := 45.5 / 1000
noncomputable def time_hr : ℝ := 9.099272058235341 / 3600
noncomputable def speed_still_water : ℝ := 9.5
noncomputable def speed_downstream : ℝ := distance_km / time_hr

theorem speed_of_current : 
  let C := speed_downstream - speed_still_water in
  C = 8.5 :=
by
  let distance_km := distance_km
  let time_hr := time_hr
  let speed_still_water := speed_still_water
  let speed_downstream := speed_downstream
  let C := speed_downstream - speed_still_water
  show C = 8.5
  sorry

end speed_of_current_l716_716867


namespace eval_floor_ceil_sum_l716_716271

theorem eval_floor_ceil_sum : int.floor (-2.54) + int.ceil (25.4) = 23 := by
  sorry

end eval_floor_ceil_sum_l716_716271


namespace probability_product_multiple_of_four_l716_716752

/-
  Problem:
  In a jar, there are twelve balls numbered from 1 to 12.
  Samuel removes one ball randomly. Then, Clara removes another ball randomly.

  Prove that the probability that the product of the numbers on the balls
  removed is a multiple of four is 13/44.
-/

theorem probability_product_multiple_of_four : 
    let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        sample_space := { (a, b) | a ∈ balls ∧ b ∈ balls ∧ a ≠ b },
        total_outcomes := 12 * 11,
        events := { p | p ∈ sample_space ∧ (p.1 * p.2) % 4 = 0 },
        favorable_outcomes := 39 in
    (favorable_outcomes / total_outcomes) = (13 / 44) :=
by {
    sorry
}

end probability_product_multiple_of_four_l716_716752


namespace find_f_neg_a_l716_716454

def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

theorem find_f_neg_a (h : f a = 11) : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l716_716454


namespace digits_once_divisible_by_4_count_l716_716730

theorem digits_once_divisible_by_4_count : 
  (∃ digits : Finset (Fin 4 → ℕ), 
    (∀ d ∈ digits, (∀ i, d i ∈ {1, 2, 3, 4}) ∧ d 0 ≠ d 1 ∧ d 0 ≠ d 2 ∧ d 0 ≠ d 3 ∧ d 1 ≠ d 2 ∧ d 1 ≠ d 3 ∧ d 2 ≠ d 3) ∧ 
    (∀ n ∈ digits, n 2 * 10 + n 3 % 4 = 0) ∧ 
    digits.card = 6
  ) := sorry

end digits_once_divisible_by_4_count_l716_716730


namespace teresa_spent_40_dollars_l716_716843

def sandwich_price : ℝ := 7.75
def number_of_sandwiches : ℕ := 2
def salami_price : ℝ := 4.00
def brie_multiplier : ℝ := 3
def olive_price_per_pound : ℝ := 10.00
def olives_weight : ℝ := 1/4
def feta_price_per_pound : ℝ := 8.00
def feta_weight : ℝ := 1/2
def bread_price : ℝ := 2.00

noncomputable def total_spent : ℝ :=
  number_of_sandwiches * sandwich_price +
  salami_price +
  brie_multiplier * salami_price +
  olive_price_per_pound * olives_weight +
  feta_price_per_pound * feta_weight +
  bread_price 

theorem teresa_spent_40_dollars :
  total_spent = 40.00 := by
  sorry

end teresa_spent_40_dollars_l716_716843


namespace speed_second_hour_l716_716003

noncomputable def speed_in_first_hour : ℝ := 95
noncomputable def average_speed : ℝ := 77.5
noncomputable def total_time : ℝ := 2
def speed_in_second_hour : ℝ := sorry -- to be deduced

theorem speed_second_hour :
  speed_in_second_hour = 60 :=
by
  sorry

end speed_second_hour_l716_716003


namespace parallel_lines_intersection_perimeter_OPQ_equal_AB_l716_716878

-- Define the triangle ABC and its incenter O
variables {A B C O M N P Q : Type*}
variables [Incenter O A B C]

-- Define lines through O parallel to sides of the triangle
variables (MO : Line) (NO : Line) (PO : Line) (QO : Line)
-- Define points of intersection
variables (M : Point) (N : Point) (P : Point) (Q : Point)

namespace geometry

theorem parallel_lines_intersection
  (hO : incenter O A B C)
  (hMO : parallel MO (line AB))
  (hNO : parallel NO (line AB))
  (hP : intersection P (line MO) (line AB))  -- P is intersection of MO and AB
  (hQ : intersection Q (line NO) (line AB))  -- Q is intersection of NO and AB
  (hM : intersection M (line MO) (line AC))  -- M is intersection of MO and AC
  (hN : intersection N (line NO) (line BC))  -- N is intersection of NO and BC) :
  (hMN : length (segment M N) = length (segment A M) + length (segment B N)) -- Question 1
  : MN = AM + BN := by sorry

theorem perimeter_OPQ_equal_AB
  (hO : incenter O A B C)
  (hMO : parallel MO (line AB))
  (hNO : parallel NO (line AB))
  (hP : intersection P (line MO) (line AB))  -- P is intersection of MO and AB
  (hQ : intersection Q (line NO) (line AB))  -- Q is intersection of NO and AB
  (hM : intersection M (line MO) (line AC))  -- M is intersection of MO and AC
  (hN : intersection N (line NO) (line BC))  -- N is intersection of NO and BC) :
  (hMN : length (segment M N) = length (segment A M) + length (segment B N))
  (hOPQ : perimeter triangle O P Q)
  : perimeter OPQ = length (segment AB) := by sorry

end geometry

end parallel_lines_intersection_perimeter_OPQ_equal_AB_l716_716878


namespace age_ratio_l716_716242

-- Define the conditions
def ArunCurrentAgeAfter6Years (A: ℕ) : Prop := A + 6 = 36
def DeepakCurrentAge : ℕ := 42

-- Define the goal statement
theorem age_ratio (A: ℕ) (hc: ArunCurrentAgeAfter6Years A) : A / gcd A DeepakCurrentAge = 5 ∧ DeepakCurrentAge / gcd A DeepakCurrentAge = 7 :=
by
  sorry

end age_ratio_l716_716242


namespace dan_minimum_speed_to_beat_cara_l716_716496

theorem dan_minimum_speed_to_beat_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 120 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (dan_speed : ℕ), dan_speed > 40 :=
by
  sorry

end dan_minimum_speed_to_beat_cara_l716_716496


namespace max_distinct_numbers_with_sum_power_of_two_l716_716837

theorem max_distinct_numbers_with_sum_power_of_two : 
  ∀ (S : Finset ℕ), (∀ a b ∈ S, a ≠ b → ∃ k : ℕ, a + b = 2^k) → S.card ≤ 2 :=
by
  -- proof to be provided
  sorry

end max_distinct_numbers_with_sum_power_of_two_l716_716837


namespace minimum_distance_l716_716827

-- Define the line equation coefficient
def A : ℝ := 3
def B : ℝ := -4
def C : ℝ := 2

-- Define the point (3, -1)
def x₀ : ℝ := 3
def y₀ : ℝ := -1

-- Define the formula for the distance from a point to a line
def distance (A B C x₀ y₀ : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / (Real.sqrt (A^2 + B^2))

-- Prove that the minimum distance from the point (3, -1) to the line 3x - 4y + 2 = 0 is 3
theorem minimum_distance : distance A B C x₀ y₀ = 3 := by
  sorry

end minimum_distance_l716_716827


namespace purchase_price_eq_80_l716_716608

/-- A store sells a certain model of glasses. The price is increased by 9 times the purchase price, 
then advertised with a "50% discount plus a 20 yuan taxi fare" promotion. Selling a pair of these glasses 
still yields a profit of 300 yuan. Prove that the purchase price of the glasses is 80 yuan. --/
theorem purchase_price_eq_80 (x : ℕ) (h1 : ∀ n : ℕ, 10 * n - 20 - n = 300) : x = 80 := 
by
  have eqn : 10*x*0.5 - 20 - x = 300 := by 
    calc 
      10*x*0.5 - 20 - x = (5*x - 20) - x : by ring
      ... = (4*x - 20) = 300 : h1  x
  sorry

end purchase_price_eq_80_l716_716608


namespace basketball_team_minimum_score_l716_716913

theorem basketball_team_minimum_score (n : ℕ) (score : ℕ) :
  (n = 12) →
  (∀ i : fin n, 7 ≤ score) →
  (∀ i : fin n, score i ≤ 23) →
  (84 ≤ (∑ i : fin n, score i)) :=
by sorry

end basketball_team_minimum_score_l716_716913


namespace expected_reflection_value_l716_716390

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) *
  (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem expected_reflection_value :
  expected_reflections = (2 / Real.pi) *
    (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end expected_reflection_value_l716_716390


namespace inequality_1_inequality_2_l716_716840

-- Define the first inequality proof problem
theorem inequality_1 (x : ℝ) : 5 * x + 3 < 11 + x ↔ x < 2 := by
  sorry

-- Define the second set of inequalities proof problem
theorem inequality_2 (x : ℝ) : 
  (2 * x + 1 < 3 * x + 3) ∧ ((x + 1) / 2 ≤ (1 - x) / 6 + 1) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end inequality_1_inequality_2_l716_716840


namespace percentage_loss_l716_716058

theorem percentage_loss (SP_loss SP_profit CP : ℝ) 
  (h₁ : SP_loss = 9) 
  (h₂ : SP_profit = 11.8125) 
  (h₃ : SP_profit = CP * 1.05) : 
  (CP - SP_loss) / CP * 100 = 20 :=
by sorry

end percentage_loss_l716_716058


namespace convert_deg_to_rad_l716_716254

theorem convert_deg_to_rad (deg : ℝ) (π : ℝ) (h : deg = 50) : (deg * (π / 180) = 5 / 18 * π) :=
by
  -- Conditions
  sorry

end convert_deg_to_rad_l716_716254


namespace square_area_from_y_coordinates_l716_716228

theorem square_area_from_y_coordinates (y1 y2 y3 y4 : ℤ) (h : {y1, y2, y3, y4} = {2, 3, 6, 7}) :
  let side := 4 in
  let area := side * side in
  area = 16 :=
by
  sorry

end square_area_from_y_coordinates_l716_716228


namespace regular_polygon_sides_l716_716167

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716167


namespace sufficient_and_necessary_condition_l716_716609

theorem sufficient_and_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, m * x ^ 2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m < -1 / 2) :=
by
  sorry

end sufficient_and_necessary_condition_l716_716609


namespace basketball_team_score_l716_716914

theorem basketball_team_score (n m k : ℕ) (h_team_size : n = 12) (h_min_points : m = 7) 
                              (h_max_points : k = 23) (scores : Fin n → ℕ)
                              (h_scores_min : ∀ i, scores i ≥ m)
                              (h_scores_max : ∀ i, scores i ≤ k) :
                              (∑ i, scores i) ≥ 84 := 
by
  sorry

end basketball_team_score_l716_716914


namespace fifth_number_in_21st_row_l716_716467

/-- Define the nth row as an arithmetic sequence -/
def nth_row (n : ℕ) : List ℕ :=
  List.range (n * n) |>.drop ((n - 1) * n) |>.map (λ x, x + 1)

/-- Define the far right number in the nth row -/
def far_right (n : ℕ) : ℕ :=
  n * n

/-- Define the start of the next row based on far_right -/
def next_start (n : ℕ) : ℕ :=
  far_right n + 1

theorem fifth_number_in_21st_row : nth_row 21 ![4] = 405 := by sorry

end fifth_number_in_21st_row_l716_716467


namespace concurrency_of_lines_l716_716009

noncomputable def is_incenter (O A B C : point) : Prop :=
  ∃ (P Q : point), midpoint P Q O ∧ segment P A = segment P B ∧ segment P Q ∠ segment O A = π/2

noncomputable def is_isosceles (A B C : point) : Prop :=
  dist A B = dist A C

noncomputable def are_concurrent (S T U V : line) : Prop :=
  ∃ (P : point), on_line P S ∧ on_line P T ∧ on_line P U ∧ on_line P V

def lies_on_same_line (S T U V : line) : Prop :=
  ∃ (L : line), on_line P L ∧ on_line P P ∧ on_line P U ∧ on_line P V 

theorem concurrency_of_lines (A B C O D K : point) (BO CA BD OK : line) :
  is_isosceles A B C → 
  is_incenter O A B C →
  perpendicular BO AC →
  perpendicular CO BD →
  altitude CA B O →
  altitude BD C O →
  radius OK intersects CA BD → 
  are_concurrent BO CA BD OK →
  lies_on_same_line BO CA BD OK :=
by sorry


end concurrency_of_lines_l716_716009


namespace mean_and_variance_unchanged_l716_716395

noncomputable def initial_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
noncomputable def replaced_set_1 : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 5, -1}
noncomputable def replaced_set_2 : Set ℤ := {-5, -3, -2, -1, 0, 1, 2, 3, 4, -5, 1}

noncomputable def mean (s : Set ℤ) : ℚ :=
  (∑ x in s.to_finset, x) / s.to_finset.card

noncomputable def variance (s : Set ℤ) : ℚ :=
  let μ := mean s
  (∑ x in s.to_finset, (x : ℚ) ^ 2) / s.to_finset.card - μ ^ 2

theorem mean_and_variance_unchanged :
  mean initial_set = 0 ∧ variance initial_set = 10 ∧
  (mean replaced_set_1 = 0 ∧ variance replaced_set_1 = 10 ∨
   mean replaced_set_2 = 0 ∧ variance replaced_set_2 = 10) := by
  sorry

end mean_and_variance_unchanged_l716_716395


namespace decagon_has_35_diagonals_l716_716602

-- Definitions based on the conditions from the problem
def decagon (D : Type) [fintype D] [decidable_eq D] : Prop :=
  ∃ (sides : finset D), sides.card = 10

def right_angles_count (D : Type) [fintype D] (P : D → Prop) : Prop :=
  finset.filter P (fintype.elems D) = 2

def specific_angles_count (D : Type) [fintype D] (Q : D → Prop) : Prop :=
  finset.filter Q (fintype.elems D) = 3

-- Define our proposition with all conditions
noncomputable def decagon_diagonals (D : Type) [fintype D] [decidable_eq D]
  (P : D → Prop) (Q : D → Prop) : Prop :=
  decagon D ∧ right_angles_count D P ∧ specific_angles_count D Q → 
  (∃ (n : ℕ), n = 35)

-- The goal is to prove this proposition
theorem decagon_has_35_diagonals (D : Type) [fintype D] [decidable_eq D]
  (P : D → Prop) (Q : D → Prop) :
  decagon_diagonals D P Q :=
by
  sorry

end decagon_has_35_diagonals_l716_716602


namespace count_valid_n_l716_716665

theorem count_valid_n :
  ∃ (count : ℕ), count = 9 ∧ 
  (∀ (n : ℕ), 0 < n ∧ n ≤ 2000 ∧ ∃ (k : ℕ), 21 * n = k * k ↔ count = 9) :=
by
  sorry

end count_valid_n_l716_716665


namespace a2022_value_l716_716001

theorem a2022_value 
  (a : Fin 2022 → ℤ)
  (h : ∀ n k : Fin 2022, a n - a k ≥ n.1^3 - k.1^3)
  (a1011 : a 1010 = 0) :
  a 2021 = 2022^3 - 1011^3 :=
by
  sorry

end a2022_value_l716_716001


namespace regular_polygon_sides_l716_716169

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716169


namespace regular_polygon_sides_l716_716192

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716192


namespace sum_of_squares_l716_716300

theorem sum_of_squares (k : ℕ) (hk : 0 < k) (n : ℕ) (hn : n ≥ 20 * k) :
  ∃ (S : Finset ℕ), S.card = n - k ∧ (∑ i in S, i^2) = n * (n + 1) * (2 * n + 1) / 6 := sorry

end sum_of_squares_l716_716300


namespace regular_polygon_sides_l716_716075

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716075


namespace relatively_prime_days_in_month_l716_716252

theorem relatively_prime_days_in_month :
  ∀ (days_in_month : ℕ) (month_number : ℕ), days_in_month = 31 → month_number = 9 →
  (finset.card (finset.filter (λ d, nat.gcd d month_number = 1) (finset.range (days_in_month + 1))) = 21) :=
by
  intros days_in_month month_number
  sorry

end relatively_prime_days_in_month_l716_716252


namespace difference_one_fourth_one_fifth_l716_716825

theorem difference_one_fourth_one_fifth 
  (N : ℝ)
  (h1 : N = 24.000000000000004)
  (h2 : (1 / 4) * N = 6.000000000000001)
  (h3 : (1 / 5) * (N + 1) = 5.000000000000001) :
   (1 / 4) * N - (1 / 5) * (N + 1) = 1.000000000000000 :=
by
  rw [h1, h2, h3]
  sorry

end difference_one_fourth_one_fifth_l716_716825


namespace regular_polygon_sides_l716_716108

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716108


namespace a_n_formula_b_n_formula_T_n_formula_l716_716693

noncomputable def S (n : ℕ) : ℕ := n^2 - n
def a (n : ℕ) : ℕ :=
  if n = 1 then 0
  else 2 * (n - 1)

def b (n : ℕ) : ℕ := 2^(n-1)
def c (n : ℕ) : ℕ := (n - 1) * 2^n

def T (n : ℕ) : ℕ :=
  (n-2) * 2^(n+1) + 4

theorem a_n_formula (n : ℕ) (hn : n ≥ 2) : a n = 2 * (n - 1) := sorry
theorem b_n_formula (n : ℕ) : b n = 2^(n-1) := sorry
theorem T_n_formula (n : ℕ) : ∑ i in Finset.range n, c (i + 1) = T n := sorry

end a_n_formula_b_n_formula_T_n_formula_l716_716693


namespace unique_x_such_that_f_eq_f_prime_l716_716675

noncomputable def f (x : ℝ) : ℝ := x ^ x

theorem unique_x_such_that_f_eq_f_prime (x : ℝ) (h : x > 0) : f(x) = (D f) x ↔ x = 1 :=
by
  sorry

end unique_x_such_that_f_eq_f_prime_l716_716675


namespace find_n_times_s_l716_716799

noncomputable def g (x : ℝ) : ℝ :=
  if x = 1 then 2011
  else if x = 2 then (1 / 2 + 2010)
  else 0 /- For purposes of the problem -/

theorem find_n_times_s :
  (∀ x y : ℝ, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x + 1 / y + 2010)) →
  ∃ n s : ℝ, n = 1 ∧ s = (4021 / 2) ∧ n * s = 4021 / 2 :=
by
  sorry

end find_n_times_s_l716_716799


namespace cannot_pay_infinite_rent_l716_716537

/--
  Suppose you receive a 1000 franc bill initially.
  - You can exchange any bill for any number of smaller denomination bills.
  - The rent is constant and needs to be paid every month.
  - Assume the requirement to pay rent for an infinite number of months.
  Prove that it is not possible to pay rent every month into infinity given these conditions.
-/
theorem cannot_pay_infinite_rent (initial_bill : ℕ)
  (exchange : ∀ (bill : ℕ), finite (set_of (λ x, x < bill)))
  (rent : ℕ)
  (life_is_infinite : ∀ (n : ℕ), n < ∞) :
  initial_bill = 1000 → ∀ (t : ℕ), t = ∞ → ¬ (∃ (seq : ℕ → ℕ), (∀ (n : ℕ), seq n < initial_bill) ∧ (∀ (n : ℕ), seq n ≥ rent)) :=
begin
  sorry
end

end cannot_pay_infinite_rent_l716_716537


namespace replace_preserve_mean_variance_l716_716400

theorem replace_preserve_mean_variance:
  ∀ (a b c : ℤ), 
    let initial_set := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map (λ x, x : ℤ) in
    let new_set := (initial_set.erase a).++[b, c] in
    let mean (s : List ℤ) := (s.sum : ℚ) / s.length in
    let variance (s : List ℤ) :=
      let m := mean s in
      (s.map (λ x, (x - m) ^ 2)).sum / s.length in
    mean initial_set = 0 ∧ variance initial_set = 10 ∧
    ((mean new_set = 0 ∧ variance new_set = 10) ↔ ((a = -4 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 5))) :=
sorry

end replace_preserve_mean_variance_l716_716400


namespace regular_polygon_sides_l716_716096

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716096


namespace inequality_for_positive_reals_l716_716280

theorem inequality_for_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
    ≥ (Real.sqrt (3 / 2) * Real.sqrt (x + y + z)) := 
sorry

end inequality_for_positive_reals_l716_716280


namespace num_valid_n_is_12_l716_716357

noncomputable def count_valid_n : ℕ :=
  let valid_ks := filter (λ k, (k % 4 = 3) ∧ (2 * k + 1 < 100)) (list.range 50)
  in valid_ks.length

theorem num_valid_n_is_12 :
  count_valid_n = 12 := sorry

end num_valid_n_is_12_l716_716357


namespace number_of_ones_minus_zeros_in_253_binary_l716_716984

def binary_representation (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | value => (binary_representation (value / 2)) * 10 + (value % 2)

-- The binary representation of 253 is 11111101
theorem number_of_ones_minus_zeros_in_253_binary : 
  let x := 1
  let y := 7
  y - x = 6 :=
by
  -- Definitions for x, y
  let x := 1
  let y := 7
  -- Required proof
  sorry

end number_of_ones_minus_zeros_in_253_binary_l716_716984


namespace range_of_x0_l716_716314

theorem range_of_x0
  (x0 y0 : ℝ)
  (h_curve : (x0^2 / 4) + y0^2 = 1)
  (h_dot_product : let MF1 := λ x0 y0, (-sqrt 3 - x0, -y0); 
                   let MF2 := λ x0 y0, (sqrt 3 - x0, -y0); 
                   inner (MF1 x0 y0) (MF2 x0 y0) < 0) :
  (- (2 * sqrt 6) / 3 < x0) ∧ (x0 < (2 * sqrt 6) / 3) :=
sorry

end range_of_x0_l716_716314


namespace regular_polygon_sides_l716_716190

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716190


namespace probability_demand_le_300_probability_profit_gt_0_l716_716591

-- Define the conditions from the problem
def purchase_cost_per_bottle : ℕ := 4
def selling_price_per_bottle : ℕ := 6
def reduced_price_per_bottle : ℕ := 2

def daily_demand (temperature : ℕ) : ℕ :=
  if temperature >= 25 then 500
  else if temperature >= 20 then 300
  else 200

def frequency_distribution : list (ℕ × ℕ) :=
  [(10, 2), (15, 16), (20, 36), (25, 25), (30, 7), (35, 4)]

-- Question 1: Prove that the probability of daily demand not exceeding 300 bottles is 3/5.
theorem probability_demand_le_300 : 
  let days_10_20 := 2 + 16 in
  let days_20_25 := 36 in
  let total_days := 2 + 16 + 36 + 25 + 7 + 4 in
  (days_10_20 + days_20_25) / total_days = (3 : ℚ) / 5 := 
by
  sorry

-- Question 2: Prove all possible values of $Y$ and the probability of $Y > 0$ being 4/5
theorem probability_profit_gt_0 : 
  let total_days := 2 + 16 + 36 + 25 + 7 + 4 in
  let temperature_gt_20_days := total_days - (2 + 16) in
  let values_of_Y := [900, 300, -100] in
  temperature_gt_20_days / total_days = (4 : ℚ) / 5 := 
by
  sorry

end probability_demand_le_300_probability_profit_gt_0_l716_716591


namespace regular_polygon_sides_l716_716071

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716071


namespace sufficient_condition_implication_l716_716699

theorem sufficient_condition_implication {A B : Prop}
  (h : (¬A → ¬B) ∧ (B → A)): (B → A) ∧ (A → ¬¬A ∧ ¬A → ¬B) :=
by
  -- Note: We would provide the proof here normally, but we skip it for now.
  sorry

end sufficient_condition_implication_l716_716699


namespace regular_polygon_sides_l716_716098

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l716_716098


namespace bus_trip_distance_l716_716053

noncomputable def distance_of_bus_trip (v : ℕ) (D : ℕ) : Prop :=
  (D / 40 - D / 45 = 1) ∧ (v = 40)

theorem bus_trip_distance (D : ℕ) : distance_of_bus_trip 40 D → D = 360 :=
by
  intro h
  cases h with h1 h2
  sorry

end bus_trip_distance_l716_716053


namespace calculate_M_l716_716796

-- Define the universal set as a constant with 14 elements
def universal_set := {1,2,3,4,5,6,7,8,9,10,11,12,13,14}

-- Define the nonempty sets A and B that partition the universal set
def valid_partition (A B : set ℕ) : Prop :=
  A ∪ B = universal_set ∧ A ∩ B = ∅ ∧
  (|A| ≠ 7) ∧ (|B| ≠ 7) ∧ (14 - |A| = |B|) ∧
  (|A| ≠ ∅) ∧ (|A| ≠ universal_set) ∧ (|B| ≠ ∅) ∧ (|B| ≠ universal_set) ∧
  (|A| ∉ A) ∧ (|B| ∉ B)

-- The statement of the theorem to be proved
theorem calculate_M : ∃ M, M = 3172 ∧
  ∀ A B, valid_partition A B → M = (count_valid_partitions A B) :=
sorry

-- Function to count valid partitions, supposed to be defined according to the conditions.
noncomputable def count_valid_partitions (A B : set ℕ) : ℕ :=
  sorry

end calculate_M_l716_716796


namespace regular_polygon_sides_l716_716161

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716161


namespace labor_cost_calculation_l716_716381

def num_men : Nat := 5
def num_women : Nat := 8
def num_boys : Nat := 10

def base_wage_man : Nat := 100
def base_wage_woman : Nat := 80
def base_wage_boy : Nat := 50

def efficiency_man_woman_ratio : Nat := 2
def efficiency_man_boy_ratio : Nat := 3

def overtime_rate_multiplier : Nat := 3 / 2 -- 1.5 as a ratio
def holiday_rate_multiplier : Nat := 2

def num_men_working_overtime : Nat := 3
def hours_worked_overtime : Nat := 10
def regular_workday_hours : Nat := 8

def is_holiday : Bool := true

theorem labor_cost_calculation : 
  (num_men * base_wage_man * holiday_rate_multiplier
    + num_women * base_wage_woman * holiday_rate_multiplier
    + num_boys * base_wage_boy * holiday_rate_multiplier
    + num_men_working_overtime * (hours_worked_overtime - regular_workday_hours) * (base_wage_man * overtime_rate_multiplier)) 
  = 4180 :=
by
  sorry

end labor_cost_calculation_l716_716381


namespace div_equal_octagons_l716_716774

-- Definitions based on the conditions
def squareArea (n : ℕ) := n * n
def isDivisor (m n : ℕ) := n % m = 0

-- Main statement
theorem div_equal_octagons (n : ℕ) (hn : n = 8) :
  (2 ∣ squareArea n) ∨ (4 ∣ squareArea n) ∨ (8 ∣ squareArea n) ∨ (16 ∣ squareArea n) :=
by
  -- We shall show the divisibility aspect later.
  sorry

end div_equal_octagons_l716_716774


namespace valid_alpha_range_l716_716289

noncomputable def sqrt (x : ℝ) : ℝ := sorry
noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem valid_alpha_range (α : ℝ) :
  (sqrt(1 + sin(2 * α)) = sin(α) + cos(α)) ↔ (-π/4 < α ∧ α < 3 * π/4) :=
sorry

end valid_alpha_range_l716_716289


namespace count_leftmost_seven_l716_716798

noncomputable def T : Set ℕ := {k : ℕ | ∃ k, 0 ≤ k ∧ k ≤ 3000 ∧ nat.digits 7 k = 7}

-- Given Conditions:
def condition1 : ∀ k, k ∈ T → 0 ≤ k ∧ k ≤ 3000 := sorry
def condition2 : nat.digits 7 3000 = 2838 := sorry
def condition3 : (nat.digits 7 3000).head = 7 := sorry

-- To Prove:
theorem count_leftmost_seven : (T.count (λ k, (nat.digits 7 k).head = 7)) = 163 := sorry

end count_leftmost_seven_l716_716798


namespace system_solution_l716_716002

theorem system_solution (x y : ℝ) :
  (x + y = 4) ∧ (2 * x - y = 2) → x = 2 ∧ y = 2 := by 
sorry

end system_solution_l716_716002


namespace hamburgers_sold_in_winter_l716_716910

theorem hamburgers_sold_in_winter:
  ∀ (T x : ℕ), 
  (T = 5 * 4) → 
  (5 + 6 + 4 + x = T) →
  (x = 5) :=
by
  intros T x hT hTotal
  sorry

end hamburgers_sold_in_winter_l716_716910


namespace equation_of_tangent_line_and_explicit_formula_of_g_max_value_of_h_l716_716332

noncomputable def f (x : ℝ) := Real.log x

noncomputable def g (x : ℝ) := (1/3) * x^3 + (1/2) * x^2 - x + 1/6

def tangent_point : ℝ × ℝ := (1, 0)

-- Equation of the tangent line l at the point (1, 0) for the functions f and g
-- Explicit formula for g(x)
theorem equation_of_tangent_line_and_explicit_formula_of_g :
  ∃ (l : ℝ → ℝ), (∀ x, l x = x - 1) ∧ (∀ x, g x = (1/3) * x^3 + (1/2) * x^2 - x + 1/6) := sorry

noncomputable def h (x : ℝ) := f x - g' x

-- Maximum value of the function h(x)
theorem max_value_of_h :
  ∃ x_max h_max, x_max = 1/2 ∧ h x_max = Real.log (1/2) - 1/4 := sorry

end equation_of_tangent_line_and_explicit_formula_of_g_max_value_of_h_l716_716332


namespace solution_set_inequality_l716_716457

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_eqn : ∀ x : ℝ, f (-x) + f x = x^2
axiom f_prime_bound : ∀ x : ℝ, x < 0 → f' x < x

-- Question and proof problem:
theorem solution_set_inequality : 
  {x : ℝ | f x + 1/2 ≥ f (1 - x) + x} = set.Iic (1/2) :=
sorry

end solution_set_inequality_l716_716457


namespace decreasing_function_condition_l716_716742

theorem decreasing_function_condition (a b : ℝ) :
  (∀ x ∈ Iic (0 : ℝ), deriv (λ x, x^2 + |x - a| + b) x ≤ 0) → a ≥ 0 :=
sorry

end decreasing_function_condition_l716_716742


namespace geometric_sequence_a3_value_l716_716299

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem geometric_sequence_a3_value :
  ∃ a : ℕ → ℝ, ∃ r : ℝ,
  geometric_seq a r ∧
  a 1 = 2 ∧
  (a 3) * (a 5) = 4 * (a 6)^2 →
  a 3 = 1 :=
sorry

end geometric_sequence_a3_value_l716_716299


namespace regular_polygon_sides_l716_716162

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716162


namespace max_digits_l716_716063

def satisfies_conditions (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ {1, 2, 3, 4, 6, 7, 8, 9} → (d ≠ 0) → (d ∈ digits 10 n) → (d ∣ n)

theorem max_digits (n : ℕ) (h : satisfies_conditions n) : 
  ¬ (5 ∈ digits 10 n) → (nat.card (digits 10 n)) ≤ 8 :=
sorry

end max_digits_l716_716063


namespace ratio_side_length_to_perimeter_l716_716939

theorem ratio_side_length_to_perimeter (s : ℝ) (hs : s = 15) : (s : ℝ) / (4 * s) = 1 / 4 :=
by simp [hs]; field_simp; norm_num

-- Placeholder to ensure the code compiles without providing the actual proof.
example : ratio_side_length_to_perimeter 15 rfl := by sorry

end ratio_side_length_to_perimeter_l716_716939


namespace quad_roots_example_l716_716733

noncomputable def roots_of_quad (a : ℝ) : (ℝ × ℝ) :=
let Δ := (1 : ℝ) - 4 * a in
((1 + real.sqrt Δ) / 2, (1 - real.sqrt Δ) / 2)

theorem quad_roots_example (a x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x1 ^ 2 - x1 + a = 0) (h3 : x2 ^ 2 - x2 + a = 0) :
  abs(x1^2 - x2^2) = 1 ↔ abs(x1^3 - x2^3) = 1 :=
by {
  sorry
}

end quad_roots_example_l716_716733


namespace coffee_bean_price_l716_716924

theorem coffee_bean_price 
  (x : ℝ)
  (price_second : ℝ) (weight_first weight_second : ℝ)
  (total_weight : ℝ) (price_mixture : ℝ) 
  (value_mixture : ℝ) 
  (h1 : price_second = 12)
  (h2 : weight_first = 25)
  (h3 : weight_second = 25)
  (h4 : total_weight = 100)
  (h5 : price_mixture = 11.25)
  (h6 : value_mixture = total_weight * price_mixture)
  (h7 : weight_first + weight_second = total_weight) :
  25 * x + 25 * 12 = 100 * 11.25 → x = 33 :=
by
  intro h
  sorry

end coffee_bean_price_l716_716924


namespace tangent_point_condition_l716_716336

open Function

def f (x : ℝ) : ℝ := x^3 - 3 * x
def tangent_line (s : ℝ) (x t : ℝ) : ℝ := (3 * s^2 - 3) * (x - 2) + s^3 - 3 * s

theorem tangent_point_condition (t : ℝ) (h_tangent : ∃s : ℝ, tangent_line s 2 t = t) 
  (h_not_on_curve : ∀ s, (2, t) ≠ (s, f s)) : t = -6 :=
by
  sorry

end tangent_point_condition_l716_716336


namespace number_of_customers_l716_716580

theorem number_of_customers (offices_sandwiches : Nat)
                            (group_per_person_sandwiches : Nat)
                            (total_sandwiches : Nat)
                            (half_group : Nat) :
  (offices_sandwiches = 3 * 10) →
  (total_sandwiches = 54) →
  (half_group * group_per_person_sandwiches = total_sandwiches - offices_sandwiches) →
  (2 * half_group = 12) := 
by
  sorry

end number_of_customers_l716_716580


namespace smallest_n_with_10_divisors_l716_716863

def has_exactly_10_divisors (n : ℕ) : Prop :=
  let divisors : ℕ → ℕ := λ n, (n.divisors).card;
  n.divisors.count = 10

theorem smallest_n_with_10_divisors : ∃ n : ℕ, has_exactly_10_divisors n ∧ ∀ m : ℕ, has_exactly_10_divisors m → n ≤ m :=
begin
  use 48,
  split,
  { 
    -- proof that 48 has exactly 10 divisors
    sorry 
  },
  {
    -- proof that 48 is the smallest such number
    sorry
  }

end smallest_n_with_10_divisors_l716_716863


namespace basketball_team_score_l716_716915

theorem basketball_team_score (n m k : ℕ) (h_team_size : n = 12) (h_min_points : m = 7) 
                              (h_max_points : k = 23) (scores : Fin n → ℕ)
                              (h_scores_min : ∀ i, scores i ≥ m)
                              (h_scores_max : ∀ i, scores i ≤ k) :
                              (∑ i, scores i) ≥ 84 := 
by
  sorry

end basketball_team_score_l716_716915


namespace inequality_solution_l716_716860

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (x^2 > x^(1 / 2)) ↔ (x > 1) :=
by
  sorry

end inequality_solution_l716_716860


namespace particle_intersects_sphere_l716_716385

-- Define the points
def pointA := (1, 2, 2 : ℝ × ℝ × ℝ)
def pointB := (-2, -2, -1 : ℝ × ℝ × ℝ)

-- Define the radius and center of the sphere
def sphere_radius : ℝ := 2
def sphere_center := (0, 0, 0 : ℝ × ℝ × ℝ)

-- Define the line parameterization function
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  ( 1 - 3 * t, 2 - 4 * t, 2 - 3 * t )

-- The quadratic equation derived from intersection
def intersection_quadratic (t : ℝ) : ℝ :=
  29 * t ^ 2 - 26 * t + 5

-- Define the solution here as a tautology for now, it has to be actually proved
theorem particle_intersects_sphere :
  ∃ t1 t2 : ℝ, intersection_quadratic t1 = 0 ∧ intersection_quadratic t2 = 0 ∧
  t1 ≠ t2 ∧
  let d := (t1 - t2).abs in
  let dist := Real.sqrt(34) * d in
  dist = 24 * Real.sqrt(119) / 29 ∧
  24 + 119 = 143 :=
by
  sorry

end particle_intersects_sphere_l716_716385


namespace probability_divisible_by_11_l716_716737

/-- 
  Given a set of five-digit numbers such that the sum of its digits is 43, 
  the probability that a randomly chosen number from this set is divisible by 11 is 1/5.
-/

theorem probability_divisible_by_11 (s : Finset ℕ) (h1 : ∀ n ∈ s, (10^4 ≤ n ∧ n < 10^5)) 
  (h2 : ∀ n ∈ s, (n.digits 10).sum = 43) : 
  (s.filter (λ n, n % 11 = 0)).card / s.card = 1 / 5 :=
sorry

end probability_divisible_by_11_l716_716737


namespace downstream_speed_l716_716927

-- Define the given conditions as constants
def V_u : ℝ := 25 -- upstream speed in kmph
def V_m : ℝ := 40 -- speed of the man in still water in kmph

-- Define the speed of the stream
def V_s := V_m - V_u

-- Define the downstream speed
def V_d := V_m + V_s

-- Assertion we need to prove
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l716_716927


namespace Moles_theorem_l716_716828

theorem Moles_theorem 
  (A B C D E F : Type)
  (triangle_ABC: triangle A B C)
  (inside_D: inside_triangle D A B C)
  (inside_E: inside_triangle E A B C)
  (inside_F: inside_triangle F A B C)
  (h1 : angle D B C = angle F B A = (1 / 3 : ℝ) * angle A B C)
  (h2 : angle F A B = angle E A C = (1 / 3 : ℝ) * angle B A C)
  (h3 : angle E C A = angle D C B = (1 / 2 : ℝ) * angle A C B) 
  : is_equilateral_triangle D E F :=
sorry

end Moles_theorem_l716_716828


namespace seq_compare_l716_716513

def seq_x (x : ℕ → ℝ) : Prop :=
∀ n, x (n + 2) = x n + (x (n + 1))^2

def seq_y (y : ℕ → ℝ) : Prop :=
∀ n, y (n + 2) = (y n)^2 + y (n + 1)

variables {x y : ℕ → ℝ}

axiom init_cond (hx1 : 1 < x 1) (hx2 : 1 < x 2) (hy1 : 1 < y 1) (hy2 : 1 < y 2) : Prop

theorem seq_compare (hx : seq_x x) (hy : seq_y y) (h : init_cond) : ∃ n, x n > y n := by
  sorry

end seq_compare_l716_716513


namespace a4_value_a_n_formula_l716_716383

theorem a4_value : a_4 = 30 := 
by
    sorry

noncomputable def a_n (n : ℕ) : ℕ :=
    (n * (n + 1)^2 * (2 * n + 1)) / 6

theorem a_n_formula (n : ℕ) : a_n n = (n * (n + 1)^2 * (2 * n + 1)) / 6 := 
by
    sorry

end a4_value_a_n_formula_l716_716383


namespace xyz_inequality_l716_716458

theorem xyz_inequality (x y z : ℝ) (h : x + y + z = 0) : 
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := 
by sorry

end xyz_inequality_l716_716458


namespace sum_of_digits_greatest_prime_divisor_of_65535_l716_716501

theorem sum_of_digits_greatest_prime_divisor_of_65535 :
  let n := 65536
  let m := n - 1
  let greatest_prime_divisor := if Prime (2 ^ 8 + 1) then (2 ^ 8 + 1) else 2 ^ 8 - 1
  ∃ (p : ℕ), p = greatest_prime_divisor ∧ ∑ d in (nat.digits 10 p), d = 14 := 
by 
  let n := 65536
  let m := n - 1
  let greatest_prime_divisor := 257
  use greatest_prime_divisor
  have h_prime : Prime greatest_prime_divisor := by 
    sorry -- Proof of primality of 257
  have h_div : m % greatest_prime_divisor = 0 := by
    sorry -- Proof of divisibility
  have h_sum_digits : ∑ d in (nat.digits 10 greatest_prime_divisor), d = 14 := by
    sorry -- Calculation of the sum of digits
  exact ⟨rfl, h_sum_digits⟩

end sum_of_digits_greatest_prime_divisor_of_65535_l716_716501


namespace regular_polygon_sides_l716_716220

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716220


namespace sum_of_middle_three_cards_l716_716835

theorem sum_of_middle_three_cards :
  let red_cards := [2, 3, 4, 5, 6]
  let blue_cards := [4, 5, 6, 7, 8]
  let stack := [2, 4, 4, 8, 6, 6, 3, 5, 5]
  let cards := stack.take 9
  (∀ (i: ℕ), i < 9 → i % 2 = 0 ∨ i % 2 = 1 →
    (i % 2 = 0 → red_cards.contains cards[i]) ∧ (i % 2 = 1 → blue_cards.contains cards[i])) →
  (∀ (i: ℕ), i < 8 → i % 2 = 0 → cards[i] % cards[i + 1] = 0 ∨ cards[i] % cards[i - 1] = 0) →
  cards[3] + cards[4] + cards[5] = 14 :=
by
  sorry

end sum_of_middle_three_cards_l716_716835


namespace hyperbola_asymptotes_l716_716758

theorem hyperbola_asymptotes
  (b : ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ)
  (h_parabola : A.2^2 = 4 * A.1)
  (h_focus : F = (1, 0))
  (h_focus_dist : (A.1 - F.1)^2 + A.2^2 = 25)
  (h_intersection : A.1^2 / 4 - A.2^2 / b^2 = 1)
  (h_b : b = 4 * real.sqrt 3 / 3) :
  ∃ k : ℝ, ∀ (x : ℝ), y = k * x → k = 2 * real.sqrt 3 / 3 ∨ k = - (2 * real.sqrt 3 / 3) :=
by
  sorry

end hyperbola_asymptotes_l716_716758


namespace compute_AB_length_l716_716790

theorem compute_AB_length (AC BC : ℝ) (angle_ABC angle_ACB : ℝ) (hAC : AC = 28) 
  (hBC : BC = 33) (h_angle : angle_ABC = 2 * angle_ACB) : ∃ (AB: ℝ), AB = 16 :=
by
  use 16
  sorry

end compute_AB_length_l716_716790


namespace complex_numbers_satisfying_conditions_l716_716646

theorem complex_numbers_satisfying_conditions (z : ℂ) (hz1 : abs z = 1) 
  (hz2 : abs ((z ^ 2 / conj(z) ^ 2) + (conj(z) ^ 2 / z ^ 2)) = 1) : 
  ∃ n : ℕ, n = 16 :=
by
  sorry

end complex_numbers_satisfying_conditions_l716_716646


namespace ellipse_equation_dot_product_range_l716_716329

-- Given conditions
def ellipse := {a b : ℝ // a > 0 ∧ b > 0 ∧ a > b}
def givenEccentricity (e : ℝ) := e = 1/2
def tangentLine (l : ℝ) := l = sqrt 6
def lineThroughPoint (x y : ℝ) := x = 4 ∧ y = 0

-- Part 1: Equation of the ellipse
theorem ellipse_equation 
  (a b : ℝ) (h_ellipse : ellipse) 
  (h_eccentricity : givenEccentricity (real.sqrt (1 - b^2 / a^2))) 
  (h_tangent : tangentLine (sqrt (1 + 1) * b)) :
  (a = 2) ∧ (b = sqrt 3) → ∀ x y, x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- Part 2: Range of values for dot product
theorem dot_product_range
  (a b : ℝ) (h_ellipse : ellipse) 
  (h_eccentricity : givenEccentricity (real.sqrt (1 - b^2 / a^2))) 
  (h_tangent : tangentLine (sqrt (1 + 1) * b))
  (x y k : ℝ) (h_point : lineThroughPoint x y) 
  (h_not_perpendicular : k ≠ 0):
  ∃ (range : set ℝ), range = Ico (-4) (13/4) :=
sorry

end ellipse_equation_dot_product_range_l716_716329


namespace train_pass_bridge_in_50_seconds_l716_716612

noncomputable def time_to_pass_bridge (length_train length_bridge : ℕ) (speed_kmh : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge 485 140 45 = 50 :=
by
  sorry

end train_pass_bridge_in_50_seconds_l716_716612


namespace sin_D_value_l716_716387

theorem sin_D_value (D E F : Type) [Triangle DEF] (h1 : right_angle DEF E)
  (h2 : 5 * real.sin D = 12 * real.cos D) : real.sin D = 12 / 13 := 
sorry

end sin_D_value_l716_716387


namespace eq_circle_and_tangents_l716_716691

theorem eq_circle_and_tangents {a b : ℝ} :
  (∃ (x y : ℝ), x + y + 1 = 0 ∧ ((x - a) ^ 2 + (y - b) ^ 2 = 25) ∧ ((-2 - a) ^ 2 + b ^ 2 = 25) ∧ ((5 - a) ^ 2 + (1 - b) ^ 2 = 25)) :=
    ((∃ (x y : ℝ), (x = 2 ∧ y = -3) ∧ ((x - a) ^ 2 + (y + 3) ^ 2 = 25)) ∧ ((x = -3) ∨ (y = 8 / 15 * (x + 3)))) :=
begin
  sorry
end

end eq_circle_and_tangents_l716_716691


namespace regular_polygon_sides_l716_716115

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716115


namespace julia_work_completion_l716_716979

theorem julia_work_completion (total_time : ℕ) (initial_days : ℕ) (initial_people : ℕ) (initial_fraction_done : ℚ) (job_fraction_remaining : ℚ) (rate_needed : ℚ) (required_people : ℕ):
  total_time = 40 → initial_days = 10 → initial_people = 10 → initial_fraction_done = 1 / 4 → job_fraction_remaining = 3 / 4 → 
  rate_needed = (3 / 4) / 30 → required_people = 10 :=
begin
  intros h_total_time h_initial_days h_initial_people h_initial_fraction_done h_job_fraction_remaining h_rate_needed,
  -- We will prove the necessary steps here to show required_people = 10
  sorry
end

end julia_work_completion_l716_716979


namespace part_1_part_2_l716_716323

variable {f : ℝ → ℝ}
variable {M : ℝ}
variable {x y : ℝ}

-- Conditions
def condition_1 (f : ℝ → ℝ) := ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → 
  f(x) * f(y) ≤ y^2 * f(x / 2) + x^2 * f(y / 2)

def condition_2 (f : ℝ → ℝ) (M : ℝ) := M > 0 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f(x)| ≤ M

-- Statements to Prove
theorem part_1 (hf1 : condition_1 f) (hf2 : condition_2 f M) : ∀ (x : ℝ), 0 ≤ x → f(x) ≤ x^2 := 
by
  intros
  sorry

theorem part_2 (hf1 : condition_1 f) (hf2 : condition_2 f M) : ∀ (x : ℝ), 0 ≤ x → f(x) ≤ (x^2 / 2) :=
by
  intros
  sorry

end part_1_part_2_l716_716323


namespace number_of_customers_l716_716582

theorem number_of_customers (total_sandwiches : ℕ) (office_orders : ℕ) (customers_half : ℕ) (num_offices : ℕ) (num_sandwiches_per_office : ℕ) 
  (sandwiches_per_customer : ℕ) (group_sandwiches : ℕ) (total_customers : ℕ) :
  total_sandwiches = 54 →
  num_offices = 3 →
  num_sandwiches_per_office = 10 →
  group_sandwiches = total_sandwiches - num_offices * num_sandwiches_per_office →
  customers_half * sandwiches_per_customer = group_sandwiches →
  sandwiches_per_customer = 4 →
  customers_half = total_customers / 2 →
  total_customers = 12 :=
by
  intros
  sorry

end number_of_customers_l716_716582


namespace suff_not_nec_condition_l716_716702

/-- f is an even function --/
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Condition x1 + x2 = 0 --/
def sum_eq_zero (x1 x2 : ℝ) : Prop := x1 + x2 = 0

/-- Prove: sufficient but not necessary condition --/
theorem suff_not_nec_condition (f : ℝ → ℝ) (h_even : is_even f) (x1 x2 : ℝ) :
  sum_eq_zero x1 x2 → f x1 - f x2 = 0 ∧ (f x1 - f x2 = 0 → ¬ sum_eq_zero x1 x2) :=
by
  sorry

end suff_not_nec_condition_l716_716702


namespace sum_of_possible_values_l716_716804

variable (a b c d : ℝ)

theorem sum_of_possible_values
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) :
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 :=
sorry

end sum_of_possible_values_l716_716804


namespace regular_polygon_sides_l716_716216

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716216


namespace length_within_cube_edge_4_l716_716231

-- Define the coordinates of the points
def point1 : ℝ × ℝ × ℝ := (0, 0, 0)
def point2 : ℝ × ℝ × ℝ := (5, 5, 14)

-- Define the cube with edge length 4
def cube_edge_length_4_min : ℝ × ℝ × ℝ := (0.5, 0.5, 5)
def cube_edge_length_4_max : ℝ × ℝ × ℝ := (4.5, 4.5, 9)

-- Define the parametric equation for the line segment XY
def parametric_point (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := point1
  let (x2, y2, z2) := point2
  (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1))

-- Define the intersection points
def intersection1 : ℝ × ℝ × ℝ := (1, 1, 5)
def intersection2 : ℝ × ℝ × ℝ := (2.5, 2.5, 9)

-- Define the distance formula for 3D points
def distance_3d (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2))

-- The theorem we need to prove
theorem length_within_cube_edge_4 : distance_3d intersection1 intersection2 = Real.sqrt 20.5 :=
  sorry

end length_within_cube_edge_4_l716_716231


namespace shortest_distance_polar_coordinates_l716_716765

theorem shortest_distance_polar_coordinates :
  ∃ (B : ℝ × ℝ), 
  B = (Math.sqrt 2, 3 * Real.pi / 4) ∧
  let A := (2, Real.pi / 2) in
  let l := λ (ρ θ : ℝ), ρ * cos θ + ρ * sin θ = 0 in
  (0 ≤ A.2 ∧ A.2 ≤ 2 * Real.pi) ∧
  ∀ (B' : ℝ × ℝ), 
  (B'.1 * cos B'.2 + B'.1 * sin B'.2 = 0) ->
  dist (A.1 * cos A.2, A.1 * sin A.2) 
       (B'.1 * cos B'.2, B'.1 * sin B'.2) ≥ 
  dist (A.1 * cos A.2, A.1 * sin A.2) 
       (B.1 * cos B.2, B.1 * sin B.2) := 
begin
  sorry
end

end shortest_distance_polar_coordinates_l716_716765


namespace number_of_customers_l716_716581

theorem number_of_customers (total_sandwiches : ℕ) (office_orders : ℕ) (customers_half : ℕ) (num_offices : ℕ) (num_sandwiches_per_office : ℕ) 
  (sandwiches_per_customer : ℕ) (group_sandwiches : ℕ) (total_customers : ℕ) :
  total_sandwiches = 54 →
  num_offices = 3 →
  num_sandwiches_per_office = 10 →
  group_sandwiches = total_sandwiches - num_offices * num_sandwiches_per_office →
  customers_half * sandwiches_per_customer = group_sandwiches →
  sandwiches_per_customer = 4 →
  customers_half = total_customers / 2 →
  total_customers = 12 :=
by
  intros
  sorry

end number_of_customers_l716_716581


namespace regular_polygon_sides_l716_716119

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716119


namespace existence_of_lambda_l716_716567

variables {R : Type} [Real R]

-- Define the ellipse with specified conditions
def ellipse (x y : R) : Prop := (x^2 / 4) + (3 * y^2 / 4) = 1

-- Define points A, B, and C
def point_A : R × R := (2, 0)
def point_B : R × R := (-1, -1)
def point_C : R × R := (1, 1)

-- Define vectors
def vector_AC := (1 - 2, 1 - 0)
def vector_BC := (1 + 1, 1 + 1)

-- Condition: AC ⋅ BC = 0 and |BC| = 2|AC|
def conditions : Prop :=
  (vector_AC.1 * vector_BC.1 + vector_AC.2 * vector_BC.2 = 0) ∧
  ((vector_BC.1^2 + vector_BC.2^2) = 2^2 * (vector_AC.1^2 + vector_AC.2^2))

-- Points P and Q lie on the ellipse and the angle bisector of ∠PCQ is perpendicular to AO
variables (P Q : R × R)
def angle_bisector_perpendicular : Prop :=
  let k := (P.snd - point_C.snd) / (P.fst - point_C.fst) in
  let m := (Q.snd - point_C.snd) / (Q.fst - point_C.fst) in
  k * m = -1

-- Prove existence of λ such that PQ = λ AB
theorem existence_of_lambda :
  ∀ {P Q : R × R}, ellipse P.fst P.snd → ellipse Q.fst Q.snd →
    angle_bisector_perpendicular P Q → ∃ λ : R, (Q.fst - P.fst, Q.snd - P.snd) = λ • (point_B.fst - point_A.fst, point_B.snd - point_A.snd) :=
by
  sorry

end existence_of_lambda_l716_716567


namespace problem_conditions_T_n_less_than_1_l716_716305

def sequence_a (n : ℕ) : ℕ → ℝ := sorry -- Define the sequence a_n
def sum_S (n : ℕ) : ℝ := (Finset.range n).sum (sequence_a n)

theorem problem_conditions (n : ℕ) (h₀ : sequence_a 1 = 1) 
  (h₁ : ∀ n, n * sequence_a (n + 1) + sum_S (n + 1) = 0) :
  n * sequence_a (n + 1) + sum_S (n + 1) = 0 := 
sorry

def sequence_b (n : ℕ) : ℕ → ℝ := λ n, sum_S n * sum_S (n + 1)
def sum_T (n : ℕ) : ℝ := (Finset.range n).sum (sequence_b n)

theorem T_n_less_than_1 (n : ℕ) (h_conditions : problem_conditions n) : sum_T n < 1 :=
sorry

end problem_conditions_T_n_less_than_1_l716_716305


namespace tile_difference_proof_l716_716011

theorem tile_difference_proof (initial_blue_tiles : ℕ) (initial_green_tiles : ℕ) (added_green_tiles : ℕ) :
  initial_blue_tiles = 20 → initial_green_tiles = 10 → added_green_tiles = 6 →
  (initial_green_tiles + added_green_tiles) - initial_blue_tiles = -4 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tile_difference_proof_l716_716011


namespace min_value_z_l716_716725

theorem min_value_z (x y z : ℤ) (h1 : x + y + z = 100) (h2 : x < y) (h3 : y < 2 * z) : z ≥ 21 :=
sorry

end min_value_z_l716_716725


namespace remaining_surface_area_l716_716853

def edge_length_original : ℝ := 9
def edge_length_small : ℝ := 2
def surface_area (a : ℝ) : ℝ := 6 * a^2

theorem remaining_surface_area :
  surface_area edge_length_original - 3 * (edge_length_small ^ 2) + 3 * (edge_length_small ^ 2) = 486 :=
by
  sorry

end remaining_surface_area_l716_716853


namespace mean_and_variance_unchanged_l716_716399

noncomputable def initial_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
noncomputable def replaced_set_1 : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 5, -1}
noncomputable def replaced_set_2 : Set ℤ := {-5, -3, -2, -1, 0, 1, 2, 3, 4, -5, 1}

noncomputable def mean (s : Set ℤ) : ℚ :=
  (∑ x in s.to_finset, x) / s.to_finset.card

noncomputable def variance (s : Set ℤ) : ℚ :=
  let μ := mean s
  (∑ x in s.to_finset, (x : ℚ) ^ 2) / s.to_finset.card - μ ^ 2

theorem mean_and_variance_unchanged :
  mean initial_set = 0 ∧ variance initial_set = 10 ∧
  (mean replaced_set_1 = 0 ∧ variance replaced_set_1 = 10 ∨
   mean replaced_set_2 = 0 ∧ variance replaced_set_2 = 10) := by
  sorry

end mean_and_variance_unchanged_l716_716399


namespace jill_arrives_15_minutes_before_jack_l716_716777

theorem jill_arrives_15_minutes_before_jack
  (distance : ℝ) (jill_speed : ℝ) (jack_speed : ℝ) (start_same_time : true)
  (h_distance : distance = 2) (h_jill_speed : jill_speed = 8) (h_jack_speed : jack_speed = 4) :
  (2 / 4 * 60) - (2 / 8 * 60) = 15 :=
by
  sorry

end jill_arrives_15_minutes_before_jack_l716_716777


namespace ratio_side_length_to_perimeter_l716_716937

theorem ratio_side_length_to_perimeter (side_length : ℝ) (perimeter : ℝ) (h_side : side_length = 15) (h_perimeter : perimeter = 4 * side_length) : side_length / perimeter = 1 / 4 :=
by
  rw [h_side, h_perimeter]
  norm_num
  simp


end ratio_side_length_to_perimeter_l716_716937


namespace bell_ring_1000th_chime_l716_716916

def ring_times : List ℕ :=
  [11] ++ (List.range (12 - 1)).map (λ n, n + 1) ++ (List.range 12).map (λ n, 1)

def total_chimes (times : List ℕ) : ℕ :=
  times.sum

def chimes_in_day : ℕ :=
  total_chimes ring_times

noncomputable def days_needed_for_chimes (n : ℕ) : ℕ :=
  n / chimes_in_day + 1

noncomputable def exact_chime_day (start_day : ℕ) (start_time : ℕ) (n : ℕ) : ℕ :=
  let days_needed := days_needed_for_chimes n
  start_day + days_needed

theorem bell_ring_1000th_chime :
  exact_chime_day 1 10 1000 = 11 := sorry

end bell_ring_1000th_chime_l716_716916


namespace regular_polygon_sides_l716_716219

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716219


namespace committee_combinations_l716_716592

-- We use a broader import to ensure all necessary libraries are included.
-- Definitions and theorem

def club_member_count : ℕ := 20
def committee_member_count : ℕ := 3

theorem committee_combinations : 
  (Nat.choose club_member_count committee_member_count) = 1140 := by
sorry

end committee_combinations_l716_716592


namespace polynomial_remainder_l716_716284

theorem polynomial_remainder (x : ℂ) : (x^1500) % (x^3 - 1) = 1 := 
sorry

end polynomial_remainder_l716_716284


namespace replace_preserve_mean_variance_l716_716401

theorem replace_preserve_mean_variance:
  ∀ (a b c : ℤ), 
    let initial_set := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map (λ x, x : ℤ) in
    let new_set := (initial_set.erase a).++[b, c] in
    let mean (s : List ℤ) := (s.sum : ℚ) / s.length in
    let variance (s : List ℤ) :=
      let m := mean s in
      (s.map (λ x, (x - m) ^ 2)).sum / s.length in
    mean initial_set = 0 ∧ variance initial_set = 10 ∧
    ((mean new_set = 0 ∧ variance new_set = 10) ↔ ((a = -4 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 5))) :=
sorry

end replace_preserve_mean_variance_l716_716401


namespace polynomial_one_negative_root_iff_l716_716999

noncomputable def polynomial_has_one_negative_real_root (p : ℝ) : Prop :=
  ∃ (x : ℝ), (x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1 = 0) ∧
  ∀ (y : ℝ), y < x → y^4 + 3*p*y^3 + 6*y^2 + 3*p*y + 1 ≠ 0

theorem polynomial_one_negative_root_iff (p : ℝ) :
  polynomial_has_one_negative_real_root p ↔ p ≥ 4 / 3 :=
sorry

end polynomial_one_negative_root_iff_l716_716999


namespace volume_ratio_l716_716233

-- Define variables for volumes of the first and second containers
variables (A B : ℝ)

-- Given conditions
def container1_initial_state : Prop := A * (2 / 3) = partial_volume_1
def container1_after_adding : Prop := partial_volume_1 + A * (1 / 6) = (5 / 6) * A
def container2_filled : Prop := (5 / 6) * A = (5 / 6) * B

-- The goal is to prove that the volumes are equal given the conditions
theorem volume_ratio (A B : ℝ) (h1 : container1_initial_state A) (h2 : container1_after_adding A) (h3 : container2_filled A B) : A = B :=
begin
  sorry
end

end volume_ratio_l716_716233


namespace basketball_team_minimum_score_l716_716912

theorem basketball_team_minimum_score (n : ℕ) (score : ℕ) :
  (n = 12) →
  (∀ i : fin n, 7 ≤ score) →
  (∀ i : fin n, score i ≤ 23) →
  (84 ≤ (∑ i : fin n, score i)) :=
by sorry

end basketball_team_minimum_score_l716_716912


namespace problem1_problem2_l716_716049

noncomputable def tan_inv_3_value : ℝ := -4 / 5

theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = tan_inv_3_value := 
sorry

noncomputable def f (θ : ℝ) : ℝ := 
  (2 * Real.cos θ ^ 3 + Real.sin (2 * Real.pi - θ) ^ 2 + 
   Real.sin (Real.pi / 2 + θ) - 3) / 
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem problem2 :
  f (Real.pi / 3) = -1 / 2 :=
sorry

end problem1_problem2_l716_716049


namespace customers_left_l716_716615

theorem customers_left (initial_customers : ℝ) (first_left : ℝ) (second_left : ℝ) : initial_customers = 36.0 ∧ first_left = 19.0 ∧ second_left = 14.0 → initial_customers - first_left - second_left = 3.0 :=
by
  intros h
  sorry

end customers_left_l716_716615


namespace Jeff_total_ounces_of_peanut_butter_l716_716426

theorem Jeff_total_ounces_of_peanut_butter
    (jars : ℕ)
    (equal_count : ℕ)
    (total_jars : jars = 9)
    (j16 : equal_count = 3) 
    (j28 : equal_count = 3)
    (j40 : equal_count = 3) :
    (3 * 16 + 3 * 28 + 3 * 40 = 252) :=
by
  sorry

end Jeff_total_ounces_of_peanut_butter_l716_716426


namespace regular_polygon_sides_l716_716203

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716203


namespace isosceles_triangle_BC_length_l716_716755

-- Define terms and conditions
variables {A B C H : Type}
variables {AB AC BC AH HC BH : ℝ}

-- Given constants
def AB := 5
def AC := 5
def AH := 2 * HC
def AC_from_AH_HC := AH + HC
def HC_from_AC := AC / 3

-- Define the altitude conditions
def BH := Real.sqrt (AB^2 - AH^2)
def BC := Real.sqrt (BH^2 + HC^2)

theorem isosceles_triangle_BC_length 
  (AB_eq : AB = 5)
  (AC_eq : AC = 5)
  (AH_eq_2HC : AH = 2 * HC)
  (AC_3HC : AC = 3 * HC)
  : BC = 5 * Real.sqrt 6 / 3 :=
by
  sorry

end isosceles_triangle_BC_length_l716_716755


namespace dwarf_reachability_l716_716468

variables {board : Type} 

noncomputable def is_dwarf_starting_on_odd_reachable : Prop :=
  ∀ (start : (Σ i j, (i, j ∈ finRange 1 6 ∧ odd (i + j))),
    ¬ ∃ path : list (ℕ × ℕ),
      path.head = start ∧ path.last = (6,6) ∧
      (∀ square ∈ path, square ∈ finRange 1 6 × finRange 1 6) ∧
      (∀ i (x : ℕ × ℕ), list.count x path = 1) ∧
      (∀ x y ∈ path, dist x y = 1))

theorem dwarf_reachability (h : is_dwarf_starting_on_odd_reachable) : true := sorry

end dwarf_reachability_l716_716468


namespace F_36_72_max_happy_pair_l716_716677

-- Define F(m, n) as specified in the problem
def F (m n : Int) : Int :=
  let m_tens := m / 10
  let m_units := m % 10
  let n_tens := n / 10
  let n_units := n % 10
  m_tens * n_units + m_units * n_tens

-- Define the conditions on a and b
def is_valid_ab (a b : Int) : Prop :=
  1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 5

-- Define whether m and n are "happy pairs" under the given problem conditions
def is_happy_pair (m n m' : Int) (a b : Int) : Prop :=
  11 ∣ (m' + 5 * (n % 10)) ∧ m = 21 * a + b ∧ n = 53 + b

-- Problem 1: Prove that F(36, 72) = 48
theorem F_36_72 : F 36 72 = 48 :=
by
  sorry

-- Problem 2: Prove the maximum value of all "happy pairs" F(m, n) is 58
theorem max_happy_pair : ∀ (a b : Int), is_valid_ab a b → 
  ∃ (m n m' : Int), is_happy_pair m n m' a b →
  F m n ≤ 58 ∧ 
  (∀ (a' b' : Int), is_valid_ab a' b' → 
  ∃ (m' n' : Int), is_happy_pair m' n' _ a' b' →
  F m' n' ≤ F m n) :=
by
  sorry

end F_36_72_max_happy_pair_l716_716677


namespace solve_triangle_distance_problem_l716_716761

def triangle_distance_problem (A B C F : Type) [distance : A → B → ℝ] : Prop :=
  let AB := distance A B
  let AC := distance A C
  let BC := distance B C
  let AF := distance A F
  let BF := distance B F
  let CF := distance C F
  ∀ (A B C F : ℝ),
  AB = 120 ∧
  AC = 160 ∧
  ∃ BC, BC = real.sqrt (AB^2 + AC^2) ∧  -- Using the Pythagorean theorem
  AF = AB + BF ∧
  AF = AC + CF ∧
  CF = BC - BF ∧
  AF = AC + CF → 
  BF = 80

theorem solve_triangle_distance_problem :
  triangle_distance_problem :=
sorry

end solve_triangle_distance_problem_l716_716761


namespace solve_equation_l716_716483

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
sorry

end solve_equation_l716_716483


namespace regular_polygon_sides_l716_716181

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l716_716181


namespace min_length_tangent_eq_one_l716_716610

theorem min_length_tangent_eq_one :
  ∀ (x y : ℝ), y = x - 1 →
  (∃ (h : x^2 + y^2 - 6 * x + 8 = 0), 
   ∀ (P : ℝ × ℝ), 
   let d := |3 - fst P + snd P - 1| / sqrt 2,
       r := 1,
       min_tangent_length := sqrt (d^2 - r^2)
   in min_tangent_length = 1) :=
by sorry

end min_length_tangent_eq_one_l716_716610


namespace area_of_parallelogram_l716_716037

theorem area_of_parallelogram (base height : ℕ) (h_base : base = 48) (h_height : height = 36) : 
  base * height = 1728 :=
by { rw [h_base, h_height], norm_num, }

end area_of_parallelogram_l716_716037


namespace regular_polygon_sides_l716_716168

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l716_716168


namespace find_base_of_first_term_l716_716739

theorem find_base_of_first_term 
  (a : ℝ) (x : ℝ) (h1 : x = 0.25)
  (h2 : a^(-x) + 25^(-2 * x) + 5^(-4 * x) = 14) :
  a ≈ 34216.6016 :=
by sorry

end find_base_of_first_term_l716_716739


namespace area_of_right_triangle_l716_716514

theorem area_of_right_triangle (a b c area: ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) (h4 : a^2 + b^2 = c^2) :
  area = (1 / 2) * a * b :=
begin
  sorry
end

example : ∃ (area : ℝ), area = 120 := 
  by {
    use 120,
    exact rfl,
  }

end area_of_right_triangle_l716_716514


namespace intervals_of_monotonicity_min_value_diff_l716_716814

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x^2 - a * x

-- Part I: Intervals of monotonicity for a = 3
theorem intervals_of_monotonicity (a : ℝ) (ha : a = 3) :
  (∀ x > 0, deriv (λ x, f x a) x > 0 ↔ (0 < x ∧ x < 1/2) ∨ x > 1) ∧
  (∀ x > 0, deriv (λ x, f x a) x < 0 ↔ 1/2 < x ∧ x < 1) :=
by sorry

-- Part II: Minimum value of f(x₁) - f(x₂) given extreme points x₁ and x₂
theorem min_value_diff (x1 x2 a : ℝ) (hx1 : 0 < x1 ∧ x1 ≤ 1) (ha : a = 3) 
  (h_extreme : deriv (λ x, f x a) x1 = 0 ∧ deriv (λ x, f x a) x2 = 0 ) :
  f x1 a - f x2 a = -3/4 + Real.log 2 :=
by sorry

end intervals_of_monotonicity_min_value_diff_l716_716814


namespace john_total_distance_l716_716427

-- Define the parameters according to the conditions
def daily_distance : ℕ := 1700
def number_of_days : ℕ := 6
def total_distance : ℕ := daily_distance * number_of_days

-- Lean theorem statement to prove the total distance run by John
theorem john_total_distance : total_distance = 10200 := by
  -- Here, the proof would go, but it is omitted as per instructions
  sorry

end john_total_distance_l716_716427


namespace questions_per_tourist_indeterminate_l716_716973

theorem questions_per_tourist_indeterminate (total_questions : ℕ) (tours : ℕ) : 
  total_questions = 68 ∧ tours = 4 → 
  ∀ (tourists_in_each_tour : ℕ → ℕ), 
  (∃ average_questions_per_tourist : ℕ, average_questions_per_tourist = total_questions / (tourists_in_each_tour 0 + tourists_in_each_tour 1 + tourists_in_each_tour 2 + tourists_in_each_tour 3)) → false :=
by
  unfold
  sorry

end questions_per_tourist_indeterminate_l716_716973


namespace parabola_equation_l716_716325

theorem parabola_equation (p : ℝ) (h_pos : p > 0) (M : ℝ) (h_Mx : M = 3) (h_MF : abs (M + p/2) = 2 * p) :
  (forall x y, y^2 = 2 * p * x) -> (forall x y, y^2 = 4 * x) :=
by
  sorry

end parabola_equation_l716_716325


namespace part1_arithmetic_sequence_part2_general_term_part3_max_m_l716_716302

-- Part (1)
theorem part1_arithmetic_sequence (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : a 1 + a 2 = 2 * m) : 
  m = 9 / 8 := 
sorry

-- Part (2)
theorem part2_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2) : 
  ∀ n, a n = 8 ^ (1 - 2 ^ (n - 1)) := 
sorry

-- Part (3)
theorem part3_max_m (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : ∀ n, a n < 4) : 
  m ≤ 2 := 
sorry

end part1_arithmetic_sequence_part2_general_term_part3_max_m_l716_716302


namespace remainder_division_l716_716647

noncomputable def polynomial_remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  let (quot, rem) := Polynomial.div_mod p q in rem

theorem remainder_division
    : polynomial_remainder (x^6 - x^5 - x^4 + x^3 + x^2 - x) ((x^2 - 1) * (x - 2)) 
    = 8.5 * x^2 - 1.5 * x - 6 := 
by 
    sorry

end remainder_division_l716_716647


namespace PQR_PQ_l716_716658

noncomputable def TriangleTransform (A B C P Q R P' Q' R' O : Point) : Prop :=
  ∃ (O : Point), 
    (dist O P = dist O P') ∧
    (dist O Q = dist O Q') ∧
    (dist O R = dist O R')

theorem PQR_PQ'R'_congruent_by_rotation (A B C P Q R P' Q' R' : Point)
    (H1 : dist A P = dist B C) 
    (H2 : dist B Q = dist A B) 
    (H3 : dist C R = dist B C)
    (H4 : dist A P' = dist C B)
    (H5 : dist B Q' = dist A C) 
    (H6 : dist C R' = dist A B) : 
    ∃ O : Point, TriangleTransform A B C P Q R P' Q' R' O := by
  sorry

end PQR_PQ_l716_716658


namespace problem1_l716_716904

noncomputable def problem1_expr (a : ℝ) : ℝ :=
  (1 / 2) * Real.log 2 +
  Real.sqrt ((Real.log (Real.sqrt 2))^2 - Real.log 2 + 1) -
  (Real.sqrt (a^9) * Real.sqrt (a^(-3)))^(1 / 3) /
  ((Real.sqrt (a^(13)) / Real.sqrt (a^(7)))^(1 / 3))

theorem problem1 (a : ℝ) (ha : a > 0) : problem1_expr a = 0 := 
  by
  sorry

end problem1_l716_716904


namespace car_original_cost_price_l716_716030

theorem car_original_cost_price :
  ∃ x : ℝ, (x >= 0) ∧ (0.86 * x * 1.20 = 54000) ∧ x = 52325.58 :=
begin
  sorry
end

end car_original_cost_price_l716_716030


namespace part_a_part_b_first_question_part_b_second_question_part_c_l716_716032

open EuclideanGeometry

-- Definition and theorems for problem part (a)
theorem part_a (A B C D X Y Z Y' : Point) (h_square : square A B C D 1)
  (hX : X ∈ segment A B) (hY : Y ∈ segment B C) (hZ : Z ∈ segment C D)
  (hY' : Y' ∈ segment C D):
  area_triangle X Y Z = area_triangle X Y' Z :=
sorry

-- Definition and theorems for problem part (b)
theorem part_b_first_question (A B C D X Y' Z : Point) (h_square : square A B C D 1)
  (hX : X ∈ segment A B) (hY' : Y' ∈ segment C D) (hZ : Z ∈ segment C D):
  max_area_triangle_one_side AB CD = 1 / 2 :=
sorry

theorem part_b_second_question (A B C D : Point) (h_square : square A B C D 1):
  max_area_triangle_inside_square (A, B, C, D) = 1 / 2 :=
sorry

-- Definition and theorems for problem part (c)
theorem part_c (pts : Fin 9 → Point) (h_square : square (pts 0) (pts 1) (pts 2) (pts 3) 2)
  (h_non_collinear : ∀ i j k, ¬ collinear i j k) :
  ∃ pt1 pt2 pt3, pt1 ≠ pt2 ∧ pt2 ≠ pt3 ∧ pt1 ≠ pt3 ∧ area_triangle pt1 pt2 pt3 ≤ 1 / 2 :=
sorry

end part_a_part_b_first_question_part_b_second_question_part_c_l716_716032


namespace determine_value_of_expression_l716_716648

theorem determine_value_of_expression (x y : ℤ) (h : y^2 + 4 * x^2 * y^2 = 40 * x^2 + 817) : 4 * x^2 * y^2 = 3484 :=
sorry

end determine_value_of_expression_l716_716648


namespace find_slope_of_line_l716_716342

noncomputable def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

theorem find_slope_of_line
  (k : ℝ)
  (h_pos : k > 0)
  (C1 C2 : ℝ → ℝ → Prop)
  (hC1 : ∀ x y, C1 x y ↔ (x - 1)^2 + y^2 = 1)
  (hC2 : ∀ x y, C2 x y ↔ (x - 3)^2 + y^2 = 1)
  (h_ratio : (2 / (Real.sqrt (k ^ 2 + 1))) = 3 * (2 * (Real.sqrt (1 - 8 * k ^ 2)) / (Real.sqrt (k ^ 2 + 1)))) :
  k = 1 / 3 := sorry

end find_slope_of_line_l716_716342


namespace maintain_mean_and_variance_l716_716417

def initial_set : Finset ℤ :=
  {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset ℤ) : ℚ :=
  (s.sum id : ℚ) / s.card

def variance (s : Finset ℤ) : ℚ :=
  (s.sum (λ x, (x * x : ℚ))) / (s.card : ℚ) - (mean s)^2

theorem maintain_mean_and_variance :
  ∃ (a b c : ℤ), a ∈ initial_set ∧
                 b ∉ initial_set ∧ 
                 c ∉ initial_set ∧ 
                 mean initial_set = mean (initial_set.erase a ∪ {b, c}) ∧
                 variance initial_set = variance (initial_set.erase a ∪ {b, c})
  :=
begin
  sorry
end

end maintain_mean_and_variance_l716_716417


namespace regular_polygon_sides_l716_716156

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716156


namespace regular_polygon_sides_l716_716091

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716091


namespace fly_path_exists_l716_716056

-- Define a position on the chessboard
structure Position where
  x : Nat
  y : Nat
  deriving DecidableEq, Repr

-- Define a step as a movement between positions
structure Step where
  from : Position
  to : Position
  deriving DecidableEq, Repr

-- Define a path as a list of steps
abbrev Path := List Step

-- Predicate to check if a position is white on a standard chessboard
def isWhite (p : Position) : Bool :=
  (p.x + p.y) % 2 = 0

-- Predicate to check if a path is valid given the conditions
def validPath (path : Path) : Bool :=
  let positions := path.bind (λ step => [step.from, step.to])
  positions.length = path.length + 1 ∧
  positions.Nodup ∧
  positions.all isWhite ∧
  let intersections := path.map (λ step => (step.from, step.to))
  intersections.Nodup

-- Define the chessboard size
def boardSize := 8

-- Define the top-left starting position
def startPos : Position := ⟨0, 0⟩

-- Problem statement: there exists a path from startPos that is valid and uses exactly 17 steps
theorem fly_path_exists :
  ∃ (path : Path),
    path.length = 17 ∧
    validPath path ∧
    path.head? = some (Step.mk startPos startPos) :=
by
  sorry -- Proof is not required

end fly_path_exists_l716_716056


namespace regular_polygon_sides_l716_716186

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l716_716186


namespace students_taking_art_l716_716918

theorem students_taking_art :
  ∀ (total_students music_students both_students neither_students : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_students = 10 →
  neither_students = 460 →
  music_students + both_students + neither_students = total_students →
  ((total_students - neither_students) - (music_students - both_students) + both_students = 20) :=
by
  intros total_students music_students both_students neither_students 
  intro h_total h_music h_both h_neither h_sum 
  sorry

end students_taking_art_l716_716918


namespace count_even_tens_digit_of_square_l716_716287

/-- There are 35 integers n in {1, 2, 3, ..., 50} for which the tens digit of n^2 is even -/
theorem count_even_tens_digit_of_square : 
  (∃ (n : ℕ), n ∈ {n : ℕ | n ≤ 50 ∧ (n^2 % 100) / 10 % 2 = 0}.card = 35) := 
sorry

end count_even_tens_digit_of_square_l716_716287


namespace find_ordered_pair_l716_716669

theorem find_ordered_pair :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a < b ∧ (real.sqrt (49 + real.sqrt (153 + 24 * real.sqrt 3)) = real.sqrt a + real.sqrt b) ∧ a = 1 ∧ b = 49 := 
by 
  sorry

end find_ordered_pair_l716_716669


namespace remainder_of_large_power_l716_716670

theorem remainder_of_large_power :
  (2^(2^(2^2))) % 500 = 36 :=
sorry

end remainder_of_large_power_l716_716670


namespace regular_polygon_sides_l716_716076

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716076


namespace length_of_inner_rectangle_is_4_l716_716584

-- Defining the conditions and the final proof statement
theorem length_of_inner_rectangle_is_4 :
  ∃ y : ℝ, y = 4 ∧
  let inner_width := 2
  let second_width := inner_width + 4
  let largest_width := second_width + 4
  let inner_area := inner_width * y
  let second_area := 6 * second_width
  let largest_area := 10 * largest_width
  let first_shaded_area := second_area - inner_area
  let second_shaded_area := largest_area - second_area
  (first_shaded_area - inner_area = second_shaded_area - first_shaded_area)
:= sorry

end length_of_inner_rectangle_is_4_l716_716584


namespace solution_inequality_part1_solution_inequality_part2_l716_716338

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem solution_inequality_part1 (x : ℝ) :
  (f x + x^2 - 4 > 0) ↔ (x > 2 ∨ x < -1) :=
sorry

theorem solution_inequality_part2 (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → (m > 3) :=
sorry

end solution_inequality_part1_solution_inequality_part2_l716_716338


namespace average_of_hidden_primes_l716_716250

theorem average_of_hidden_primes :
  (∃ (p1 p2 p3 : ℕ), 
    nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ 
    81 + p1 = 52 + p2 ∧ 
    81 + p1 = 47 + p3) → 
  ((81 + 52 + 47) + p1 + p2 + p3) / 6 = 39.67 :=
by sorry

end average_of_hidden_primes_l716_716250


namespace simplify_fraction_l716_716482

variable {a b : ℝ}

theorem simplify_fraction (ha : a ≠ 0) (hb : b ≠ 0) : 
  4 * a^(2/3) * b^(-1/3) / (- (2/3) * a^(-1/3) * b^(2/3)) = - (6 * a) / b := 
by sorry

end simplify_fraction_l716_716482


namespace proof_x_y_l716_716689

-- Define proof conditions and hypotheses
variables (x y : ℝ)
variable (h : abs (x - log10 y) = x + log10 y)

-- Prove that x * (y - 1) = 0
theorem proof_x_y (x y : ℝ) (h : abs(x - log10 y) = x + log10 y) : x * (y - 1) = 0 :=
sorry

end proof_x_y_l716_716689


namespace irr_ratio_pi_div_2_l716_716553

theorem irr_ratio_pi_div_2 :
    ¬ ∃ (q : ℚ), (q : ℝ) = real.pi / 2 := 
sorry

end irr_ratio_pi_div_2_l716_716553


namespace rival_awards_l716_716780

theorem rival_awards (jessie_multiple : ℕ) (scott_awards : ℕ) (rival_multiple : ℕ) 
  (h1 : jessie_multiple = 3) 
  (h2 : scott_awards = 4) 
  (h3 : rival_multiple = 2) 
  : (rival_multiple * (jessie_multiple * scott_awards) = 24) :=
by 
  sorry

end rival_awards_l716_716780


namespace expand_expression_l716_716272

theorem expand_expression : ∀ (x : ℝ), (17 * x + 21) * 3 * x = 51 * x^2 + 63 * x :=
by
  intro x
  sorry

end expand_expression_l716_716272


namespace people_not_buying_coffee_l716_716470

theorem people_not_buying_coffee (total_people : ℕ) (fraction_buying_coffee : ℚ)
  (h_total_people : total_people = 25)
  (h_fraction_buying_coffee : fraction_buying_coffee = 3/5) :
  (total_people - (fraction_buying_coffee * total_people).toNat) = 10 := by
  sorry

end people_not_buying_coffee_l716_716470


namespace sum_of_integer_solutions_l716_716892

theorem sum_of_integer_solutions :
  (∑ n in {n : ℤ | |n| < |n - 5| ∧ |n - 5| < 5}.to_finset, n) = 3 := 
by
  sorry

end sum_of_integer_solutions_l716_716892


namespace ball_height_less_than_2_feet_l716_716577

def initial_height : ℝ := 20
def bounce_ratio : ℝ := 2 / 3
def air_resistance_factor : ℝ := 0.95
def height_after_n_bounces (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio * air_resistance_factor) ^ n

theorem ball_height_less_than_2_feet :
  ∃ (n : ℕ), height_after_n_bounces n < 2 ∧ 
             ∀ m < n, ¬(height_after_n_bounces m < 2) :=
sorry

end ball_height_less_than_2_feet_l716_716577


namespace problem_1_problem_2_problem_3_l716_716249

noncomputable def x1 := 0.25 + (-9) + (-1/4) - (+11)
noncomputable def x2 := -15 + 5 + (1/3) * (-6)
noncomputable def x3 := ((-3 / 8) - (1 / 6) + (3 / 4)) * 24

theorem problem_1 : x1 = -20 := by
  sorry

theorem problem_2 : x2 = -12 := by
  sorry

theorem problem_3 : x3 = 5 := by
  sorry

end problem_1_problem_2_problem_3_l716_716249


namespace train_crosses_signal_pole_l716_716576

theorem train_crosses_signal_pole 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (time_cross_platform : ℝ) 
  (speed : ℝ) 
  (time_cross_signal_pole : ℝ) : 
  length_train = 400 → 
  length_platform = 200 → 
  time_cross_platform = 45 → 
  speed = (length_train + length_platform) / time_cross_platform → 
  time_cross_signal_pole = length_train / speed -> 
  time_cross_signal_pole = 30 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1] at h5
  -- Add the necessary calculations here
  sorry

end train_crosses_signal_pole_l716_716576


namespace increasing_interval_of_f_l716_716645

def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - x - 1)

theorem increasing_interval_of_f :
  ∀ x y : ℝ, x < y → x ∈ Set.Ioo (-(⊤ : ℝ)) (1 / 2) → y ∈ Set.Ioo (-(⊤ : ℝ)) (1 / 2) → f x < f y :=
by
  sorry

end increasing_interval_of_f_l716_716645


namespace son_work_rate_l716_716061

-- Define the conditions
def man_work_rate : ℚ := 1 / 4
def combined_work_rate : ℚ := 1 / 3

-- Define the problem statement as a theorem
theorem son_work_rate : ∃ (S: ℚ), man_work_rate + S = combined_work_rate ∧ 1 / S = 12 :=
by
  use 1 / 12
  split
  . rw man_work_rate
    rw combined_work_rate
    norm_num
    sorry
  . norm_num
    sorry

end son_work_rate_l716_716061


namespace coefficient_of_x_five_in_expansion_of_3_sub_x_7_l716_716262

noncomputable def coefficient_x_five : ℤ := 
  let a := 3
  let b := λ x: ℤ, x
  let n := 7
  let r := 5
  (-1)^r * a^(n - r) * nat.choose n r

theorem coefficient_of_x_five_in_expansion_of_3_sub_x_7 : coefficient_x_five = -189 :=
by
  sorry

end coefficient_of_x_five_in_expansion_of_3_sub_x_7_l716_716262


namespace tan_theta_cos_double_angle_minus_pi_over_3_l716_716700

open Real

-- Given conditions
variable (θ : ℝ)
axiom sin_theta : sin θ = 3 / 5
axiom theta_in_second_quadrant : π / 2 < θ ∧ θ < π

-- Questions and answers to prove:
theorem tan_theta : tan θ = - 3 / 4 :=
sorry

theorem cos_double_angle_minus_pi_over_3 : cos (2 * θ - π / 3) = (7 - 24 * Real.sqrt 3) / 50 :=
sorry

end tan_theta_cos_double_angle_minus_pi_over_3_l716_716700


namespace wickets_taken_in_last_match_l716_716929

noncomputable def wickets_last_match
  (W : ℕ) -- Wickets before last match
  (initial_average : ℚ)
  (runs_last_match : ℚ)
  (decrease_in_average : ℚ)
  (new_average : ℚ) 
  : ℚ :=
  let total_runs_before := initial_average * W
  let total_runs_after := total_runs_before + runs_last_match
  let total_wickets_after := W + new_average
  (total_runs_after / total_wickets_after - new_average + decrease_in_average) 

/-- Proof statement: Given the conditions, the number of wickets taken in the last match is 5 -/
theorem wickets_taken_in_last_match
  (initial_average runs_last_match decrease_in_average : ℚ)
  (W : ℕ)
  (approx_W : W ≈ 85)
  (total_runs_before := initial_average * W)
  (total_runs_after := total_runs_before + runs_last_match)
  (new_wicket_count := W + 5) -- Based on the answer
  (new_average := initial_average - decrease_in_average) :
  (total_runs_after / new_wicket_count = new_average) :=
by
  sorry

end wickets_taken_in_last_match_l716_716929


namespace inequality_for_real_numbers_l716_716475

theorem inequality_for_real_numbers (n : ℕ) (a : Fin n → ℝ) :
  ∑ i in Finset.range n, Real.sqrt ((a i)^2 + (1 - a (i+1) % n)^2) ≥ (n * Real.sqrt 2) / 2 :=
sorry

end inequality_for_real_numbers_l716_716475


namespace not_directly_or_inversely_proportional_l716_716772

theorem not_directly_or_inversely_proportional
  (P : ∀ x y : ℝ, x + y = 0 → (∃ k : ℝ, x = k * y))
  (Q : ∀ x y : ℝ, 3 * x * y = 10 → ∃ k : ℝ, x * y = k)
  (R : ∀ x y : ℝ, x = 5 * y → (∃ k : ℝ, x = k * y))
  (S : ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y))
  (T : ∀ x y : ℝ, x / y = Real.sqrt 3 → (∃ k : ℝ, x = k * y)) :
  ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y) := by
  sorry

end not_directly_or_inversely_proportional_l716_716772


namespace find_value_of_expression_l716_716687

variable {x : ℝ}

theorem find_value_of_expression (h : x^2 - 2 * x = 3) : 3 * x^2 - 6 * x - 4 = 5 :=
sorry

end find_value_of_expression_l716_716687


namespace Gianna_daily_savings_l716_716678

theorem Gianna_daily_savings 
  (total_saved : ℕ) (days_in_year : ℕ) 
  (H1 : total_saved = 14235) 
  (H2 : days_in_year = 365) : 
  total_saved / days_in_year = 39 := 
by 
  sorry

end Gianna_daily_savings_l716_716678


namespace area_qadec_correct_l716_716762

-- Given conditions
variables {A B C D E : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (AB AC AD BD DE : ℝ)
variables (angle_C_is_90 : EuclideanGeometry.Angle A C B = 90)
variables (AD_eq_DB : AD = BD)
variables (DE_perpendicular_AB : ⊥ DE AB)
variables (length_AB : AB = 30)
variables (length_AC : AC = 18)

-- Definition of the area calculation problem
def area_qadec : ℝ :=
  131.625

-- Statement of the proof
theorem area_qadec_correct :
  quadrilateral_area A D E C = 131.625 :=
sorry

end area_qadec_correct_l716_716762


namespace sum_of_coefficients_of_x_minus_3y_pow_20_l716_716265

/-- The sum of the numerical coefficients of all terms in the expansion of (x - 3y)^{20} is 2^{20}. -/
theorem sum_of_coefficients_of_x_minus_3y_pow_20 :
  let f := (fun (x y : ℝ) => (x - 3 * y) ^ 20) in
  f 1 1 = 2 ^ 20 := by
  sorry

end sum_of_coefficients_of_x_minus_3y_pow_20_l716_716265


namespace inequality_solution_set_l716_716480

   theorem inequality_solution_set : 
     {x : ℝ | (4 * x - 5)^2 + (3 * x - 2)^2 < (x - 3)^2} = {x : ℝ | (2 / 3 : ℝ) < x ∧ x < (5 / 4 : ℝ)} :=
   by
     sorry
   
end inequality_solution_set_l716_716480


namespace palm_trees_total_l716_716596

theorem palm_trees_total
  (forest_palm_trees : ℕ := 5000)
  (desert_palm_trees : ℕ := forest_palm_trees - (3 * forest_palm_trees / 5)) :
  desert_palm_trees + forest_palm_trees = 7000 :=
by
  sorry

end palm_trees_total_l716_716596


namespace value_of_b_l716_716318

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end value_of_b_l716_716318


namespace find_a_plus_b_l716_716808

def star (a b : ℕ) : ℕ := a^b - a*b + 5

theorem find_a_plus_b (a b : ℕ) (ha : 2 ≤ a) (hb : 3 ≤ b) (h : star a b = 13) : a + b = 6 :=
  sorry

end find_a_plus_b_l716_716808


namespace teacher_age_frequency_l716_716945

theorem teacher_age_frequency (f_less_than_30 : ℝ) (f_between_30_and_50 : ℝ) (h1 : f_less_than_30 = 0.3) (h2 : f_between_30_and_50 = 0.5) :
  1 - f_less_than_30 - f_between_30_and_50 = 0.2 :=
by
  rw [h1, h2]
  norm_num

end teacher_age_frequency_l716_716945


namespace find_m_l716_716719

theorem find_m 
  (m : ℝ) 
  (h1 : |m + 1| ≠ 0)
  (h2 : m^2 = 1) : 
  m = 1 := sorry

end find_m_l716_716719


namespace probability_crane_reaches_lily_pad_14_l716_716511

theorem probability_crane_reaches_lily_pad_14 :
  let num_pads := 16
  let predators := {4, 7, 12}
  let food_pad := 14
  let start_pad := 0
  let hop_prob := (1 : ℚ) / 2
  let leap_prob := (1 : ℚ) / 2
  let reach_prob := (3 : ℚ) / 512
  start_pad < num_pads ∧ food_pad < num_pads ∧ (∀ p ∈ predators, p < num_pads)
  → ∃ path: list ℕ, (path.head = some start_pad ∧ path.last = some food_pad ∧ path.forall (λ p, p ∉ predators))
  → (reach_prob = 3 / 512 : ℚ) := by
  sorry

end probability_crane_reaches_lily_pad_14_l716_716511


namespace rival_awards_l716_716787

theorem rival_awards (S J R : ℕ) (h1 : J = 3 * S) (h2 : S = 4) (h3 : R = 2 * J) : R = 24 := 
by sorry

end rival_awards_l716_716787


namespace regular_polygon_sides_l716_716150

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l716_716150


namespace max_points_on_circle_l716_716067

noncomputable def max_intersection_points (r : ℝ) : ℕ :=
if 4 ≤ r ∧ r ≤ 12 then 2 else 0

theorem max_points_on_circle (r : ℝ) (dist_PC : ℝ) (dist_center : ℝ) :
  dist_center = 8 ∧ dist_PC = 4 → max_intersection_points r = 2 :=
begin
  intros h,
  cases h with h1 h2,
  unfold max_intersection_points,
  split_ifs,
  { obvious }, -- we can assume 4 ≤ r ∧ r ≤ 12 case
  { obvious }, -- we can assume ¬ (4 ≤ r ∧ r ≤ 12) case
  sorry
end

end max_points_on_circle_l716_716067


namespace harmonic_mean_closest_to_one_l716_716969

-- Define the given conditions a = 1/4 and b = 2048
def a : ℚ := 1 / 4
def b : ℚ := 2048

-- Define the harmonic mean of two numbers
def harmonic_mean (x y : ℚ) : ℚ := 2 * x * y / (x + y)

-- State the theorem proving the harmonic mean is closest to 1
theorem harmonic_mean_closest_to_one : abs (harmonic_mean a b - 1) < 1 :=
sorry

end harmonic_mean_closest_to_one_l716_716969


namespace liars_count_l716_716906

inductive Person
| Knight
| Liar
| Eccentric

open Person

def isLiarCondition (p : Person) (right : Person) : Prop :=
  match p with
  | Knight => right = Liar
  | Liar => right ≠ Liar
  | Eccentric => True

theorem liars_count (people : Fin 100 → Person) (h : ∀ i, isLiarCondition (people i) (people ((i + 1) % 100))) :
  (∃ n : ℕ, n = 0 ∨ n = 50) :=
sorry

end liars_count_l716_716906


namespace find_costs_of_items_find_number_of_volleyballs_l716_716054

-- Definitions for the first part
def cost_volleyball (x : ℝ) := x
def cost_soccerball (x : ℝ) := x + 30
def total_cost_first (x : ℝ) := 40 * cost_soccerball(x) + 30 * cost_volleyball(x)

-- Definitions for the second part
def cost_soccerball_second := 70 * 1.1
def cost_volleyball_second := 40 * 0.9
def total_cost_second (m : ℕ) := 77 * (50 - m) + 36 * m

theorem find_costs_of_items :
  ∃ x, total_cost_first(x) = 4000 ∧ cost_volleyball(x) = 40 ∧ cost_soccerball(x) = 70 :=
begin
  sorry
end

theorem find_number_of_volleyballs :
  ∃ m, total_cost_second(m) = 3440 ∧ m = 10 :=
begin
  sorry
end

end find_costs_of_items_find_number_of_volleyballs_l716_716054


namespace combination_recurrence_l716_716822

variable {n r : ℕ}
variable (C : ℕ → ℕ → ℕ)

theorem combination_recurrence (hn : n > 0) (hr : r > 0) (h : n > r)
  (h2 : ∀ (k : ℕ), k = 1 → C 2 1 = C 1 1 + C 1) 
  (h3 : ∀ (k : ℕ), k = 1 → C 3 1 = C 2 1 + C 2) 
  (h4 : ∀ (k : ℕ), k = 2 → C 3 2 = C 2 2 + C 2 1)
  (h5 : ∀ (k : ℕ), k = 1 → C 4 1 = C 3 1 + C 3) 
  (h6 : ∀ (k : ℕ), k = 2 → C 4 2 = C 3 2 + C 3 1)
  (h7 : ∀ (k : ℕ), k = 3 → C 4 3 = C 3 3 + C 3 2)
  (h8 : ∀ n r : ℕ, (n > r) → C n r = C (n-1) r + C (n-1) (r-1)) :
  C n r = C (n-1) r + C (n-1) (r-1) :=
sorry

end combination_recurrence_l716_716822


namespace find_borrow_interest_rate_l716_716931

-- Define the problem conditions and question as a theorem
theorem find_borrow_interest_rate : 
  ∀ (P B L G : ℝ), P = 8000 → L = 6/100 → G = 160 → 
  (320 = (r / 100) * P) → r = 4 :=
by 
  intros,
  sorry -- No proof is required.

end find_borrow_interest_rate_l716_716931


namespace regular_polygon_sides_l716_716073

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716073


namespace length_of_PS_l716_716770

noncomputable def triangle_segments : ℝ := 
  let PR := 15
  let ratio_PS_SR := 3 / 4
  let total_length := 15
  let SR := total_length / (1 + ratio_PS_SR)
  let PS := ratio_PS_SR * SR
  PS

theorem length_of_PS :
  triangle_segments = 45 / 7 :=
by
  sorry

end length_of_PS_l716_716770


namespace regular_polygon_sides_l716_716215

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l716_716215


namespace discount_percentage_is_correct_l716_716227

noncomputable def calculate_discount_percentage
  (C : ℝ)  -- Cost price of the article
  (Sd : ℝ) -- Selling price with discount
  (S : ℝ)  -- Selling price without discount
  (hSd : Sd = 1.2255 * C)
  (hS : S = 1.29 * C) :
  Prop :=
  let d := ((S - Sd) / S) * 100 in
  d = 5

theorem discount_percentage_is_correct {C Sd S : ℝ}
  (hSd : Sd = 1.2255 * C) (hS : S = 1.29 * C) :
  calculate_discount_percentage C Sd S hSd hS :=
by
  unfold calculate_discount_percentage
  sorry

end discount_percentage_is_correct_l716_716227


namespace difference_between_C_and_D_l716_716656

def C : ℕ := ∑ i in finset.filter (λ n, n % 2 = 0) (finset.range 41), n * (n + 1) / 2 + 40

def D : ℕ :=  ∑ i in finset.filter (λ n, (n % 2 = 1 ∧ n < 39)) (finset.range 41), n * (n + 1) / 2

theorem difference_between_C_and_D : |C - D| = 1159 := 
by 
sory

end difference_between_C_and_D_l716_716656


namespace initial_people_l716_716528

theorem initial_people (x : ℕ) (h₁ : 40 * x ≥ 40) (h₂ : ∀ t, t ≤ 40)
  (h₃ : ∀ p, p = x) (h₄ : 2 * x + 4 * (x - 2) = 40) : x = 8 :=
begin
  sorry
end

end initial_people_l716_716528


namespace sum_of_all_n_l716_716722

noncomputable def a₁ := 3/2
def S (n : ℕ) : ℝ := 3 - 3 * (1/2)^n

lemma sum_satisfying_n_lt_17 (n : ℕ) : (18/17 < S (2 * n) / S n ∧ S (2 * n) / S n < 8/7) ↔ (3 ≤ n ∧ n ≤ 4) := sorry

theorem sum_of_all_n : ∑ n in {n | 18/17 < S (2 * n) / S n ∧ S (2 * n) / S n < 8/7}.to_finset, n = 7 := 
begin
  have h3 : 18/17 < S (2 * 3) / S 3 ∧ S (2 * 3) / S 3 < 8/7, from sorry,
  have h4 : 18/17 < S (2 * 4) / S 4 ∧ S (2 * 4) / S 4 < 8/7, from sorry,
  have hn : ∀ n', (n' ∈ {n | 18/17 < S (2 * n) / S n ∧ S (2 * n) / S n < 8/7}.to_finset) ↔ (n' = 3 ∨ n' = 4), 
    from sorry,
  rw finset.sum_eq_add (by simp [set.singleton_iff.mpr h3, set.singleton_iff.mpr h4]),
  exact h3.right,
  exact h4.right,
  finset.sum_congr rfl hn,
  norm_num
end

end sum_of_all_n_l716_716722


namespace trig_proof_l716_716315

theorem trig_proof (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α = sqrt 2 / 10) (h4 : Real.sin β = sqrt 10 / 10) :
  Real.cos (2 * β) = 4 / 5 ∧ α + 2 * β = π / 4 := 
by
  -- Proof goes here
  sorry

end trig_proof_l716_716315


namespace determine_height_impossible_l716_716747

-- Definitions used in the conditions
def shadow_length_same (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) : Prop :=
  xiao_ming_height / xiao_ming_distance = xiao_qiang_height / xiao_qiang_distance

-- The proof problem: given that the shadow lengths are the same under the same street lamp,
-- prove that it is impossible to determine who is taller.
theorem determine_height_impossible (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) :
  shadow_length_same xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance →
  ¬ (xiao_ming_height ≠ xiao_qiang_height ↔ true) :=
by
  intro h
  sorry -- Proof not required as per instructions

end determine_height_impossible_l716_716747


namespace prob_of_ξ_l716_716746

noncomputable def ξ : MeasureTheory.Measure ℝ := MeasureTheory.ProbabilityMeasure.gaussian (-1) σ^2

theorem prob_of_ξ :
  (∃ξ : ℝ → MeasureTheory.ProbabilityTheory.Measure ℝ, MeasureTheory.ProbabilityMeasure.gaussian (-1) σ^2 = ξ) →
  MeasureTheory.MeasureTheory.ProbabilityMeasure.P(-3 ≤ ξ ∧ ξ ≤ -1) = 0.4 →
  MeasureTheory.MeasureTheory.ProbabilityMeasure.P(ξ ≥ 1) = 0.1 :=
by
  sorry

end prob_of_ξ_l716_716746


namespace richard_played_games_l716_716625

theorem richard_played_games (x : ℕ) 
  (h1 : ∃ x, ∀ t, t = 6 * x) -- Richard has scored 6x touchdowns in x games
  (h2 : ∀ t, 3 * 2 = 6) -- Richard will score 3 touchdowns per game in the final 2 games
  (h3 : ∀ t, t ≥ 90) -- Richard needs to score at least 90 touchdowns to beat the record
  (h4 : ∀ t, t = 6 * x + 6) -- Total touchdowns after the final 2 games
  (h5 : 6 * x + 6 = 90) : x = 14 :=
begin
  sorry
end

end richard_played_games_l716_716625


namespace baseball_card_decrease_l716_716911

theorem baseball_card_decrease (x : ℝ) :
  (0 < x) ∧ (x < 100) ∧ (100 - x) * 0.9 = 45 → x = 50 :=
by
  intros h
  sorry

end baseball_card_decrease_l716_716911


namespace regular_polygon_sides_l716_716086

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716086


namespace maintain_mean_and_variance_l716_716416

def initial_set : Finset ℤ :=
  {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset ℤ) : ℚ :=
  (s.sum id : ℚ) / s.card

def variance (s : Finset ℤ) : ℚ :=
  (s.sum (λ x, (x * x : ℚ))) / (s.card : ℚ) - (mean s)^2

theorem maintain_mean_and_variance :
  ∃ (a b c : ℤ), a ∈ initial_set ∧
                 b ∉ initial_set ∧ 
                 c ∉ initial_set ∧ 
                 mean initial_set = mean (initial_set.erase a ∪ {b, c}) ∧
                 variance initial_set = variance (initial_set.erase a ∪ {b, c})
  :=
begin
  sorry
end

end maintain_mean_and_variance_l716_716416


namespace find_PF2_l716_716435

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

def foci (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-6, 0) ∧ F2 = (6, 0)

def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem find_PF2 :
  ∀ (P F1 F2 : ℝ × ℝ),
  point_on_hyperbola P →
  foci F1 F2 → 
  distance P F1 = 9 →
  distance P F2 = 17 :=
by
  intros P F1 F2 hP hf hdist
  sorry

end find_PF2_l716_716435


namespace rival_awards_l716_716786

theorem rival_awards (S J R : ℕ) (h1 : J = 3 * S) (h2 : S = 4) (h3 : R = 2 * J) : R = 24 := 
by sorry

end rival_awards_l716_716786


namespace simplify_and_evaluate_l716_716839

def expr (x : ℤ) : ℤ := (x + 2) * (x - 2) - (x - 1) ^ 2

theorem simplify_and_evaluate : expr (-1) = -7 := by
  sorry

end simplify_and_evaluate_l716_716839


namespace coordinates_T_l716_716274

noncomputable def point := (ℝ × ℝ)
def O : point := (0, 0)
def Q : point := (3, 3)
def P : point := (3, 0)
def R : point := (0, 3)
def area_square_OPQR : ℝ := 9
def area_ΔPQT : ℝ := 9
def T : point := (3, 3 * Real.sqrt 2)

theorem coordinates_T (h_area_square : area_square_OPQR = 9) 
                      (h_area_triangle : area_ΔPQT = 9) :
  T = (3, 3 * Real.sqrt 2) :=
sorry

end coordinates_T_l716_716274


namespace SX_TY_AD_concurrent_and_Z_eq_H_l716_716420

variables {A B C D P Q R S T X Y Z H : Type*}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] 
variables [inhabited P] [inhabited Q] [inhabited R] [inhabited S] 
variables [inhabited T] [inhabited X] [inhabited Y] [inhabited Z] 
variables [inhabited H]

-- Definitions of geometric entities, properties, and hypotheses
def triangle (A B C : Type*) := ∃ (AD : Type*) (P : Type*), 
  (P ∈ AD) ∧ 
  (is_foot_perpendicular (P Q : Type*) (Q ∈ AB)) ∧
  (is_foot_perpendicular (P R : Type*) (R ∈ AC)) ∧
  (QP_meets_BC_at_S (Q P S : Type*)) ∧
  (RP_meets_BC_at_T (R P T : Type*)) ∧
  (circumcircles_meet_QR (circumcircle_BQS circumcircle_CRT QR : Type*) (X Y : Type*)) ∧
  (concurrent_SX_TY_AD (S X T Y AD : Type*) (Z : Type*)) ∧
  (on_QR_iff_Z_eq_H (Z QR : Type*) (H : Type*))

-- Main theorem statement
theorem SX_TY_AD_concurrent_and_Z_eq_H (A B C D P Q R S T X Y Z H : Type*) 
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  [inhabited P] [inhabited Q] [inhabited R] [inhabited S] 
  [inhabited T] [inhabited X] [inhabited Y] [inhabited Z] 
  [inhabited H] :
  triangle A B C → 
  (concurrent_SX_TY_AD → Z ∈ QR ↔ Z = H) :=
sorry

end SX_TY_AD_concurrent_and_Z_eq_H_l716_716420


namespace sum_even_integers_divisible_by_4_l716_716544

theorem sum_even_integers_divisible_by_4 (h₁ : ∀ n, n ∈ {i : ℕ | 402 ≤ i ∧ i ≤ 1000 → i % 2 = 0 ∧ i % 4 = 0}) :
  (∑ i in finset.filter (λ x, x % 4 = 0) (finset.Icc 402 1000), i) = 105300 :=
by
  sorry

end sum_even_integers_divisible_by_4_l716_716544


namespace smallest_n_satisfies_l716_716673

noncomputable def smallest_n : ℕ :=
  778556334111889667445223

theorem smallest_n_satisfies (N : ℕ) : 
  (N > 0 ∧ ∃ k : ℕ, ∀ m:ℕ, N * 999 = (7 * ((10^k - 1) / 9) )) → N = smallest_n :=
begin
  sorry
end 

end smallest_n_satisfies_l716_716673


namespace natural_number_increased_by_one_l716_716062

theorem natural_number_increased_by_one (a : ℕ) 
  (h : (a + 1) ^ 2 - a ^ 2 = 1001) : 
  a = 500 := 
sorry

end natural_number_increased_by_one_l716_716062


namespace regular_polygon_sides_l716_716085

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716085


namespace value_of_a_l716_716005

theorem value_of_a (k : ℝ) (a : ℝ) (b : ℝ) (h1 : a = k / b^2) (h2 : a = 10) (h3 : b = 24) :
  a = 40 :=
sorry

end value_of_a_l716_716005


namespace regular_polygon_sides_l716_716201

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716201


namespace final_value_is_1_l716_716891

-- Define the initial state and the transition function.
def initial_state : (ℕ × ℕ) := (0, 5)

def transition : (ℕ × ℕ) → (ℕ × ℕ)
| (S, 5) := (S + 5, 4)
| (S, 4) := (S + 4, 4)
| (S, 4) := (S + 4, 3)
| (S, 3) := (S + 3, 3)
| (S, 3) := (S + 3, 2)
| (S, 2) := (S + 2, 2)
| (S, 2) := (S + 2, 1)
| (S, 1) := (S + 1, 1)
| (S, n) := (S, n) -- No change for other states

-- Define the function to compute the final states after a finite number of transitions.
def final_state (n : ℕ) : (ℕ × ℕ) :=
  (nat.iterate transition n initial_state)

-- The theorem to prove the final state results in (17, 1).
theorem final_value_is_1 : (final_state 5).2 = 1 := by
  -- Proof is skipped
  sorry

end final_value_is_1_l716_716891


namespace regular_polygon_sides_l716_716109

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716109


namespace average_daily_low_temperature_l716_716619

theorem average_daily_low_temperature (temps : List ℕ) (h_len : temps.length = 5) 
  (h_vals : temps = [40, 47, 45, 41, 39]) : 
  (temps.sum / 5 : ℝ) = 42.4 := 
by
  sorry

end average_daily_low_temperature_l716_716619


namespace determine_b_2056_l716_716442

namespace SequenceProblem

noncomputable def b : ℕ → ℝ
| 0       := sorry -- base case
| 1       := 2 + real.sqrt 12
| 2       := sorry -- base case
| (n + 1) := (b n) / (b (n - 1))

theorem determine_b_2056 (h_rec : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h_b1 : b 1 = 2 + real.sqrt 12)
  (h_b2023 : b 2023 = 14 + real.sqrt 12) :
  b 2056 = 1 + 2 * real.sqrt 12 :=
sorry

end SequenceProblem

end determine_b_2056_l716_716442


namespace unique_solution_for_a_l716_716330

theorem unique_solution_for_a (a : ℝ) :
  (∃! x : ℝ, 2 ^ |2 * x - 2| - a * Real.cos (1 - x) = 0) ↔ a = 1 :=
sorry

end unique_solution_for_a_l716_716330


namespace quadratic_expression_value_l716_716004

theorem quadratic_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 * x2 = 2) (hx : x1^2 - 4 * x1 + 2 = 0) :
  x1^2 - 4 * x1 + 2 * x1 * x2 = 2 :=
sorry

end quadratic_expression_value_l716_716004


namespace count_valid_numbers_l716_716355

def is_valid_n (N : ℕ) :=
  7000 ≤ N ∧ N < 9000 ∧
  let a := N / 1000 in
  let b := (N / 100) % 10 in
  let c := (N / 10) % 10 in
  let d := N % 10 in
  d = (b + c) % 5 ∧
  3 ≤ b ∧ b < c ∧ c ≤ 7

theorem count_valid_numbers : 
  (∑ N in Finset.range 10000, if is_valid_n N then 1 else 0) = 20 :=
by
  sorry

end count_valid_numbers_l716_716355


namespace number_of_classes_l716_716753

theorem number_of_classes (n : ℕ) (a₁ : ℕ) (d : ℤ) (S : ℕ) (h₁ : d = -2) (h₂ : a₁ = 25) (h₃ : S = 105) : n = 5 :=
by
  /- We state the theorem and the necessary conditions without proving it -/
  sorry

end number_of_classes_l716_716753


namespace servings_in_box_l716_716587

def totalCereal : ℕ := 18
def servingSize : ℕ := 2

theorem servings_in_box : totalCereal / servingSize = 9 := by
  sorry

end servings_in_box_l716_716587


namespace problem_253_base2_l716_716982

def decimal_to_binary (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec f (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc.reverse
      else f (n / 2) ((n % 2) :: acc)
    in f n []

def count_digits (lst : list ℕ) : ℕ × ℕ :=
  lst.foldr (λ d (acc : ℕ × ℕ), if d = 0 then (acc.1 + 1, acc.2) else (acc.1, acc.2 + 1)) (0, 0)

theorem problem_253_base2 :
  let bits := decimal_to_binary 253
  let (x, y) := count_digits bits
  y - x = 6 :=
by
  sorry

end problem_253_base2_l716_716982


namespace probability_length_error_in_interval_l716_716324

noncomputable def normal_dist_prob (μ σ : ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

theorem probability_length_error_in_interval :
  normal_dist_prob 0 3 3 6 = 0.1359 :=
by
  sorry

end probability_length_error_in_interval_l716_716324


namespace regular_polygon_sides_l716_716114

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l716_716114


namespace proof_incorrect_statement_c_l716_716550

noncomputable def incorrect_statement (x : ℝ) : Prop :=
  (0 < x ∧ x < 1) → (log x + log10 x < 2)

theorem proof_incorrect_statement_c (x : ℝ) :
  incorrect_statement x :=
by
  sorry

end proof_incorrect_statement_c_l716_716550


namespace regular_polygon_sides_l716_716199

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l716_716199


namespace max_value_ellipse_l716_716434

theorem max_value_ellipse (M : ℝ × ℝ) (hM : M.1^2 / 4 + M.2^2 = 1)
  (F1 F2 : ℝ × ℝ) (hF1 : F1 = (-√3, 0)) (hF2 : F2 = (√3, 0))
  (e : ℝ) (hE : e = √3 / 2) :
  ∃ x0 y0 : ℝ, (x0, y0) = M ∧ -2 ≤ x0 ∧ x0 ≤ 2 ∧ 
  ( √( ( (2 + √3 / 2 * x0) * (2 - √3 / 2 * x0) ) / (e^2) )
    = 4 * √3 / 3 ) :=
sorry

end max_value_ellipse_l716_716434


namespace problem1_minimum_problem2_maximum_l716_716048

-- Problem 1: Minimum value of f(x) given x > 0
theorem problem1_minimum (x : ℝ) (hx : x > 0) : 
  let f := (x^2 + 3*x + 2) / x in
  (∃ (c : ℝ), c = 2 * Real.sqrt 2 + 3 ∧ ∀ (y : ℝ), y = (x^2 + 3*x + 2) / x → y ≥ c) :=
  sorry

-- Problem 2: Maximum value of y given 0 < x < 1/2
theorem problem2_maximum (x : ℝ) (hx1 : 0 < x) (hx2 : x < 1/2) :
  let y := (1/2) * x * (1 - 2 * x) in
  (∃ (c : ℝ), c = 1 / 16 ∧ ∀ (y' : ℝ), y' = (1 / 2) * x * (1 - 2 * x) → y' ≤ c) :=
  sorry

end problem1_minimum_problem2_maximum_l716_716048
