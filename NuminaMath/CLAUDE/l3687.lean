import Mathlib

namespace square_difference_equal_l3687_368765

theorem square_difference_equal (a b : ℝ) : (a - b)^2 = (b - a)^2 := by
  sorry

end square_difference_equal_l3687_368765


namespace product_b3_b17_l3687_368736

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem product_b3_b17 (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_cond : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_eq : a 8 = b 10) :
  b 3 * b 17 = 36 := by
  sorry

end product_b3_b17_l3687_368736


namespace min_perimeter_triangle_l3687_368703

theorem min_perimeter_triangle (a b c : ℕ) : 
  a = 47 → b = 53 → c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a + b + c ≥ 107 :=
by sorry

end min_perimeter_triangle_l3687_368703


namespace negation_of_existential_proposition_l3687_368721

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 3 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 > 0) := by
  sorry

end negation_of_existential_proposition_l3687_368721


namespace new_assistant_drawing_time_main_theorem_l3687_368740

/-- Represents a beer barrel with two taps -/
structure BeerBarrel where
  capacity : ℕ
  midwayTapRate : ℚ  -- litres per minute
  lowerTapRate : ℚ   -- litres per minute

/-- Calculates the time taken to empty half the barrel using the midway tap -/
def timeToHalfEmpty (barrel : BeerBarrel) : ℚ :=
  (barrel.capacity / 2) / barrel.midwayTapRate

/-- Calculates the additional time the lower tap was used -/
def additionalLowerTapTime : ℕ := 24

/-- Theorem: The new assistant drew beer for 150 minutes -/
theorem new_assistant_drawing_time (barrel : BeerBarrel)
    (h1 : barrel.capacity = 36)
    (h2 : barrel.midwayTapRate = 1 / 6)
    (h3 : barrel.lowerTapRate = 1 / 4)
    : ℚ :=
  150

/-- Main theorem to prove -/
theorem main_theorem (barrel : BeerBarrel)
    (h1 : barrel.capacity = 36)
    (h2 : barrel.midwayTapRate = 1 / 6)
    (h3 : barrel.lowerTapRate = 1 / 4)
    : new_assistant_drawing_time barrel h1 h2 h3 = 150 := by
  sorry

end new_assistant_drawing_time_main_theorem_l3687_368740


namespace circle_max_values_l3687_368707

theorem circle_max_values (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = Real.sqrt 6 - 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = 7 + 4*Real.sqrt 3) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀ ≠ 0 ∧ y₀ / x₀ = Real.sqrt 3) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀ + y₀ = 2 + Real.sqrt 3) :=
by
  sorry

end circle_max_values_l3687_368707


namespace peasant_money_problem_l3687_368702

theorem peasant_money_problem (initial_money : ℕ) : 
  let after_first := initial_money / 2 - 1
  let after_second := after_first / 2 - 2
  let after_third := after_second / 2 - 1
  (after_third = 0) → initial_money = 6 := by
sorry

end peasant_money_problem_l3687_368702


namespace projectile_max_height_l3687_368729

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

/-- Theorem stating that the maximum height of the projectile is 60 meters -/
theorem projectile_max_height : 
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 60 := by
  sorry

end projectile_max_height_l3687_368729


namespace mens_average_weight_l3687_368795

theorem mens_average_weight (num_men : ℕ) (num_women : ℕ) (avg_women : ℝ) (avg_total : ℝ) :
  num_men = 8 →
  num_women = 6 →
  avg_women = 120 →
  avg_total = 160 →
  let total_people := num_men + num_women
  let avg_men := (avg_total * total_people - avg_women * num_women) / num_men
  avg_men = 190 := by
sorry

end mens_average_weight_l3687_368795


namespace modulus_of_z_l3687_368755

theorem modulus_of_z (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : ∃ (b : ℝ), (2 - i) / (a + i) = b * i) :
  Complex.abs (2 * a + Complex.I * Real.sqrt 3) = 2 := by
  sorry

end modulus_of_z_l3687_368755


namespace square_pentagon_side_ratio_l3687_368738

theorem square_pentagon_side_ratio (perimeter : ℝ) (square_side : ℝ) (pentagon_side : ℝ)
  (h1 : perimeter > 0)
  (h2 : 4 * square_side = perimeter)
  (h3 : 5 * pentagon_side = perimeter) :
  pentagon_side / square_side = 4 / 5 := by
sorry

end square_pentagon_side_ratio_l3687_368738


namespace fair_spending_l3687_368758

theorem fair_spending (initial_amount : ℝ) (ride_fraction : ℝ) (dessert_cost : ℝ) : 
  initial_amount = 30 →
  ride_fraction = 1/2 →
  dessert_cost = 5 →
  initial_amount - (ride_fraction * initial_amount) - dessert_cost = 10 := by
sorry

end fair_spending_l3687_368758


namespace point_on_line_l3687_368742

/-- Given a line with equation x = 2y + 3 and two points (m, n) and (m + 2, n + k) on this line,
    prove that k = 0. -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2 * n + 3) → 
  (m + 2 = 2 * (n + k) + 3) → 
  k = 0 := by
  sorry

end point_on_line_l3687_368742


namespace min_distance_to_line_l3687_368772

/-- Given a line 5x + 12y = 60, the minimum distance from the origin (0, 0) to any point (x, y) on this line is 60/13 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | 5 * x + 12 * y = 60}
  ∃ (d : ℝ), d = 60 / 13 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ line → 
      d ≤ Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2)) :=
by sorry

end min_distance_to_line_l3687_368772


namespace line_parallel_plane_implies_parallel_to_all_lines_false_l3687_368723

/-- A line in 3D space --/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space --/
structure Plane3D where
  -- Define properties of a plane

/-- Defines when a line is parallel to a plane --/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane --/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel --/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- The statement to be proven false --/
theorem line_parallel_plane_implies_parallel_to_all_lines_false :
  ¬ (∀ (l : Line3D) (p : Plane3D),
    line_parallel_plane l p →
    ∀ (l' : Line3D), line_in_plane l' p →
    lines_parallel l l') :=
  sorry

end line_parallel_plane_implies_parallel_to_all_lines_false_l3687_368723


namespace two_pants_three_tops_six_looks_l3687_368791

/-- The number of possible looks given a number of pants and tops -/
def number_of_looks (pants : ℕ) (tops : ℕ) : ℕ := pants * tops

/-- Theorem stating that 2 pairs of pants and 3 pairs of tops result in 6 looks -/
theorem two_pants_three_tops_six_looks : 
  number_of_looks 2 3 = 6 := by
  sorry

end two_pants_three_tops_six_looks_l3687_368791


namespace parabola_line_intersection_dot_product_l3687_368732

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = (2/3)*(x + 2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the intersection points
def M : ℝ × ℝ := (1, 2)
def N : ℝ × ℝ := (4, 4)

-- Define vectors FM and FN
def FM : ℝ × ℝ := (M.1 - focus.1, M.2 - focus.2)
def FN : ℝ × ℝ := (N.1 - focus.1, N.2 - focus.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem parabola_line_intersection_dot_product :
  parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
  line M.1 M.2 ∧ line N.1 N.2 →
  dot_product FM FN = 8 :=
by sorry

end parabola_line_intersection_dot_product_l3687_368732


namespace part_one_part_two_l3687_368787

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬(q x)) →
  (∀ a : ℝ, 1 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l3687_368787


namespace unique_solution_n_times_s_l3687_368730

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * f y + 2 * x) = 2 * x * y + f x

/-- The theorem stating that f(3) = -2 is the only solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 3 = -2 := by
  sorry

/-- The number of possible values for f(3) -/
def n : ℕ := 1

/-- The sum of all possible values for f(3) -/
def s : ℝ := -2

/-- The product of n and s -/
theorem n_times_s : n * s = -2 := by
  sorry

end unique_solution_n_times_s_l3687_368730


namespace min_value_of_sequence_l3687_368708

theorem min_value_of_sequence (n : ℝ) : 
  ∃ (m : ℝ), ∀ (n : ℝ), n^2 - 8*n + 5 ≥ m ∧ ∃ (k : ℝ), k^2 - 8*k + 5 = m :=
by
  -- The minimum value is -11
  use -11
  sorry

end min_value_of_sequence_l3687_368708


namespace geometric_progression_first_term_l3687_368748

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 10)
  (h2 : sum_first_two = 6) :
  ∃ (a : ℝ), 
    (a = 10 - 10 * Real.sqrt (2/5) ∨ a = 10 + 10 * Real.sqrt (2/5)) ∧ 
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end geometric_progression_first_term_l3687_368748


namespace x_over_y_value_l3687_368794

theorem x_over_y_value (x y z : ℝ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21)
  (eq4 : y / z = 6) :
  x / y = 1 / 3 := by
sorry

end x_over_y_value_l3687_368794


namespace unknown_blanket_rate_l3687_368747

/-- Given information about blanket purchases and average price, 
    prove the unknown rate of two blankets. -/
theorem unknown_blanket_rate 
  (blanket_count_1 blanket_count_2 blanket_count_unknown : ℕ)
  (price_1 price_2 average_price : ℚ)
  (h1 : blanket_count_1 = 3)
  (h2 : blanket_count_2 = 3)
  (h3 : blanket_count_unknown = 2)
  (h4 : price_1 = 100)
  (h5 : price_2 = 150)
  (h6 : average_price = 150)
  (h7 : (blanket_count_1 * price_1 + blanket_count_2 * price_2 + 
         blanket_count_unknown * unknown_rate) / 
        (blanket_count_1 + blanket_count_2 + blanket_count_unknown) = 
        average_price) :
  unknown_rate = 225 := by
  sorry


end unknown_blanket_rate_l3687_368747


namespace consecutive_integers_sqrt_3_l3687_368753

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 3 ∧ Real.sqrt 3 < ↑b) → a + b = 3 := by
  sorry

end consecutive_integers_sqrt_3_l3687_368753


namespace sequence_fifth_term_l3687_368711

theorem sequence_fifth_term (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
  sorry

end sequence_fifth_term_l3687_368711


namespace exists_convex_polyhedron_with_triangular_section_l3687_368709

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- A cross-section of a polyhedron -/
structure CrossSection where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Predicate to check if a cross-section is triangular -/
def is_triangular (cs : CrossSection) : Prop :=
  sorry

/-- Predicate to check if a cross-section passes through vertices -/
def passes_through_vertices (p : ConvexPolyhedron) (cs : CrossSection) : Prop :=
  sorry

/-- Function to count the number of edges meeting at a vertex -/
def edges_at_vertex (p : ConvexPolyhedron) (v : ℕ) : ℕ :=
  sorry

/-- Theorem stating the existence of a convex polyhedron with the specified properties -/
theorem exists_convex_polyhedron_with_triangular_section :
  ∃ (p : ConvexPolyhedron) (cs : CrossSection),
    is_triangular cs ∧
    ¬passes_through_vertices p cs ∧
    ∀ (v : ℕ), edges_at_vertex p v = 5 :=
  sorry

end exists_convex_polyhedron_with_triangular_section_l3687_368709


namespace zeros_in_square_expansion_l3687_368720

theorem zeros_in_square_expansion (n : ℕ) : 
  (∃ k : ℕ, (10^15 - 3)^2 = k * 10^n ∧ k % 10 ≠ 0) → n = 29 :=
sorry

end zeros_in_square_expansion_l3687_368720


namespace mark_and_carolyn_money_l3687_368725

theorem mark_and_carolyn_money : 
  3/4 + 3/10 = 21/20 := by sorry

end mark_and_carolyn_money_l3687_368725


namespace function_zeros_count_l3687_368724

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_zeros_count
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_periodic : is_periodic f 3)
  (h_sin : ∀ x ∈ Set.Ioo 0 (3/2), f x = Real.sin (Real.pi * x))
  (h_zero : f (3/2) = 0) :
  ∃ S : Finset ℝ, S.card = 7 ∧ (∀ x ∈ S, x ∈ Set.Icc 0 6 ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 6, f x = 0 → x ∈ S) :=
sorry

end function_zeros_count_l3687_368724


namespace problem_1_problem_2_problem_3_l3687_368769

theorem problem_1 (x y : ℝ) : 3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2 := by
  sorry

theorem problem_2 (a b : ℝ) (h : a^2 - 2*b = 2) : 4*a^2 - 8*b - 9 = -1 := by
  sorry

theorem problem_3 (a b c d : ℝ) (h1 : a - 2*b = 4) (h2 : b - c = -5) (h3 : 3*c + d = 10) :
  (a + 3*c) - (2*b + c) + (b + d) = 9 := by
  sorry

end problem_1_problem_2_problem_3_l3687_368769


namespace all_cells_happy_l3687_368733

def Board := Fin 10 → Fin 10 → Bool

def isBlue (board : Board) (i j : Fin 10) : Bool :=
  (i.val + j.val) % 2 = 0

def neighbors (i j : Fin 10) : List (Fin 10 × Fin 10) :=
  [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    |> List.filter (fun (x, y) => x.val < 10 && y.val < 10)

def countBlueNeighbors (board : Board) (i j : Fin 10) : Nat :=
  (neighbors i j).filter (fun (x, y) => isBlue board x y) |>.length

theorem all_cells_happy (board : Board) :
  ∀ i j : Fin 10, countBlueNeighbors board i j = 2 := by
  sorry

#check all_cells_happy

end all_cells_happy_l3687_368733


namespace circle_symmetry_l3687_368722

/-- Given two circles and a line of symmetry, prove the value of a parameter -/
theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ 
    ∃ x' y' : ℝ, x'^2 + y'^2 = 1 ∧ 
    (x + x')/2 - (y + y')/2 = 1 ∧
    (x - x')^2 + (y - y')^2 = ((x + x')/2 - x)^2 + ((y + y')/2 - y)^2) →
  a = 2 := by
sorry

end circle_symmetry_l3687_368722


namespace computer_price_proof_l3687_368785

theorem computer_price_proof (P : ℝ) 
  (h1 : 1.3 * P = 364)
  (h2 : 2 * P = 560) : 
  P = 280 := by sorry

end computer_price_proof_l3687_368785


namespace range_of_H_l3687_368788

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ y ∈ Set.Icc (-4) 4 := by sorry

end range_of_H_l3687_368788


namespace original_fraction_is_two_thirds_l3687_368727

theorem original_fraction_is_two_thirds :
  ∀ (a b : ℕ), 
    a ≠ 0 → b ≠ 0 →
    (a^3 : ℚ) / (b + 3 : ℚ) = 2 * (a : ℚ) / (b : ℚ) →
    (∀ d : ℕ, d ≠ 0 → d ∣ a ∧ d ∣ b → d = 1) →
    a = 2 ∧ b = 3 := by
  sorry

end original_fraction_is_two_thirds_l3687_368727


namespace correct_statement_l3687_368777

-- Define propositions P and Q
def P : Prop := Real.pi < 2
def Q : Prop := Real.pi > 3

-- Theorem statement
theorem correct_statement :
  (P ∨ Q) ∧ (¬P) := by sorry

end correct_statement_l3687_368777


namespace unique_q_value_l3687_368797

theorem unique_q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p*q = 4) : q = 2 := by
  sorry

end unique_q_value_l3687_368797


namespace discount_percentage_l3687_368796

def ticket_price : ℝ := 25
def sale_price : ℝ := 18.75

theorem discount_percentage : 
  (ticket_price - sale_price) / ticket_price * 100 = 25 := by
sorry

end discount_percentage_l3687_368796


namespace sum_of_coefficients_l3687_368718

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end sum_of_coefficients_l3687_368718


namespace shirt_price_proof_l3687_368771

theorem shirt_price_proof (P : ℝ) : 
  (0.75 * (0.75 * P) = 18) → P = 32 := by
  sorry

end shirt_price_proof_l3687_368771


namespace inverse_mod_59_l3687_368716

theorem inverse_mod_59 (h : (17⁻¹ : ZMod 59) = 23) : (42⁻¹ : ZMod 59) = 36 := by
  sorry

end inverse_mod_59_l3687_368716


namespace range_of_trig_function_l3687_368734

open Real

theorem range_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ 2 * cos x + sin (2 * x)
  ∃ (a b : ℝ), a = -3 * Real.sqrt 3 / 2 ∧ b = 3 * Real.sqrt 3 / 2 ∧
    (∀ x, f x ∈ Set.Icc a b) ∧
    (∀ y ∈ Set.Icc a b, ∃ x, f x = y) :=
by sorry

end range_of_trig_function_l3687_368734


namespace unique_factorization_2210_l3687_368766

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2210

theorem unique_factorization_2210 :
  ∃! p : ℕ × ℕ, valid_factorization p.1 p.2 ∧ p.1 ≤ p.2 :=
sorry

end unique_factorization_2210_l3687_368766


namespace triangle_inequality_inner_point_l3687_368762

/-- Given a triangle ABC and a point P on side AB, prove that PC · AB < PA · BC + PB · AC -/
theorem triangle_inequality_inner_point (A B C P : ℝ × ℝ) : 
  (P.1 > A.1 ∧ P.1 < B.1) → -- P is an inner point of AB
  (dist P C * dist A B < dist P A * dist B C + dist P B * dist A C) := by
  sorry

#check triangle_inequality_inner_point

end triangle_inequality_inner_point_l3687_368762


namespace inequality_generalization_l3687_368799

theorem inequality_generalization (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end inequality_generalization_l3687_368799


namespace masha_room_number_l3687_368741

theorem masha_room_number 
  (total_rooms : ℕ) 
  (masha_room : ℕ) 
  (alina_room : ℕ) 
  (h1 : total_rooms = 10000)
  (h2 : 1 ≤ masha_room ∧ masha_room < alina_room ∧ alina_room ≤ total_rooms)
  (h3 : masha_room + alina_room = 2022)
  (h4 : (((alina_room - masha_room - 1) * (masha_room + alina_room)) / 2) = 3033) :
  masha_room = 1009 := by
sorry

end masha_room_number_l3687_368741


namespace negative_power_equality_l3687_368731

theorem negative_power_equality : -2010^2011 = (-2010)^2011 := by sorry

end negative_power_equality_l3687_368731


namespace camp_wonka_ratio_l3687_368719

theorem camp_wonka_ratio : 
  ∀ (total_campers : ℕ) (boys girls : ℕ) (marshmallows : ℕ),
    total_campers = 96 →
    girls = total_campers / 3 →
    boys = total_campers - girls →
    marshmallows = 56 →
    (boys : ℚ) * (1/2) + (girls : ℚ) * (3/4) = marshmallows →
    (boys : ℚ) / (total_campers : ℚ) = 2/3 :=
by sorry

end camp_wonka_ratio_l3687_368719


namespace complex_exponential_sum_l3687_368715

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/2 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/2 : ℂ) - (1/3 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l3687_368715


namespace quadratic_inequality_range_l3687_368704

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a + 3 ≥ 0) ↔ a ∈ Set.Ici (0 : ℝ) :=
sorry

end quadratic_inequality_range_l3687_368704


namespace tax_reduction_theorem_l3687_368743

theorem tax_reduction_theorem (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 100) : 
  (1 - x / 100) * 1.1 = 0.825 → x = 25 := by
  sorry

end tax_reduction_theorem_l3687_368743


namespace seating_probability_l3687_368745

/-- Represents the number of delegates -/
def num_delegates : ℕ := 9

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 3

/-- Calculates the total number of seating arrangements -/
def total_arrangements : ℕ := (num_delegates.factorial) / ((delegates_per_country.factorial) ^ num_countries)

/-- Calculates the number of unwanted arrangements (where at least one country's delegates sit together) -/
def unwanted_arrangements : ℕ := 
  num_countries * num_delegates * ((num_delegates - delegates_per_country).factorial / ((delegates_per_country.factorial) ^ (num_countries - 1))) -
  (num_countries.choose 2) * num_delegates * (num_delegates - 2 * delegates_per_country + 1) +
  num_delegates * 2

/-- The probability that each delegate sits next to at least one delegate from another country -/
def probability : ℚ := (total_arrangements - unwanted_arrangements : ℚ) / total_arrangements

theorem seating_probability : probability = 41 / 56 := by
  sorry

end seating_probability_l3687_368745


namespace zoo_count_l3687_368751

/-- Counts the total number of animals observed during a zoo trip --/
def count_animals (snakes : ℕ) (arctic_foxes : ℕ) (leopards : ℕ) : ℕ :=
  let bee_eaters := 10 * (snakes / 2 + 2 * leopards)
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

/-- Theorem stating the total number of animals counted during the zoo trip --/
theorem zoo_count : count_animals 100 80 20 = 481340 := by
  sorry

end zoo_count_l3687_368751


namespace max_value_expression_l3687_368773

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 2 * y^2 + 2) ≤ Real.sqrt 29 := by
  sorry

end max_value_expression_l3687_368773


namespace minimum_value_theorem_l3687_368768

theorem minimum_value_theorem (m n : ℝ) (a : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, y = a^(x+2) - 2 ∧ y = -n/m * x - 1/m) →
  1/m + 1/n ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end minimum_value_theorem_l3687_368768


namespace sum_of_special_numbers_l3687_368783

theorem sum_of_special_numbers (a b c : ℤ) : 
  (∀ n : ℕ, n ≥ a) → 
  (∀ m : ℤ, m < 0 → m ≤ b) → 
  (c = -c) → 
  a + b + c = 0 :=
by sorry

end sum_of_special_numbers_l3687_368783


namespace initial_deck_size_l3687_368782

theorem initial_deck_size (red_cards : ℕ) (black_cards : ℕ) : 
  (red_cards : ℚ) / (red_cards + black_cards) = 1/3 →
  (red_cards : ℚ) / (red_cards + black_cards + 4) = 1/4 →
  red_cards + black_cards = 12 := by
  sorry

end initial_deck_size_l3687_368782


namespace simplify_expression_l3687_368706

theorem simplify_expression (y : ℝ) : 
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 := by
  sorry

end simplify_expression_l3687_368706


namespace water_formed_moles_l3687_368737

-- Define the chemical species
inductive ChemicalSpecies
| NaOH
| HCl
| H2O
| NaCl

-- Define a function to represent the stoichiometric coefficient in the balanced equation
def stoichiometric_coefficient (reactant product : ChemicalSpecies) : ℕ :=
  match reactant, product with
  | ChemicalSpecies.NaOH, ChemicalSpecies.H2O => 1
  | ChemicalSpecies.HCl, ChemicalSpecies.H2O => 1
  | _, _ => 0

-- Define the given amounts of reactants
def initial_NaOH : ℕ := 2
def initial_HCl : ℕ := 2

-- State the theorem
theorem water_formed_moles :
  min initial_NaOH initial_HCl = 
  stoichiometric_coefficient ChemicalSpecies.NaOH ChemicalSpecies.H2O * 
  stoichiometric_coefficient ChemicalSpecies.HCl ChemicalSpecies.H2O * 2 :=
by sorry

end water_formed_moles_l3687_368737


namespace positive_integer_expression_l3687_368793

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem positive_integer_expression (m n : ℕ) :
  ∃ k : ℕ+, k = (factorial (2 * m) * factorial (2 * n)) / (factorial m * factorial n * factorial (m + n)) :=
by sorry

end positive_integer_expression_l3687_368793


namespace dodecahedron_interior_diagonals_l3687_368756

/-- A dodecahedron is a 3-dimensional figure with specific properties -/
structure Dodecahedron where
  faces : ℕ
  vertices : ℕ
  faces_per_vertex : ℕ
  faces_are_pentagonal : Prop
  h_faces : faces = 12
  h_vertices : vertices = 20
  h_faces_per_vertex : faces_per_vertex = 3

/-- The number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - 1 - d.faces_per_vertex)) / 2

/-- Theorem stating the number of interior diagonals in a dodecahedron -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l3687_368756


namespace simple_interest_principal_l3687_368764

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℚ) (rate : ℚ) (time : ℚ) :
  interest = 8625 →
  rate = 50 / 3 →
  time = 3 / 4 →
  ∃ principal : ℚ, principal = 69000 ∧ interest = principal * rate * time / 100 := by
  sorry

end simple_interest_principal_l3687_368764


namespace simple_interest_problem_l3687_368705

theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 820 → 
  P + (P * R * 6) / 100 = 1020 → 
  P = 720 := by
  sorry

end simple_interest_problem_l3687_368705


namespace ellipse_cosine_theorem_l3687_368749

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let a := 2
  let c := Real.sqrt 3
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Define the distance ratio condition
def distance_ratio (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  d₁ = 3 * d₂

-- Theorem statement
theorem ellipse_cosine_theorem (P F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  point_on_ellipse P →
  distance_ratio P F₁ F₂ →
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let d₃ := Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)
  (d₁^2 + d₂^2 - d₃^2) / (2 * d₁ * d₂) = -1/3 :=
by sorry

end ellipse_cosine_theorem_l3687_368749


namespace max_xy_value_l3687_368784

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 2 * y = 110) : x * y ≤ 216 := by
  sorry

end max_xy_value_l3687_368784


namespace min_coach_handshakes_l3687_368735

/-- Represents the number of handshakes in a soccer tournament -/
def tournament_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  n.choose 2 + k

/-- Theorem stating the minimum number of coach handshakes -/
theorem min_coach_handshakes :
  ∃ (n : ℕ) (k : ℕ), tournament_handshakes n k = 406 ∧ k = 0 ∧ 
  ∀ (m : ℕ) (j : ℕ), tournament_handshakes m j = 406 → j ≥ k :=
by sorry

end min_coach_handshakes_l3687_368735


namespace first_day_duration_l3687_368726

def total_distance : ℝ := 115

def day2_distance : ℝ := 6 * 6 + 3 * 3

def day3_distance : ℝ := 7 * 5

def day1_speed : ℝ := 5

theorem first_day_duration : ∃ (hours : ℝ), 
  hours * day1_speed + day2_distance + day3_distance = total_distance ∧ hours = 7 := by
  sorry

end first_day_duration_l3687_368726


namespace revenue_is_432_l3687_368776

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  canoe_kayak_difference : ℕ

/-- Calculates the total revenue for the day -/
def total_revenue (rb : RentalBusiness) : ℕ :=
  let kayaks := rb.canoe_kayak_difference * 3
  let canoes := kayaks + rb.canoe_kayak_difference
  kayaks * rb.kayak_price + canoes * rb.canoe_price

/-- Theorem stating that the total revenue for the given scenario is $432 -/
theorem revenue_is_432 (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 9)
  (h2 : rb.kayak_price = 12)
  (h3 : rb.canoe_kayak_ratio = 4/3)
  (h4 : rb.canoe_kayak_difference = 6) :
  total_revenue rb = 432 := by
  sorry

#eval total_revenue { canoe_price := 9, kayak_price := 12, canoe_kayak_ratio := 4/3, canoe_kayak_difference := 6 }

end revenue_is_432_l3687_368776


namespace intersection_M_N_l3687_368779

def M : Set ℕ := {x | x > 0 ∧ x ≤ 2}
def N : Set ℕ := {2, 6}

theorem intersection_M_N : M ∩ N = {2} := by sorry

end intersection_M_N_l3687_368779


namespace decimal_to_binary_51_l3687_368746

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 51

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary representation of 51 is [1,1,0,0,1,1] -/
theorem decimal_to_binary_51 : toBinary decimalNumber = expectedBinary := by
  sorry

end decimal_to_binary_51_l3687_368746


namespace system_no_solution_l3687_368717

/-- The coefficient matrix of the system of equations -/
def A (n : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![n, 1, 1;
     1, n, 1;
     1, 1, n]

/-- The theorem stating that the system has no solution iff n = -2 -/
theorem system_no_solution (n : ℝ) :
  (∀ x y z : ℝ, n * x + y + z ≠ 1 ∨ x + n * y + z ≠ 1 ∨ x + y + n * z ≠ 1) ↔ n = -2 := by
  sorry

end system_no_solution_l3687_368717


namespace expression_undefined_at_13_l3687_368714

-- Define the numerator and denominator of the expression
def numerator (x : ℝ) : ℝ := 3 * x^3 - 5
def denominator (x : ℝ) : ℝ := x^2 - 26 * x + 169

-- Theorem stating that the expression is undefined when x = 13
theorem expression_undefined_at_13 : denominator 13 = 0 := by sorry

end expression_undefined_at_13_l3687_368714


namespace extra_bananas_l3687_368728

theorem extra_bananas (total_children absent_children original_bananas : ℕ) 
  (h1 : total_children = 840)
  (h2 : absent_children = 420)
  (h3 : original_bananas = 2) : 
  let present_children := total_children - absent_children
  let total_bananas := total_children * original_bananas
  let actual_bananas := total_bananas / present_children
  actual_bananas - original_bananas = 2 := by sorry

end extra_bananas_l3687_368728


namespace sum_of_bases_l3687_368792

-- Define the fractions F₁ and F₂
def F₁ (R : ℕ) : ℚ := (4 * R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5 * R + 4) / (R^2 - 1)

-- Define the conditions
def condition1 (R₁ : ℕ) : Prop := F₁ R₁ = 5 / 11
def condition2 (R₁ : ℕ) : Prop := F₂ R₁ = 6 / 11
def condition3 (R₂ : ℕ) : Prop := F₁ R₂ = 3 / 7
def condition4 (R₂ : ℕ) : Prop := F₂ R₂ = 4 / 7

-- State the theorem
theorem sum_of_bases (R₁ R₂ : ℕ) :
  condition1 R₁ → condition2 R₁ → condition3 R₂ → condition4 R₂ → R₁ + R₂ = 16 := by
  sorry

end sum_of_bases_l3687_368792


namespace pages_left_to_read_l3687_368744

theorem pages_left_to_read (total_pages read_pages : ℕ) : 
  total_pages = 17 → read_pages = 11 → total_pages - read_pages = 6 := by
  sorry

end pages_left_to_read_l3687_368744


namespace gas_cost_per_gallon_l3687_368790

/-- Calculates the cost of gas per gallon given Carla's trip details --/
theorem gas_cost_per_gallon 
  (distance_to_grocery : ℝ) 
  (distance_to_school : ℝ) 
  (distance_to_soccer : ℝ) 
  (miles_per_gallon : ℝ) 
  (total_gas_cost : ℝ) 
  (h1 : distance_to_grocery = 8) 
  (h2 : distance_to_school = 6) 
  (h3 : distance_to_soccer = 12) 
  (h4 : miles_per_gallon = 25) 
  (h5 : total_gas_cost = 5) :
  (total_gas_cost / ((distance_to_grocery + distance_to_school + distance_to_soccer + 2 * distance_to_soccer) / miles_per_gallon)) = 2.5 := by
sorry

end gas_cost_per_gallon_l3687_368790


namespace f_properties_l3687_368752

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ a then (1/a) * x
  else if a < x ∧ x ≤ 1 then (1/(1-a)) * (1-x)
  else 0

def is_turning_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (f x) = x ∧ f x ≠ x

theorem f_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (f (1/2) (f (1/2) (4/5)) = 4/5 ∧ is_turning_point (f (1/2)) (4/5)) ∧
  (∀ x : ℝ, a < x → x ≤ 1 →
    f a (f a x) = if x < a^2 - a + 1
                  then 1/(1-a) * (1 - 1/(1-a) * (1-x))
                  else 1/(a*(1-a)) * (1-x)) ∧
  (is_turning_point (f a) (1/(2-a)) ∧ is_turning_point (f a) (1/(1+a-a^2))) :=
by sorry

end

end f_properties_l3687_368752


namespace circle_radius_l3687_368775

/-- Given a circle and a line passing through its center, prove the radius is 3 -/
theorem circle_radius (m : ℝ) : 
  (∀ x y, x^2 + y^2 - 2*x + m*y - 4 = 0 → (x - 1)^2 + (y + m/2)^2 = 9) ∧ 
  (2 * 1 + (-m/2) = 0) →
  ∃ r, r = 3 ∧ ∀ x y, (x - 1)^2 + (y + m/2)^2 = r^2 → x^2 + y^2 - 2*x + m*y - 4 = 0 :=
by sorry

end circle_radius_l3687_368775


namespace complex_simplification_l3687_368700

theorem complex_simplification :
  (4 - 3 * Complex.I) - (6 - 5 * Complex.I) + (2 + 3 * Complex.I) = 5 * Complex.I := by
  sorry

end complex_simplification_l3687_368700


namespace fraction_simplification_l3687_368760

theorem fraction_simplification :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) := by
  sorry

end fraction_simplification_l3687_368760


namespace contrapositive_square_sum_zero_l3687_368710

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_square_sum_zero_l3687_368710


namespace valid_sequence_power_of_two_l3687_368778

/-- A sequence of pairwise distinct reals satisfying the given condition -/
def ValidSequence (N : ℕ) (a : ℕ → ℝ) : Prop :=
  N ≥ 3 ∧
  (∀ i j, i < N → j < N → i ≠ j → a i ≠ a j) ∧
  (∀ i, i < N → a i ≥ a ((2 * i) % N))

/-- The theorem stating that N must be a power of 2 -/
theorem valid_sequence_power_of_two (N : ℕ) (a : ℕ → ℝ) :
  ValidSequence N a → ∃ k : ℕ, N = 2^k :=
sorry

end valid_sequence_power_of_two_l3687_368778


namespace equation_solution_l3687_368767

theorem equation_solution : 
  ∃! x : ℝ, (3*x - 2 ≥ 0) ∧ (Real.sqrt (3*x - 2) + 9 / Real.sqrt (3*x - 2) = 6) ∧ (x = 11/3) :=
by sorry

end equation_solution_l3687_368767


namespace exam_disturbance_probability_l3687_368786

theorem exam_disturbance_probability :
  let n : ℕ := 6  -- number of students
  let p_undisturbed : ℚ := 2 / n * 2 / (n - 1) * 2 / (n - 2) * 2 / (n - 3)
  (1 : ℚ) - p_undisturbed = 43 / 45 :=
by sorry

end exam_disturbance_probability_l3687_368786


namespace inequality_system_solution_l3687_368757

theorem inequality_system_solution : 
  let S := {x : ℤ | x > 0 ∧ 5 + 3*x < 13 ∧ (x+2)/3 - (x-1)/2 ≤ 2}
  S = {1, 2} := by sorry

end inequality_system_solution_l3687_368757


namespace starting_number_is_24_l3687_368713

/-- Given that there are 35 even integers between a starting number and 95,
    prove that the starting number is 24. -/
theorem starting_number_is_24 (start : ℤ) : 
  (start < 95) →
  (∃ (evens : Finset ℤ), evens.card = 35 ∧ 
    (∀ n ∈ evens, start < n ∧ n < 95 ∧ Even n) ∧
    (∀ n, start < n ∧ n < 95 ∧ Even n → n ∈ evens)) →
  start = 24 := by
sorry

end starting_number_is_24_l3687_368713


namespace min_value_exponential_sum_l3687_368754

theorem min_value_exponential_sum (x : ℝ) : 16^x + 4^x - 2^x + 1 ≥ (3/4 : ℝ) := by
  sorry

end min_value_exponential_sum_l3687_368754


namespace triangle_inequality_proof_l3687_368761

theorem triangle_inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > 
  a^3 + b^3 + c^3 := by
sorry

end triangle_inequality_proof_l3687_368761


namespace product_bounds_l3687_368781

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ 1/4 + Real.sqrt 3/8 := by
  sorry

end product_bounds_l3687_368781


namespace probability_half_correct_l3687_368789

/-- The probability of getting exactly k successes in n trials with probability p for each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- The number of choices for each question -/
def num_choices : ℕ := 3

/-- The probability of guessing a question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The number of questions to get correct -/
def target_correct : ℕ := num_questions / 2

theorem probability_half_correct :
  binomial_probability num_questions target_correct prob_correct = 189399040 / 3486784401 := by
  sorry

end probability_half_correct_l3687_368789


namespace jim_age_proof_l3687_368712

/-- Calculates Jim's age X years from now -/
def jim_future_age (x : ℕ) : ℕ :=
  let tom_age_5_years_ago : ℕ := 32
  let years_since_tom_32 : ℕ := 5
  let years_to_past_reference : ℕ := 7
  let jim_age_difference : ℕ := 5
  27 + x

/-- Proves that Jim's age X years from now is (27 + X) -/
theorem jim_age_proof (x : ℕ) :
  jim_future_age x = 27 + x := by
  sorry

end jim_age_proof_l3687_368712


namespace total_ipods_l3687_368750

-- Define the initial number of iPods Emmy has
def emmy_initial : ℕ := 14

-- Define the number of iPods Emmy loses
def emmy_lost : ℕ := 6

-- Define Emmy's remaining iPods
def emmy_remaining : ℕ := emmy_initial - emmy_lost

-- Define Rosa's iPods in terms of Emmy's remaining
def rosa : ℕ := emmy_remaining / 2

-- Theorem to prove
theorem total_ipods : emmy_remaining + rosa = 12 := by
  sorry

end total_ipods_l3687_368750


namespace islander_liar_count_l3687_368759

/-- Represents the type of islander: either a knight or a liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  size : Nat
  statement : Nat

/-- The total number of islanders -/
def totalIslanders : Nat := 19

/-- The three groups of islanders making statements -/
def groups : List IslanderGroup := [
  { size := 3, statement := 3 },
  { size := 6, statement := 6 },
  { size := 9, statement := 9 }
]

/-- Determines if a statement is true given the actual number of liars -/
def isStatementTrue (statement : Nat) (actualLiars : Nat) : Bool :=
  statement == actualLiars

/-- Determines if an islander is telling the truth based on their type and statement -/
def isTellingTruth (type : IslanderType) (statementTrue : Bool) : Bool :=
  match type with
  | IslanderType.Knight => statementTrue
  | IslanderType.Liar => ¬statementTrue

/-- The main theorem to prove -/
theorem islander_liar_count :
  ∀ (liarCount : Nat),
  (liarCount ≤ totalIslanders) →
  (∀ (group : IslanderGroup),
    group ∈ groups →
    (∀ (type : IslanderType),
      (isTellingTruth type (isStatementTrue group.statement liarCount)) →
      (type = IslanderType.Knight ↔ liarCount = group.statement))) →
  (liarCount = 9 ∨ liarCount = 18 ∨ liarCount = 19) :=
sorry

end islander_liar_count_l3687_368759


namespace area_ratio_bounds_l3687_368774

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a line passing through the centroid
structure CentroidLine where
  angle : ℝ  -- Angle of the line with respect to a reference

-- Define the two parts created by the line
structure TriangleParts where
  part1 : ℝ
  part2 : ℝ
  parts_positive : part1 > 0 ∧ part2 > 0
  parts_sum : part1 + part2 = 1  -- Normalized to total area 1

-- Main theorem
theorem area_ratio_bounds (t : EquilateralTriangle) (l : CentroidLine) 
  (p : TriangleParts) : 
  4/5 ≤ min (p.part1 / p.part2) (p.part2 / p.part1) ∧ 
  max (p.part1 / p.part2) (p.part2 / p.part1) ≤ 5/4 :=
sorry

end area_ratio_bounds_l3687_368774


namespace statement_D_not_always_true_l3687_368798

-- Define the space
variable (Space : Type)

-- Define lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the specific lines and planes
variable (a b c : Line)
variable (α β : Plane)

-- State the theorem
theorem statement_D_not_always_true :
  ¬(∀ (b c : Line) (α : Plane),
    (subset b α ∧ ¬subset c α ∧ parallel_line_plane c α) → parallel b c) :=
by sorry

end statement_D_not_always_true_l3687_368798


namespace special_function_sum_property_l3687_368770

/-- A function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ x y, x ∈ ({x | x < -1} ∪ {x | x > 1}) → 
          y ∈ ({x | x < -1} ∪ {x | x > 1}) → 
          f (1/x) + f (1/y) = f ((x+y)/(1+x*y))) ∧
  (∀ x, x ∈ {x | -1 < x ∧ x < 0} → f x > 0)

/-- The theorem to be proved -/
theorem special_function_sum_property (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∑' (n : ℕ), f (1 / (n^2 + 7*n + 11)) > f (1/2) := by
  sorry

end special_function_sum_property_l3687_368770


namespace max_consecutive_sum_l3687_368763

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (n : ℕ) : ℤ := n * (2 * a + n - 1) / 2

/-- The target sum -/
def target_sum : ℕ := 528

/-- The maximum number of consecutive integers summing to the target -/
def max_consecutive : ℕ := 1056

theorem max_consecutive_sum :
  (∃ a : ℤ, sum_consecutive a max_consecutive = target_sum) ∧
  (∀ n : ℕ, n > max_consecutive → ¬∃ a : ℤ, sum_consecutive a n = target_sum) :=
sorry

end max_consecutive_sum_l3687_368763


namespace go_stones_theorem_l3687_368701

/-- Represents a stone on the grid -/
inductive Stone
| Black
| White

/-- Represents the grid configuration -/
def Grid (n : ℕ) := Fin (2*n) → Fin (2*n) → Option Stone

/-- Predicate to check if a stone exists at a given position -/
def has_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  ∃ (s : Stone), grid i j = some s

/-- Predicate to check if a black stone exists at a given position -/
def has_black_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  grid i j = some Stone.Black

/-- Predicate to check if a white stone exists at a given position -/
def has_white_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  grid i j = some Stone.White

/-- The grid after removing black stones that share a column with any white stone -/
def remove_black_stones (grid : Grid n) : Grid n :=
  sorry

/-- The grid after removing white stones that share a row with any remaining black stone -/
def remove_white_stones (grid : Grid n) : Grid n :=
  sorry

/-- Count the number of stones of a given type in the grid -/
def count_stones (grid : Grid n) (stone_type : Stone) : ℕ :=
  sorry

theorem go_stones_theorem (n : ℕ) (initial_grid : Grid n) :
  let final_grid := remove_white_stones (remove_black_stones initial_grid)
  (count_stones final_grid Stone.Black ≤ n^2) ∨ (count_stones final_grid Stone.White ≤ n^2) :=
sorry

end go_stones_theorem_l3687_368701


namespace student_square_substitution_l3687_368780

theorem student_square_substitution (a b : ℕ) : 
  (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9 → a = 3 ∧ ∀ n : ℕ, (3 + 2 * n - 3)^2 = 3^2 + 4 * n^2 - 9 :=
by sorry

end student_square_substitution_l3687_368780


namespace cookie_jar_solution_l3687_368739

def cookie_jar_problem (initial_amount doris_spent martha_spent remaining : ℕ) : Prop :=
  doris_spent = 6 ∧
  martha_spent = doris_spent / 2 ∧
  remaining = 15 ∧
  initial_amount = doris_spent + martha_spent + remaining

theorem cookie_jar_solution :
  ∃ initial_amount doris_spent martha_spent remaining,
    cookie_jar_problem initial_amount doris_spent martha_spent remaining ∧
    initial_amount = 24 := by sorry

end cookie_jar_solution_l3687_368739
