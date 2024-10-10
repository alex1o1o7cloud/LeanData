import Mathlib

namespace unique_prime_satisfying_conditions_l2895_289507

theorem unique_prime_satisfying_conditions : 
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    ∃ x y : ℕ, 
      x > 0 ∧ 
      y > 0 ∧ 
      p - 1 = 2 * x^2 ∧ 
      p^2 - 1 = 2 * y^2 := by
  sorry

end unique_prime_satisfying_conditions_l2895_289507


namespace projection_onto_yOz_plane_l2895_289598

-- Define the types for points and vectors in 3D space
def Point3D := ℝ × ℝ × ℝ
def Vector3D := ℝ × ℝ × ℝ

-- Define the projection onto the yOz plane
def projectOntoYOZ (p : Point3D) : Point3D :=
  (0, p.2.1, p.2.2)

-- Define the vector from origin to a point
def vectorFromOrigin (p : Point3D) : Vector3D := p

-- Theorem statement
theorem projection_onto_yOz_plane (A : Point3D) (h : A = (1, 6, 2)) :
  vectorFromOrigin (projectOntoYOZ A) = (0, 6, 2) := by
  sorry

end projection_onto_yOz_plane_l2895_289598


namespace not_right_triangle_l2895_289529

theorem not_right_triangle (a b c : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 4) (hc : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) := by
  sorry

end not_right_triangle_l2895_289529


namespace perimeter_division_ratio_l2895_289544

/-- A square with a point M on its diagonal and a line passing through M -/
structure SquareWithDiagonalPoint where
  /-- Side length of the square -/
  s : ℝ
  /-- Point M divides the diagonal AC in the ratio AM : MC = 3 : 2 -/
  m_divides_diagonal : ℝ
  /-- Ratio of areas divided by the line passing through M -/
  area_ratio : ℝ × ℝ
  /-- Assumption that s > 0 -/
  s_pos : s > 0
  /-- Assumption that m_divides_diagonal is the ratio 3 : 2 -/
  m_divides_diagonal_eq : m_divides_diagonal = 3 / 5
  /-- Assumption that area_ratio is 9 : 11 -/
  area_ratio_eq : area_ratio = (9, 11)

/-- Theorem: The line divides the perimeter in the ratio 19 : 21 -/
theorem perimeter_division_ratio (sq : SquareWithDiagonalPoint) : 
  (19 : ℝ) / 21 = 19 / (19 + 21) := by sorry

end perimeter_division_ratio_l2895_289544


namespace alyssa_soccer_games_l2895_289526

/-- Represents the number of soccer games Alyssa participated in over three years -/
def total_games (this_year_in_person this_year_online last_year_in_person next_year_in_person next_year_online : ℕ) : ℕ :=
  this_year_in_person + this_year_online + last_year_in_person + next_year_in_person + next_year_online

/-- Theorem stating that Alyssa will participate in 57 soccer games over three years -/
theorem alyssa_soccer_games : 
  total_games 11 8 13 15 10 = 57 := by
  sorry

#check alyssa_soccer_games

end alyssa_soccer_games_l2895_289526


namespace floor_equality_sufficient_not_necessary_l2895_289553

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_equality_sufficient_not_necessary :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) :=
sorry

end floor_equality_sufficient_not_necessary_l2895_289553


namespace chocolate_cake_eggs_l2895_289503

/-- The number of eggs needed for each cheesecake -/
def eggs_per_cheesecake : ℕ := 8

/-- The number of additional eggs needed for 9 cheesecakes compared to 5 chocolate cakes -/
def additional_eggs : ℕ := 57

/-- The number of cheesecakes in the comparison -/
def num_cheesecakes : ℕ := 9

/-- The number of chocolate cakes in the comparison -/
def num_chocolate_cakes : ℕ := 5

/-- The number of eggs needed for each chocolate cake -/
def eggs_per_chocolate_cake : ℕ := 3

theorem chocolate_cake_eggs :
  eggs_per_chocolate_cake * num_chocolate_cakes = 
  eggs_per_cheesecake * num_cheesecakes - additional_eggs :=
by sorry

end chocolate_cake_eggs_l2895_289503


namespace unique_divisible_by_7_l2895_289588

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

theorem unique_divisible_by_7 : 
  (is_divisible_by_7 41126) ∧ 
  (∀ B : ℕ, B < 10 → B ≠ 1 → ¬(is_divisible_by_7 (40000 + 2000 * B + 100 * B + 26))) :=
by sorry

end unique_divisible_by_7_l2895_289588


namespace solution_satisfies_equations_l2895_289591

theorem solution_satisfies_equations :
  ∃ (j k : ℚ), 5 * j - 42 * k = 1 ∧ 2 * k - j = 3 ∧ j = -4 ∧ k = -1/2 := by
  sorry

end solution_satisfies_equations_l2895_289591


namespace new_man_weight_l2895_289555

/-- Given a boat with 15 men, if replacing a 75 kg man with a new man increases the average weight by 2 kg, then the weight of the new man is 105 kg. -/
theorem new_man_weight (n : ℕ) (w_old w_new : ℝ) (h1 : n = 15) (h2 : w_old = 75) 
  (h3 : (n * w_new - w_old) / n - (n * w_old - w_old) / n = 2) : w_new = 105 := by
  sorry

end new_man_weight_l2895_289555


namespace lines_perpendicular_to_plane_are_parallel_l2895_289502

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l2895_289502


namespace coat_drive_l2895_289574

theorem coat_drive (total_coats : ℕ) (elementary_coats : ℕ) (high_school_coats : ℕ) :
  total_coats = 9437 →
  elementary_coats = 2515 →
  high_school_coats = total_coats - elementary_coats →
  high_school_coats = 6922 :=
by
  sorry

end coat_drive_l2895_289574


namespace exists_good_board_bad_after_recolor_l2895_289579

/-- Represents a 6x6 board with two colors -/
def Board := Fin 6 → Fin 6 → Bool

/-- Checks if a cell has a neighboring cell of the same color -/
def has_same_color_neighbor (b : Board) (i j : Fin 6) : Prop :=
  (i > 0 ∧ b (i-1) j = b i j) ∨
  (i < 5 ∧ b (i+1) j = b i j) ∨
  (j > 0 ∧ b i (j-1) = b i j) ∨
  (j < 5 ∧ b i (j+1) = b i j)

/-- Checks if the board is good (each cell has a same-color neighbor) -/
def is_good (b : Board) : Prop :=
  ∀ i j, has_same_color_neighbor b i j

/-- Recolors a row of the board -/
def recolor_row (b : Board) (row : Fin 6) : Board :=
  λ i j => if i = row then ¬(b i j) else b i j

/-- Recolors a column of the board -/
def recolor_column (b : Board) (col : Fin 6) : Board :=
  λ i j => if j = col then ¬(b i j) else b i j

/-- The main theorem: there exists a good board that becomes bad after any row or column recoloring -/
theorem exists_good_board_bad_after_recolor :
  ∃ b : Board, is_good b ∧
    (∀ row, ¬(is_good (recolor_row b row))) ∧
    (∀ col, ¬(is_good (recolor_column b col))) :=
sorry

end exists_good_board_bad_after_recolor_l2895_289579


namespace total_cost_squat_rack_and_barbell_l2895_289540

/-- The total cost of a squat rack and a barbell, where the squat rack costs $2500
    and the barbell costs 1/10 as much as the squat rack, is $2750. -/
theorem total_cost_squat_rack_and_barbell :
  let squat_rack_cost : ℕ := 2500
  let barbell_cost : ℕ := squat_rack_cost / 10
  squat_rack_cost + barbell_cost = 2750 :=
by sorry

end total_cost_squat_rack_and_barbell_l2895_289540


namespace continuity_at_4_l2895_289571

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |f x - f 4| < ε :=
by sorry

end continuity_at_4_l2895_289571


namespace culprit_left_in_carriage_l2895_289535

theorem culprit_left_in_carriage :
  -- Condition 1
  (∀ P W : Prop, P ∨ W) →
  -- Condition 2
  (∀ A P : Prop, A → P) →
  -- Condition 3
  (∀ A K : Prop, (¬A ∧ ¬K) ∨ (A ∧ K)) →
  -- Condition 4
  (∀ K : Prop, K) →
  -- Conclusion
  (∀ P : Prop, P) :=
by sorry

end culprit_left_in_carriage_l2895_289535


namespace taxi_trip_distance_l2895_289563

/-- Calculates the trip distance given the initial fee, per-increment charge, increment distance, and total charge -/
def calculate_trip_distance (initial_fee : ℚ) (per_increment_charge : ℚ) (increment_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let num_increments := distance_charge / per_increment_charge
  num_increments * increment_distance

theorem taxi_trip_distance :
  let initial_fee : ℚ := 9/4  -- $2.25
  let per_increment_charge : ℚ := 1/4  -- $0.25
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let total_charge : ℚ := 9/2  -- $4.50
  calculate_trip_distance initial_fee per_increment_charge increment_distance total_charge = 18/5 := by
  sorry

end taxi_trip_distance_l2895_289563


namespace r_power_four_plus_inverse_l2895_289594

theorem r_power_four_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_power_four_plus_inverse_l2895_289594


namespace remove_one_gives_avg_seven_point_five_l2895_289518

def original_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def remove_element (lst : List Nat) (n : Nat) : List Nat :=
  lst.filter (· ≠ n)

def average (lst : List Nat) : Rat :=
  (lst.sum : Rat) / lst.length

theorem remove_one_gives_avg_seven_point_five :
  average (remove_element original_sequence 1) = 15/2 := by
  sorry

end remove_one_gives_avg_seven_point_five_l2895_289518


namespace base_12_5_equivalence_l2895_289538

def is_valid_base_12_digit (d : ℕ) : Prop := d < 12
def is_valid_base_5_digit (d : ℕ) : Prop := d < 5

def to_base_10_from_base_12 (a b : ℕ) : ℕ := 12 * a + b
def to_base_10_from_base_5 (b a : ℕ) : ℕ := 5 * b + a

theorem base_12_5_equivalence (a b : ℕ) :
  is_valid_base_12_digit a →
  is_valid_base_12_digit b →
  is_valid_base_5_digit a →
  is_valid_base_5_digit b →
  to_base_10_from_base_12 a b = to_base_10_from_base_5 b a →
  to_base_10_from_base_12 a b = 0 :=
by sorry

end base_12_5_equivalence_l2895_289538


namespace one_positive_number_l2895_289575

theorem one_positive_number (numbers : List ℝ := [3, -2.1, -1/2, 0, -9]) :
  (numbers.filter (λ x => x > 0)).length = 1 := by
  sorry

end one_positive_number_l2895_289575


namespace cos_100_in_terms_of_sin_80_l2895_289557

theorem cos_100_in_terms_of_sin_80 (a : ℝ) (h : Real.sin (80 * π / 180) = a) :
  Real.cos (100 * π / 180) = -Real.sqrt (1 - a^2) := by
  sorry

end cos_100_in_terms_of_sin_80_l2895_289557


namespace marble_arrangement_l2895_289545

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles --/
def yellow_marbles : ℕ := 17

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose (total_marbles + blue_marbles - 1) blue_marbles

/-- The theorem to be proved --/
theorem marble_arrangement :
  arrangement_count = 100947 ∧ arrangement_count % 1000 = 947 := by
  sorry


end marble_arrangement_l2895_289545


namespace jacket_pricing_l2895_289589

theorem jacket_pricing (x : ℝ) : x + 28 = 0.8 * (1.5 * x) → 
  (∃ (markup_percent : ℝ) (discount_percent : ℝ) (profit : ℝ),
    markup_percent = 0.5 ∧
    discount_percent = 0.2 ∧
    profit = 28 ∧
    x + profit = (1 - discount_percent) * (1 + markup_percent) * x) :=
by
  sorry

end jacket_pricing_l2895_289589


namespace certain_number_proof_l2895_289506

theorem certain_number_proof (w : ℝ) (x : ℝ) 
  (h1 : x = 13 * w / (1 - w)) 
  (h2 : w^2 = 1) : 
  x = -13/2 := by
  sorry

end certain_number_proof_l2895_289506


namespace problem_statement_l2895_289586

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + y * z + z * x ≠ 1)
  (h2 : (x^2 - 1) * (y^2 - 1) / (x * y) + (y^2 - 1) * (z^2 - 1) / (y * z) + (z^2 - 1) * (x^2 - 1) / (z * x) = 4) :
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 1) ∧
  (9 * (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z * (x * y + y * z + z * x)) :=
by sorry

end problem_statement_l2895_289586


namespace missing_fraction_sum_l2895_289543

theorem missing_fraction_sum (x : ℚ) : 
  (1/3 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-5/6 : ℚ) + x = 5/6 → x = 13/6 := by
  sorry

end missing_fraction_sum_l2895_289543


namespace absolute_value_of_w_l2895_289550

theorem absolute_value_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : 
  Complex.abs w = 2 := by
sorry

end absolute_value_of_w_l2895_289550


namespace notebooks_distributed_sang_woo_distribution_l2895_289583

theorem notebooks_distributed (initial_notebooks : ℕ) (initial_pencils : ℕ) 
  (remaining_total : ℕ) : ℕ :=
  let distributed_notebooks := 
    (initial_notebooks + initial_pencils - remaining_total) / 4
  distributed_notebooks

theorem sang_woo_distribution : notebooks_distributed 12 34 30 = 4 := by
  sorry

end notebooks_distributed_sang_woo_distribution_l2895_289583


namespace number_and_multiple_l2895_289536

theorem number_and_multiple (x k : ℝ) : x = -7.0 ∧ 3 * x = k * x - 7 → k = 2 := by
  sorry

end number_and_multiple_l2895_289536


namespace circles_externally_tangent_l2895_289533

-- Define the equation for the radii
def radius_equation (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Define the radii R and r as real numbers satisfying the equation
def R : ℝ := sorry
def r : ℝ := sorry

-- Define the distance between centers
def d : ℝ := 3

-- State the theorem
theorem circles_externally_tangent :
  radius_equation R ∧ radius_equation r ∧ R ≠ r ∧ d = 3 →
  d = R + r :=
sorry

end circles_externally_tangent_l2895_289533


namespace g_inequality_l2895_289585

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
axiom f_continuous : Continuous f
axiom f_property : ∀ m n : ℝ, Real.exp n * f m + Real.exp (2 * m) * f (n - m) = Real.exp m * f n
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- Define the function g
def g (x : ℝ) : ℝ :=
  if x < 1 then Real.exp (x - 1) * f (1 - x)
  else Real.exp (1 - x) * f (x - 1)

-- State the theorem to be proved
theorem g_inequality : g 2 < g (-1) := by
  sorry

end

end g_inequality_l2895_289585


namespace find_a_l2895_289511

-- Define the sets U and A as functions of a
def U (a : ℝ) : Set ℝ := {1, 3*a+5, a^2+1}
def A (a : ℝ) : Set ℝ := {1, a+1}

-- Define the complement of A in U
def C_U_A (a : ℝ) : Set ℝ := U a \ A a

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = {1, 3*a+5, a^2+1}) ∧ 
  (A a = {1, a+1}) ∧ 
  (C_U_A a = {5}) ∧ 
  (a = -2) := by
  sorry

end find_a_l2895_289511


namespace equation_solutions_l2895_289576

theorem equation_solutions : 
  {x : ℝ | (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 36} = {3, 4} := by
sorry

end equation_solutions_l2895_289576


namespace equation_solutions_l2895_289572

theorem equation_solutions :
  (∀ x : ℝ, (1/2 * x^2 - 8 = 0) ↔ (x = 4 ∨ x = -4)) ∧
  (∀ x : ℝ, ((x - 5)^3 = -27) ↔ (x = 2)) :=
by sorry

end equation_solutions_l2895_289572


namespace cube_sum_geq_triple_product_l2895_289537

theorem cube_sum_geq_triple_product (x y z : ℝ) : x^3 + y^3 + z^3 ≥ 3*x*y*z := by
  sorry

end cube_sum_geq_triple_product_l2895_289537


namespace polyhedron_edge_face_relation_l2895_289532

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  face_counts : ℕ → ℕ
  convex : True

/-- The sum of k * f_k for all k ≥ 3 equals twice the number of edges -/
theorem polyhedron_edge_face_relation (P : ConvexPolyhedron) :
  2 * P.edges = ∑' k, k * P.face_counts k :=
sorry

end polyhedron_edge_face_relation_l2895_289532


namespace fish_population_estimate_l2895_289531

/-- Estimates the number of fish in a pond using the capture-recapture method. -/
def estimate_fish_population (initially_marked : ℕ) (second_sample : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initially_marked * second_sample) / marked_in_second

/-- Theorem stating that the estimated fish population is 500 given the specific conditions. -/
theorem fish_population_estimate :
  let initially_marked := 10
  let second_sample := 100
  let marked_in_second := 2
  estimate_fish_population initially_marked second_sample marked_in_second = 500 := by
  sorry

end fish_population_estimate_l2895_289531


namespace average_age_combined_l2895_289505

theorem average_age_combined (num_students : ℕ) (num_teachers : ℕ) 
  (avg_age_students : ℚ) (avg_age_teachers : ℚ) :
  num_students = 40 →
  num_teachers = 60 →
  avg_age_students = 13 →
  avg_age_teachers = 42 →
  ((num_students : ℚ) * avg_age_students + (num_teachers : ℚ) * avg_age_teachers) / 
   ((num_students : ℚ) + (num_teachers : ℚ)) = 30.4 := by
  sorry

end average_age_combined_l2895_289505


namespace jack_money_l2895_289568

theorem jack_money (jack ben eric : ℕ) : 
  (eric = ben - 10) → 
  (ben = jack - 9) → 
  (jack + ben + eric = 50) → 
  jack = 26 := by sorry

end jack_money_l2895_289568


namespace parking_spots_total_l2895_289516

/-- Calculates the total number of open parking spots in a 4-story parking area -/
def total_parking_spots (first_level : ℕ) (second_level_diff : ℕ) (third_level_diff : ℕ) (fourth_level : ℕ) : ℕ :=
  let second_level := first_level + second_level_diff
  let third_level := second_level + third_level_diff
  first_level + second_level + third_level + fourth_level

/-- Theorem: The total number of open parking spots is 46 -/
theorem parking_spots_total : 
  total_parking_spots 4 7 6 14 = 46 := by
  sorry

end parking_spots_total_l2895_289516


namespace a_5_equals_9_l2895_289552

-- Define the sequence a_n implicitly through S_n
def S (n : ℕ) : ℕ := n^2 - 1

-- Define a_n as the difference between consecutive S_n
def a (n : ℕ) : ℤ :=
  if n = 0 then 0
  else (S n : ℤ) - (S (n-1) : ℤ)

-- State the theorem
theorem a_5_equals_9 : a 5 = 9 := by sorry

end a_5_equals_9_l2895_289552


namespace train_passing_time_l2895_289523

/-- The time it takes for a train to pass a man running in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length > 0 →
  train_speed > man_speed →
  train_speed > 0 →
  man_speed ≥ 0 →
  ∃ (t : ℝ), t > 0 ∧ t < 22 ∧ t * (train_speed - man_speed) * (1000 / 3600) = train_length :=
by
  sorry

/-- Example with given values -/
example : 
  ∃ (t : ℝ), t > 0 ∧ t < 22 ∧ t * (68 - 8) * (1000 / 3600) = 350 :=
by
  apply train_passing_time 350 68 8
  · linarith
  · linarith
  · linarith
  · linarith

end train_passing_time_l2895_289523


namespace other_root_of_quadratic_l2895_289501

/-- Given a quadratic equation 5x^2 + kx = 8 with one root equal to 2, 
    prove that the other root is -4/5 -/
theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x = 8) ∧ (2 : ℝ) ∈ {x : ℝ | 5 * x^2 + k * x = 8} →
  (-4/5 : ℝ) ∈ {x : ℝ | 5 * x^2 + k * x = 8} :=
by sorry

end other_root_of_quadratic_l2895_289501


namespace arcsin_one_half_equals_pi_over_six_l2895_289542

theorem arcsin_one_half_equals_pi_over_six : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_equals_pi_over_six_l2895_289542


namespace pen_price_ratio_l2895_289581

/-- The ratio of gel pen price to ballpoint pen price -/
def gel_to_ballpoint_ratio : ℝ := 8

theorem pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
  x > 0 → y > 0 → b > 0 → g > 0 →
  (x + y : ℝ) * g = 4 * (x * b + y * g) →
  (x + y : ℝ) * b = (1 / 2) * (x * b + y * g) →
  g / b = gel_to_ballpoint_ratio :=
by sorry

end pen_price_ratio_l2895_289581


namespace jerry_collection_cost_l2895_289556

/-- Calculates the cost to complete an action figure collection -/
def cost_to_complete_collection (current : ℕ) (total_needed : ℕ) (cost_per_item : ℕ) : ℕ :=
  (total_needed - current) * cost_per_item

/-- Theorem: Jerry needs $216 to finish his collection -/
theorem jerry_collection_cost :
  let current := 9
  let total_needed := 27
  let cost_per_item := 12
  cost_to_complete_collection current total_needed cost_per_item = 216 := by
sorry

end jerry_collection_cost_l2895_289556


namespace partial_fraction_sum_zero_l2895_289558

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l2895_289558


namespace max_volume_rectangular_solid_l2895_289597

/-- Given a rectangular solid where the sum of all edges is 18 meters,
    and the length is twice the width, the maximum volume is 3 cubic meters. -/
theorem max_volume_rectangular_solid :
  ∃ (w l h : ℝ),
    w > 0 ∧ l > 0 ∧ h > 0 ∧
    l = 2 * w ∧
    4 * w + 4 * l + 4 * h = 18 ∧
    ∀ (w' l' h' : ℝ),
      w' > 0 → l' > 0 → h' > 0 →
      l' = 2 * w' →
      4 * w' + 4 * l' + 4 * h' = 18 →
      w * l * h ≥ w' * l' * h' ∧
    w * l * h = 3 :=
by sorry

end max_volume_rectangular_solid_l2895_289597


namespace median_is_165_l2895_289530

/-- Represents the size categories of school uniforms -/
inductive UniformSize
| s150
| s155
| s160
| s165
| s170
| s175
| s180

/-- Represents the frequency distribution of uniform sizes -/
def uniformDistribution : List (UniformSize × Nat) :=
  [(UniformSize.s150, 1),
   (UniformSize.s155, 6),
   (UniformSize.s160, 8),
   (UniformSize.s165, 12),
   (UniformSize.s170, 5),
   (UniformSize.s175, 4),
   (UniformSize.s180, 2)]

/-- Calculates the median size of school uniforms -/
def medianUniformSize (distribution : List (UniformSize × Nat)) : UniformSize :=
  sorry

/-- Theorem stating that the median uniform size is 165cm -/
theorem median_is_165 :
  medianUniformSize uniformDistribution = UniformSize.s165 := by
  sorry

end median_is_165_l2895_289530


namespace exponent_addition_l2895_289565

theorem exponent_addition (a : ℝ) : a^3 + a^3 = 2*a^3 := by
  sorry

end exponent_addition_l2895_289565


namespace father_and_sons_ages_l2895_289551

/-- Given a father and his three sons, prove that the ages of the middle and oldest sons are 3 and 4 years respectively. -/
theorem father_and_sons_ages (father_age : ℕ) (youngest_son_age : ℕ) (middle_son_age : ℕ) (oldest_son_age : ℕ) :
  father_age = 33 →
  youngest_son_age = 2 →
  father_age + 12 = (youngest_son_age + 12) + (middle_son_age + 12) + (oldest_son_age + 12) →
  (middle_son_age = 3 ∧ oldest_son_age = 4) ∨ (middle_son_age = 4 ∧ oldest_son_age = 3) :=
by sorry

end father_and_sons_ages_l2895_289551


namespace fertilizer_production_range_l2895_289541

/-- Represents the production capacity and constraints of a fertilizer plant --/
structure FertilizerPlant where
  maxWorkers : Nat
  maxHoursPerWorker : Nat
  minExpectedSales : Nat
  hoursPerBag : Nat
  rawMaterialPerBag : Nat
  initialRawMaterial : Nat
  usedRawMaterial : Nat
  supplementedRawMaterial : Nat

/-- Calculates the maximum number of bags that can be produced given the plant's constraints --/
def maxBagsProduced (plant : FertilizerPlant) : Nat :=
  min
    (plant.maxWorkers * plant.maxHoursPerWorker / plant.hoursPerBag)
    ((plant.initialRawMaterial - plant.usedRawMaterial + plant.supplementedRawMaterial) * 1000 / plant.rawMaterialPerBag)

/-- Theorem stating the range of bags that can be produced --/
theorem fertilizer_production_range (plant : FertilizerPlant)
  (h1 : plant.maxWorkers = 200)
  (h2 : plant.maxHoursPerWorker = 2100)
  (h3 : plant.minExpectedSales = 80000)
  (h4 : plant.hoursPerBag = 4)
  (h5 : plant.rawMaterialPerBag = 20)
  (h6 : plant.initialRawMaterial = 800)
  (h7 : plant.usedRawMaterial = 200)
  (h8 : plant.supplementedRawMaterial = 1200) :
  80000 ≤ maxBagsProduced plant ∧ maxBagsProduced plant = 90000 := by
  sorry

#check fertilizer_production_range

end fertilizer_production_range_l2895_289541


namespace gcd_1234_1987_l2895_289570

theorem gcd_1234_1987 : Int.gcd 1234 1987 = 1 := by
  sorry

end gcd_1234_1987_l2895_289570


namespace min_runs_ninth_game_l2895_289577

def runs_5_to_8 : List Nat := [19, 15, 13, 22]

theorem min_runs_ninth_game
  (h1 : (List.sum runs_5_to_8 + x) / 8 > (List.sum runs_5_to_8 + x) / 4)
  (h2 : (List.sum runs_5_to_8 + x + y) / 9 > 17)
  (h3 : y ≥ 19)
  : ∀ z < 19, (List.sum runs_5_to_8 + x + z) / 9 ≤ 17 :=
by sorry

#check min_runs_ninth_game

end min_runs_ninth_game_l2895_289577


namespace exists_abc_factorial_sum_l2895_289528

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem exists_abc_factorial_sum :
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    100 * a + 10 * b + c = factorial a + factorial b + factorial c :=
by sorry

end exists_abc_factorial_sum_l2895_289528


namespace age_difference_proof_l2895_289520

theorem age_difference_proof (A B n : ℚ) : 
  A = B + n →
  A - 2 = 6 * (B - 2) →
  A = 2 * B + 3 →
  n = 25 / 4 := by
sorry

#eval (25 : ℚ) / 4  -- To verify that 25/4 equals 6.25

end age_difference_proof_l2895_289520


namespace chime_2003_date_l2895_289592

/-- Represents a date with year, month, and day -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Represents a time with hour and minute -/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the number of chimes for a given hour -/
def hourChimes (hour : Nat) : Nat :=
  hour % 12

/-- Calculates the total number of chimes from a start date and time to an end date -/
def totalChimes (startDate : Date) (startTime : Time) (endDate : Date) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem chime_2003_date :
  let startDate := Date.mk 2003 2 28
  let startTime := Time.mk 15 15
  let endDate := Date.mk 2003 3 22
  totalChimes startDate startTime endDate = 2003 :=
sorry

end chime_2003_date_l2895_289592


namespace eighth_root_two_power_l2895_289519

theorem eighth_root_two_power (n : ℝ) : (8 : ℝ)^(1/3) = 2^n → n = 1 := by
  sorry

end eighth_root_two_power_l2895_289519


namespace triangle_interior_point_inequality_l2895_289590

open Real

variable (A B C M : ℝ × ℝ)

def isInside (M A B C : ℝ × ℝ) : Prop := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem triangle_interior_point_inequality 
  (h : isInside M A B C) :
  min (distance M A) (min (distance M B) (distance M C)) + 
  distance M A + distance M B + distance M C < 
  distance A B + distance B C + distance C A := by
  sorry

end triangle_interior_point_inequality_l2895_289590


namespace algebraic_substitution_l2895_289564

theorem algebraic_substitution (a b : ℝ) (h : a - 2 * b = 7) : 
  6 - 2 * a + 4 * b = -8 := by
sorry

end algebraic_substitution_l2895_289564


namespace sum_of_reciprocal_roots_for_given_polynomial_l2895_289580

/-- Given a quadratic polynomial ax^2 + bx + c, 
    returns the sum of the reciprocals of its roots -/
def sum_of_reciprocal_roots (a b c : ℚ) : ℚ := -b / c

theorem sum_of_reciprocal_roots_for_given_polynomial : 
  sum_of_reciprocal_roots 7 2 6 = -1/3 := by
  sorry

end sum_of_reciprocal_roots_for_given_polynomial_l2895_289580


namespace smallest_number_of_students_l2895_289510

theorem smallest_number_of_students (n : ℕ) : 
  (∃ x : ℕ, 
    n = 5 * x + 3 ∧ 
    n > 50 ∧ 
    ∀ m : ℕ, m > 50 → (∃ y : ℕ, m = 5 * y + 3) → m ≥ n) → 
  n = 53 := by
sorry

end smallest_number_of_students_l2895_289510


namespace tournament_points_l2895_289525

def number_of_teams : ℕ := 16
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def total_draws : ℕ := 30
def max_losses_per_team : ℕ := 2

def total_games : ℕ := number_of_teams * (number_of_teams - 1) / 2

theorem tournament_points :
  let total_wins : ℕ := total_games - total_draws
  let points_from_wins : ℕ := total_wins * points_for_win
  let points_from_draws : ℕ := total_draws * points_for_draw * 2
  points_from_wins + points_from_draws = 330 :=
by sorry

end tournament_points_l2895_289525


namespace necessary_but_not_sufficient_l2895_289567

def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem necessary_but_not_sufficient : 
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ b : ℝ, b ∈ M ∧ b ∉ N) :=
sorry

end necessary_but_not_sufficient_l2895_289567


namespace line_segment_endpoint_l2895_289534

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), 
    prove that the other endpoint is (-1, 5). -/
theorem line_segment_endpoint (M x₁ y₁ x₂ y₂ : ℝ) : 
  M = 3 ∧ 
  x₁ = 7 ∧ 
  y₁ = -3 ∧ 
  M = (x₁ + x₂) / 2 ∧ 
  1 = (y₁ + y₂) / 2 → 
  x₂ = -1 ∧ y₂ = 5 := by
sorry

end line_segment_endpoint_l2895_289534


namespace perpendicular_distance_is_four_l2895_289595

/-- A rectangular parallelepiped with vertices H, E, F, and G -/
structure Parallelepiped where
  H : ℝ × ℝ × ℝ
  E : ℝ × ℝ × ℝ
  F : ℝ × ℝ × ℝ
  G : ℝ × ℝ × ℝ

/-- The specific parallelepiped described in the problem -/
def specificParallelepiped : Parallelepiped :=
  { H := (0, 0, 0)
    E := (5, 0, 0)
    F := (0, 6, 0)
    G := (0, 0, 4) }

/-- The perpendicular distance from a point to a plane -/
noncomputable def perpendicularDistance (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The plane containing points E, F, and G -/
def planEFG (p : Parallelepiped) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The theorem to be proved -/
theorem perpendicular_distance_is_four :
  perpendicularDistance specificParallelepiped.H (planEFG specificParallelepiped) = 4 :=
by sorry

end perpendicular_distance_is_four_l2895_289595


namespace office_age_problem_l2895_289514

theorem office_age_problem (total_persons : Nat) (total_avg : ℚ) (group1_persons : Nat) 
  (group1_avg : ℚ) (person15_age : Nat) (group2_persons : Nat) :
  total_persons = 20 →
  total_avg = 15 →
  group1_persons = 5 →
  group1_avg = 14 →
  person15_age = 86 →
  group2_persons = 9 →
  let total_age : ℚ := total_persons * total_avg
  let group1_age : ℚ := group1_persons * group1_avg
  let remaining_age : ℚ := total_age - group1_age - person15_age
  let group2_age : ℚ := remaining_age - (total_persons - group1_persons - group2_persons - 1) * total_avg
  let group2_avg : ℚ := group2_age / group2_persons
  group2_avg = 23/3 := by sorry

end office_age_problem_l2895_289514


namespace plants_eaten_first_day_l2895_289515

theorem plants_eaten_first_day (total : ℕ) (remaining : ℕ) :
  total = 30 ∧ 
  remaining = 4 ∧
  (∃ x y : ℕ, x + y + remaining + 1 = total ∧ y = (x + y + 1) / 2) →
  x = 20 :=
by sorry

end plants_eaten_first_day_l2895_289515


namespace greatest_integer_solution_l2895_289546

theorem greatest_integer_solution (x : ℝ) : 
  x^3 = 7 - 2*x → 
  (∀ n : ℤ, n > (x - 2 : ℝ) → n ≤ 3) ∧ 
  (3 : ℝ) > (x - 2 : ℝ) := by
  sorry

end greatest_integer_solution_l2895_289546


namespace smallest_k_for_64_power_gt_4_power_19_l2895_289549

theorem smallest_k_for_64_power_gt_4_power_19 : 
  ∃ k : ℕ, (∀ m : ℕ, 64^m > 4^19 → k ≤ m) ∧ 64^k > 4^19 :=
by
  -- The proof goes here
  sorry

end smallest_k_for_64_power_gt_4_power_19_l2895_289549


namespace smallest_positive_solution_floor_equation_l2895_289500

theorem smallest_positive_solution_floor_equation :
  ∃ x : ℝ, x > 0 ∧
    (∀ y : ℝ, y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 25 → x ≤ y) ∧
    ⌊x^2⌋ - ⌊x⌋^2 = 25 ∧
    x = 7 * Real.sqrt 3 := by
  sorry

end smallest_positive_solution_floor_equation_l2895_289500


namespace infinite_solutions_iff_b_eq_neg_twelve_l2895_289578

theorem infinite_solutions_iff_b_eq_neg_twelve (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by sorry

end infinite_solutions_iff_b_eq_neg_twelve_l2895_289578


namespace intersection_of_M_and_N_l2895_289599

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 > 0}
def N : Set ℝ := {x | 2*x - 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = Set.Ioo (-1 : ℝ) (1/2 : ℝ) := by sorry

end intersection_of_M_and_N_l2895_289599


namespace vector_sum_magnitude_l2895_289559

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (1, -Real.sqrt 3) →
  Real.sqrt ((a.1 ^ 2 + a.2 ^ 2)) = 2 →
  Real.sqrt ((b.1 ^ 2 + b.2 ^ 2)) = 1 →
  a.1 * b.1 + a.2 * b.2 = -1 →
  Real.sqrt (((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2)) = Real.sqrt 3 := by
sorry

end vector_sum_magnitude_l2895_289559


namespace inscribed_cube_volume_l2895_289522

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_face_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : Pyramid
  covers_base : Bool
  touches_summit : Bool

/-- The volume of the inscribed cube -/
def cube_volume (cube : InscribedCube) : ℝ := sorry

theorem inscribed_cube_volume 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side = 2) 
  (h2 : cube.pyramid.lateral_face_equilateral = true) 
  (h3 : cube.covers_base = true) 
  (h4 : cube.touches_summit = true) : 
  cube_volume cube = 8 := by sorry

end inscribed_cube_volume_l2895_289522


namespace autumn_pencils_bought_l2895_289513

/-- Calculates the number of pencils Autumn bought given the initial number,
    misplaced pencils, broken pencils, found pencils, and final number of pencils. -/
def pencils_bought (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (final : ℕ) : ℕ :=
  final - (initial - misplaced - broken + found)

/-- Proves that Autumn bought 2 pencils given the specific scenario. -/
theorem autumn_pencils_bought :
  pencils_bought 20 7 3 4 16 = 2 := by
  sorry

#eval pencils_bought 20 7 3 4 16

end autumn_pencils_bought_l2895_289513


namespace sqrt_product_equals_200_l2895_289566

theorem sqrt_product_equals_200 : Real.sqrt 100 * Real.sqrt 50 * Real.sqrt 8 = 200 := by
  sorry

end sqrt_product_equals_200_l2895_289566


namespace flea_count_l2895_289554

/-- The total number of fleas on three chickens (Gertrude, Maud, and Olive) -/
def total_fleas (gertrude_fleas : ℕ) (olive_fleas : ℕ) (maud_fleas : ℕ) : ℕ :=
  gertrude_fleas + olive_fleas + maud_fleas

/-- Theorem stating the total number of fleas on the three chickens is 40 -/
theorem flea_count :
  ∀ (gertrude_fleas olive_fleas maud_fleas : ℕ),
  gertrude_fleas = 10 →
  olive_fleas = gertrude_fleas / 2 →
  maud_fleas = 5 * olive_fleas →
  total_fleas gertrude_fleas olive_fleas maud_fleas = 40 :=
by
  sorry

#check flea_count

end flea_count_l2895_289554


namespace greatest_prime_base_angle_l2895_289587

-- Define the triangle and its properties
def IsoscelesTriangle (a b c : ℕ) : Prop :=
  a = b ∧ a + b + c = 180 ∧ c = 60 ∧ a < 90 ∧ Nat.Prime a

-- State the theorem
theorem greatest_prime_base_angle :
  ∃ (a : ℕ), IsoscelesTriangle a a 60 ∧
  ∀ (x : ℕ), IsoscelesTriangle x x 60 → x ≤ a :=
by sorry

end greatest_prime_base_angle_l2895_289587


namespace brad_start_time_l2895_289560

/-- Proves that Brad started running 9 hours after Maxwell started walking -/
theorem brad_start_time (maxwell_speed : ℝ) (brad_speed : ℝ) (total_distance : ℝ) (maxwell_time : ℝ) :
  maxwell_speed = 4 →
  brad_speed = 6 →
  total_distance = 94 →
  maxwell_time = 10 →
  total_distance - maxwell_speed * maxwell_time = brad_speed * (maxwell_time - 9) :=
by
  sorry

#check brad_start_time

end brad_start_time_l2895_289560


namespace cookie_distribution_l2895_289527

theorem cookie_distribution (num_boxes : ℕ) (cookies_per_box : ℕ) (num_people : ℕ) :
  num_boxes = 7 →
  cookies_per_box = 10 →
  num_people = 5 →
  (num_boxes * cookies_per_box) / num_people = 14 :=
by sorry

end cookie_distribution_l2895_289527


namespace min_value_theorem_l2895_289561

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 2 → 2/x + 1/(y-1) ≥ 2/a + 1/(b-1)) →
  2/a + 1/(b-1) = 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_theorem_l2895_289561


namespace arithmetic_sequence_properties_l2895_289504

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -3
  condition : 11 * (a 5) = 5 * (a 8) - 13
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The common difference of the arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ := seq.a 2 - seq.a 1

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- The main theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (common_difference seq = 5/9) ∧
  (∃ n : ℕ, sum_n_terms seq n = -29/3 ∧ 
    ∀ m : ℕ, sum_n_terms seq m ≥ sum_n_terms seq n) :=
by
  sorry

end arithmetic_sequence_properties_l2895_289504


namespace two_heads_in_succession_probability_l2895_289569

-- Define a function to count sequences without two heads in succession
def g : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => g (n + 1) + g n

-- Theorem statement
theorem two_heads_in_succession_probability :
  (1024 - g 10 : ℚ) / 1024 = 55 / 64 := by sorry

end two_heads_in_succession_probability_l2895_289569


namespace ordering_abc_l2895_289539

def a : ℝ := (2 : ℝ) ^ (4/3)
def b : ℝ := (4 : ℝ) ^ (2/5)
def c : ℝ := (25 : ℝ) ^ (1/3)

theorem ordering_abc : b < a ∧ a < c := by sorry

end ordering_abc_l2895_289539


namespace uniform_price_uniform_price_is_75_l2895_289517

/-- Calculates the price of a uniform given the conditions of a servant's employment --/
theorem uniform_price (full_year_salary : ℕ) (months_worked : ℕ) (payment_received : ℕ) : ℕ :=
  let prorated_salary := full_year_salary * months_worked / 12
  prorated_salary - payment_received

/-- Proves that the price of the uniform is 75 given the specific conditions --/
theorem uniform_price_is_75 :
  uniform_price 500 9 300 = 75 := by
  sorry

end uniform_price_uniform_price_is_75_l2895_289517


namespace min_value_product_l2895_289584

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / b + b / c + c / a + b / a + c / b + a / c = 7) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 35 / 2 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    a₀ / b₀ + b₀ / c₀ + c₀ / a₀ + b₀ / a₀ + c₀ / b₀ + a₀ / c₀ = 7 ∧
    (a₀ / b₀ + b₀ / c₀ + c₀ / a₀) * (b₀ / a₀ + c₀ / b₀ + a₀ / c₀) = 35 / 2 :=
by sorry

end min_value_product_l2895_289584


namespace ellipse_foci_distance_l2895_289509

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 36 is 8√2 -/
theorem ellipse_foci_distance : 
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, dist p f₁ + dist p f₂ = 2 * 6 ∧
    dist f₁ f₂ = 8 * Real.sqrt 2 :=
by sorry

end ellipse_foci_distance_l2895_289509


namespace triangle_problem_l2895_289596

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > c →
  b = 3 →
  (a * c : ℝ) * (1 / 3 : ℝ) = 2 →  -- This represents BA · BC = 2 and cos B = 1/3
  a^2 + c^2 = b^2 + 2 * (a * c : ℝ) * (1 / 3 : ℝ) →  -- Law of cosines
  (a = 3 ∧ c = 2) ∧ 
  Real.cos (B - C) = 23 / 27 := by
sorry

end triangle_problem_l2895_289596


namespace ceilings_left_to_paint_l2895_289524

theorem ceilings_left_to_paint (total : ℕ) (this_week : ℕ) (next_week_fraction : ℚ) : 
  total = 28 → 
  this_week = 12 → 
  next_week_fraction = 1/4 →
  total - (this_week + next_week_fraction * this_week) = 13 := by
  sorry

end ceilings_left_to_paint_l2895_289524


namespace investment_time_ratio_l2895_289512

/-- Represents the business investment scenario of Krishan and Nandan -/
structure Investment where
  nandan_amount : ℝ
  nandan_time : ℝ
  krishan_time_ratio : ℝ
  nandan_gain : ℝ
  total_gain : ℝ

/-- The conditions of the investment scenario -/
def investment_conditions (i : Investment) : Prop :=
  i.nandan_gain = i.nandan_amount * i.nandan_time ∧
  i.total_gain = i.nandan_gain + 6 * i.nandan_amount * i.krishan_time_ratio * i.nandan_time ∧
  i.nandan_gain = 6000 ∧
  i.total_gain = 78000

/-- The theorem stating that under the given conditions, 
    Krishan's investment time is twice that of Nandan's -/
theorem investment_time_ratio 
  (i : Investment) 
  (h : investment_conditions i) : 
  i.krishan_time_ratio = 2 := by
  sorry

end investment_time_ratio_l2895_289512


namespace martha_centerpiece_cost_l2895_289521

/-- Calculates the total cost of flowers for centerpieces -/
def total_flower_cost (num_centerpieces : ℕ) (roses_per_centerpiece : ℕ) 
  (orchids_per_centerpiece : ℕ) (lilies_per_centerpiece : ℕ) (cost_per_flower : ℕ) : ℕ :=
  num_centerpieces * (roses_per_centerpiece + orchids_per_centerpiece + lilies_per_centerpiece) * cost_per_flower

/-- Theorem: The total cost for flowers for 6 centerpieces is $2700 -/
theorem martha_centerpiece_cost : 
  total_flower_cost 6 8 16 6 15 = 2700 := by
  sorry

end martha_centerpiece_cost_l2895_289521


namespace driver_net_pay_rate_driver_net_pay_rate_example_l2895_289573

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) 
  (fuel_cost : ℝ) (maintenance_threshold : ℝ) (maintenance_cost : ℝ) : ℝ :=
  let distance := hours * speed
  let fuel_used := distance / fuel_efficiency
  let earnings := distance * pay_per_mile
  let fuel_expense := fuel_used * fuel_cost
  let maintenance_expense := if distance > maintenance_threshold then maintenance_cost else 0
  let total_expense := fuel_expense + maintenance_expense
  let net_earnings := earnings - total_expense
  let net_rate := net_earnings / hours
  net_rate

/-- The driver's net rate of pay is approximately 21.67 dollars per hour --/
theorem driver_net_pay_rate_example : 
  ∃ ε > 0, |driver_net_pay_rate 3 50 25 0.60 2.50 100 10 - 21.67| < ε :=
sorry

end driver_net_pay_rate_driver_net_pay_rate_example_l2895_289573


namespace no_common_tangent_for_three_circles_l2895_289508

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents a configuration of three circles -/
structure ThreeCircleConfig where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  h12 : are_externally_tangent c1 c2
  h23 : are_externally_tangent c2 c3
  h31 : are_externally_tangent c3 c1

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to a circle -/
def is_tangent_to (l : Line) (c : Circle) : Prop :=
  let (x, y) := c.center
  (l.a * x + l.b * y + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

/-- The main theorem -/
theorem no_common_tangent_for_three_circles (config : ThreeCircleConfig) 
    (h1 : config.c1.radius = 3)
    (h2 : config.c2.radius = 4)
    (h3 : config.c3.radius = 5) :
  ¬∃ (l : Line), is_tangent_to l config.c1 ∧ is_tangent_to l config.c2 ∧ is_tangent_to l config.c3 :=
by sorry

end no_common_tangent_for_three_circles_l2895_289508


namespace factorial_10_mod_13_l2895_289547

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/-- The remainder when 10! is divided by 13 is 7 -/
theorem factorial_10_mod_13 : factorial 10 % 13 = 7 := by
  sorry

end factorial_10_mod_13_l2895_289547


namespace segment_ratio_vector_coefficients_l2895_289593

-- Define the vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points C, D, and Q
variable (C D Q : V)

-- Define the condition that Q is on the line segment CD with the given ratio
def on_segment_with_ratio (C D Q : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D ∧ t = 5 / 8

-- Theorem statement
theorem segment_ratio_vector_coefficients
  (h : on_segment_with_ratio C D Q) :
  ∃ (s v : ℝ), Q = s • C + v • D ∧ s = 5/8 ∧ v = 3/8 :=
sorry

end segment_ratio_vector_coefficients_l2895_289593


namespace dolls_count_l2895_289548

theorem dolls_count (total_toys : ℕ) (action_figure_fraction : ℚ) : 
  total_toys = 24 → action_figure_fraction = 1/4 → 
  total_toys - (total_toys * action_figure_fraction).floor = 18 := by
sorry

end dolls_count_l2895_289548


namespace cricket_run_rate_theorem_l2895_289582

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.targetRuns - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.2)
  (h4 : game.targetRuns = 252) :
  requiredRunRate game = 5.5 := by
  sorry

end cricket_run_rate_theorem_l2895_289582


namespace systematic_sampling_theorem_l2895_289562

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  interval : Nat

/-- Checks if a student number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = k * s.interval + (n % s.interval)

/-- Main theorem: If students 6, 32, and 45 are in a systematic sample of 4 from 52 students,
    then student 19 must also be in the sample -/
theorem systematic_sampling_theorem (s : SystematicSample) 
  (h1 : s.total_students = 52)
  (h2 : s.sample_size = 4)
  (h3 : in_sample s 6)
  (h4 : in_sample s 32)
  (h5 : in_sample s 45) :
  in_sample s 19 := by
  sorry

end systematic_sampling_theorem_l2895_289562
