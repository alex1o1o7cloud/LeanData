import Mathlib

namespace cos_double_angle_proof_l1025_102553

theorem cos_double_angle_proof (α : ℝ) (a : ℝ × ℝ) : 
  a = (Real.cos α, Real.sqrt 2 / 2) → 
  Real.sqrt ((a.1)^2 + (a.2)^2) = Real.sqrt 3 / 2 → 
  Real.cos (2 * α) = -1/2 := by
  sorry

end cos_double_angle_proof_l1025_102553


namespace don_bottles_from_shop_C_l1025_102502

/-- The number of bottles Don buys from Shop A -/
def bottles_from_A : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def bottles_from_B : ℕ := 180

/-- The total number of bottles Don is capable of buying -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop C -/
def bottles_from_C : ℕ := total_bottles - (bottles_from_A + bottles_from_B)

theorem don_bottles_from_shop_C :
  bottles_from_C = 220 :=
by sorry

end don_bottles_from_shop_C_l1025_102502


namespace two_sets_satisfying_union_condition_l1025_102579

theorem two_sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    S.card = 2 ∧ 
    ∀ M ∈ S, M ∪ {1} = {1, 2, 3} ∧
    ∀ M, M ∪ {1} = {1, 2, 3} → M ∈ S :=
by sorry

end two_sets_satisfying_union_condition_l1025_102579


namespace cyclic_permutation_sum_equality_l1025_102595

def is_cyclic_shift (a : Fin n → ℕ) : Prop :=
  ∃ i, ∀ j, a j = ((j.val + i - 1) % n) + 1

def is_permutation (b : Fin n → ℕ) : Prop :=
  Function.Bijective b ∧ ∀ i, b i ≤ n

theorem cyclic_permutation_sum_equality (n : ℕ) :
  (∃ (a b : Fin n → ℕ),
    is_cyclic_shift a ∧
    is_permutation b ∧
    ∀ i j : Fin n, i.val + 1 + a i + b i = j.val + 1 + a j + b j) ↔
  Odd n :=
sorry

end cyclic_permutation_sum_equality_l1025_102595


namespace max_value_sqrt_sum_l1025_102550

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l1025_102550


namespace distribute_five_objects_l1025_102522

/-- The number of ways to distribute n distinguishable objects into 2 indistinguishable containers,
    such that neither container is empty -/
def distribute (n : ℕ) : ℕ :=
  (2^n - 2) / 2

/-- Theorem: There are 15 ways to distribute 5 distinguishable objects into 2 indistinguishable containers,
    such that neither container is empty -/
theorem distribute_five_objects : distribute 5 = 15 := by
  sorry

end distribute_five_objects_l1025_102522


namespace angle_with_special_supplement_complement_l1025_102507

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  (0 < x) ∧ (x < 180) ∧ (180 - x = 4 * (90 - x)) ∧ (x = 60) := by
  sorry

end angle_with_special_supplement_complement_l1025_102507


namespace surfer_ratio_l1025_102580

/-- Proves that the ratio of surfers on Malibu beach to Santa Monica beach is 2:1 -/
theorem surfer_ratio :
  ∀ (malibu santa_monica : ℕ),
  santa_monica = 20 →
  malibu + santa_monica = 60 →
  (malibu : ℚ) / santa_monica = 2 / 1 := by
sorry

end surfer_ratio_l1025_102580


namespace rectangle_folding_l1025_102554

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the folding properties of the rectangle -/
structure FoldedRectangle extends Rectangle where
  pointE : Point
  pointF : Point
  coincideOnDiagonal : Bool

/-- The main theorem statement -/
theorem rectangle_folding (rect : FoldedRectangle) (k m : ℕ) :
  rect.width = 2 ∧ 
  rect.height = 1 ∧
  rect.pointE.x = rect.width - rect.pointF.x ∧
  rect.coincideOnDiagonal = true ∧
  Real.sqrt k - m = rect.pointE.x
  → k + m = 14 := by
  sorry

end rectangle_folding_l1025_102554


namespace problem_solution_l1025_102582

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1/5) := by
  sorry

end problem_solution_l1025_102582


namespace rotten_bananas_percentage_l1025_102591

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 886 / 1000)
  : (total_bananas - (good_fruits_percentage * (total_oranges + total_bananas) - (1 - rotten_oranges_percentage) * total_oranges)) / total_bananas = 6 / 100 := by
  sorry

end rotten_bananas_percentage_l1025_102591


namespace average_speed_calculation_l1025_102503

-- Define the distance traveled in the first and second hours
def distance_first_hour : ℝ := 98
def distance_second_hour : ℝ := 70

-- Define the total time
def total_time : ℝ := 2

-- Theorem statement
theorem average_speed_calculation :
  let total_distance := distance_first_hour + distance_second_hour
  (total_distance / total_time) = 84 := by sorry

end average_speed_calculation_l1025_102503


namespace contrapositive_equivalence_l1025_102519

theorem contrapositive_equivalence (p q : Prop) :
  (¬p → q) → (¬q → p) := by sorry

end contrapositive_equivalence_l1025_102519


namespace sine_inequality_l1025_102516

theorem sine_inequality (n : ℕ) (x : ℝ) : 
  Real.sin x * (n * Real.sin x - Real.sin (n * x)) ≥ 0 := by
  sorry

end sine_inequality_l1025_102516


namespace smallest_number_satisfying_conditions_l1025_102581

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def leaves_remainder_1 (n : ℕ) (d : ℕ) : Prop := n % d = 1

theorem smallest_number_satisfying_conditions : 
  (∀ d : ℕ, 2 ≤ d → d ≤ 8 → leaves_remainder_1 6721 d) ∧ 
  is_divisible_by_11 6721 ∧
  (∀ m : ℕ, m < 6721 → 
    (¬(∀ d : ℕ, 2 ≤ d → d ≤ 8 → leaves_remainder_1 m d) ∨ 
     ¬(is_divisible_by_11 m))) :=
by sorry

end smallest_number_satisfying_conditions_l1025_102581


namespace clowns_in_mobiles_l1025_102589

/-- Given a number of clown mobiles and a total number of clowns,
    calculate the number of clowns in each mobile assuming even distribution -/
def clowns_per_mobile (num_mobiles : ℕ) (total_clowns : ℕ) : ℕ :=
  total_clowns / num_mobiles

/-- Theorem stating that with 5 clown mobiles and 140 clowns in total,
    there are 28 clowns in each mobile -/
theorem clowns_in_mobiles :
  clowns_per_mobile 5 140 = 28 := by
  sorry


end clowns_in_mobiles_l1025_102589


namespace quadratic_roots_sum_powers_l1025_102531

theorem quadratic_roots_sum_powers (t q : ℝ) (a₁ a₂ : ℝ) : 
  (∀ x : ℝ, x^2 - t*x + q = 0 ↔ x = a₁ ∨ x = a₂) →
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1003 → a₁^n + a₂^n = a₁ + a₂) →
  a₁^1004 + a₂^1004 = 2 := by
sorry

end quadratic_roots_sum_powers_l1025_102531


namespace min_value_of_difference_l1025_102504

theorem min_value_of_difference (x y z : ℝ) : 
  0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 4 →
  y^2 - x^2 = 2 →
  z^2 - y^2 = 2 →
  (2 * (2 - Real.sqrt 3) : ℝ) ≤ |x - y| + |y - z| ∧ |x - y| + |y - z| ≤ 2 :=
by sorry

end min_value_of_difference_l1025_102504


namespace max_pieces_on_chessboard_l1025_102535

/-- Represents a chessboard with red and blue pieces -/
structure Chessboard :=
  (size : Nat)
  (red_pieces : Finset (Nat × Nat))
  (blue_pieces : Finset (Nat × Nat))

/-- Counts the number of pieces of the opposite color that a piece can see -/
def count_opposite_color (board : Chessboard) (pos : Nat × Nat) (is_red : Bool) : Nat :=
  sorry

/-- Checks if the chessboard configuration is valid -/
def is_valid_configuration (board : Chessboard) : Prop :=
  board.size = 200 ∧
  (∀ pos ∈ board.red_pieces, count_opposite_color board pos true = 5) ∧
  (∀ pos ∈ board.blue_pieces, count_opposite_color board pos false = 5)

/-- The main theorem stating the maximum number of pieces on the chessboard -/
theorem max_pieces_on_chessboard (board : Chessboard) :
  is_valid_configuration board →
  Finset.card board.red_pieces + Finset.card board.blue_pieces ≤ 3800 :=
sorry

end max_pieces_on_chessboard_l1025_102535


namespace modulus_range_of_complex_l1025_102546

theorem modulus_range_of_complex (Z : ℂ) (a : ℝ) (h1 : 0 < a) (h2 : a < 2) 
  (h3 : Z.re = a) (h4 : Z.im = 1) : 1 < Complex.abs Z ∧ Complex.abs Z < Real.sqrt 5 := by
  sorry

end modulus_range_of_complex_l1025_102546


namespace regular_pentagon_angle_excess_prove_regular_pentagon_angle_excess_l1025_102532

theorem regular_pentagon_angle_excess : ℝ → Prop :=
  λ total_excess : ℝ =>
    -- Define a regular pentagon
    ∃ (interior_angle : ℝ),
      -- The sum of interior angles of a pentagon is (5-2)*180 = 540 degrees
      5 * interior_angle = 540 ∧
      -- The total excess over 90 degrees for all angles
      5 * (interior_angle - 90) = total_excess ∧
      -- The theorem to prove
      total_excess = 90

-- The proof of the theorem
theorem prove_regular_pentagon_angle_excess :
  ∃ total_excess : ℝ, regular_pentagon_angle_excess total_excess :=
by
  sorry

end regular_pentagon_angle_excess_prove_regular_pentagon_angle_excess_l1025_102532


namespace correct_operation_l1025_102577

theorem correct_operation (a : ℝ) : 2 * a + 3 * a = 5 * a := by
  sorry

end correct_operation_l1025_102577


namespace apples_in_baskets_l1025_102515

theorem apples_in_baskets (total_apples : ℕ) (num_baskets : ℕ) (apples_removed : ℕ) : 
  total_apples = 64 → num_baskets = 4 → apples_removed = 3 →
  (total_apples / num_baskets) - apples_removed = 13 :=
by
  sorry

#check apples_in_baskets

end apples_in_baskets_l1025_102515


namespace max_intersection_points_l1025_102596

/-- Represents a line segment -/
structure Segment where
  id : ℕ

/-- Represents an intersection point -/
structure IntersectionPoint where
  id : ℕ

/-- The set of all segments -/
def segments : Finset Segment :=
  sorry

/-- The set of all intersection points -/
def intersectionPoints : Finset IntersectionPoint :=
  sorry

/-- Function that returns the number of intersections for a given segment -/
def intersectionsForSegment (s : Segment) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (segments.card = 10) →
  (∀ s ∈ segments, intersectionsForSegment s = 3) →
  intersectionPoints.card ≤ 15 := by
  sorry

end max_intersection_points_l1025_102596


namespace disprove_statement_l1025_102559

theorem disprove_statement : ∃ (a b c : ℤ), c < b ∧ b < a ∧ a * c < 0 ∧ a * b ≥ a * c := by
  sorry

end disprove_statement_l1025_102559


namespace square_characterization_l1025_102547

theorem square_characterization (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔
  (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i)^2 - A)) :=
by sorry

end square_characterization_l1025_102547


namespace factorization_equality_l1025_102506

theorem factorization_equality (x : ℝ) : 5*x*(x-2) + 9*(x-2) = (x-2)*(5*x+9) := by
  sorry

end factorization_equality_l1025_102506


namespace stephanie_oranges_l1025_102526

theorem stephanie_oranges (store_visits : ℕ) (oranges_per_visit : ℕ) : 
  store_visits = 8 → oranges_per_visit = 2 → store_visits * oranges_per_visit = 16 := by
sorry

end stephanie_oranges_l1025_102526


namespace sad_children_count_l1025_102585

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) :
  total = 60 →
  happy = 30 →
  neither = 20 →
  boys = 22 →
  girls = 38 →
  happy_boys = 6 →
  sad_girls = 4 →
  neither_boys = 10 →
  total = happy + neither + (total - happy - neither) →
  total - happy - neither = 10 := by
sorry

end sad_children_count_l1025_102585


namespace solid_is_triangular_prism_l1025_102599

/-- Represents a three-dimensional solid -/
structure Solid :=
  (front_view : Shape)
  (side_view : Shape)

/-- Represents geometric shapes -/
inductive Shape
  | Triangle
  | Quadrilateral
  | Other

/-- Defines a triangular prism -/
def is_triangular_prism (s : Solid) : Prop :=
  s.front_view = Shape.Triangle ∧ s.side_view = Shape.Quadrilateral

/-- Theorem: A solid with triangular front view and quadrilateral side view is a triangular prism -/
theorem solid_is_triangular_prism (s : Solid) 
  (h1 : s.front_view = Shape.Triangle) 
  (h2 : s.side_view = Shape.Quadrilateral) : 
  is_triangular_prism s := by
  sorry

end solid_is_triangular_prism_l1025_102599


namespace valid_p_values_l1025_102556

def is_valid_p (p : ℤ) : Prop :=
  ∃ (k : ℤ), k > 0 ∧ (4 * p + 20) = k * (3 * p - 6)

theorem valid_p_values :
  {p : ℤ | is_valid_p p} = {3, 4, 15, 28} :=
by sorry

end valid_p_values_l1025_102556


namespace rowing_current_rate_l1025_102545

/-- Proves that the current rate is 1.1 km/hr given the conditions of the rowing problem -/
theorem rowing_current_rate (man_speed : ℝ) (upstream_time_ratio : ℝ) :
  man_speed = 3.3 →
  upstream_time_ratio = 2 →
  ∃ (current_rate : ℝ),
    current_rate = 1.1 ∧
    (man_speed + current_rate) * upstream_time_ratio = man_speed - current_rate :=
by sorry

end rowing_current_rate_l1025_102545


namespace proper_divisor_cube_difference_l1025_102590

theorem proper_divisor_cube_difference (n : ℕ) : 
  (∃ (x y : ℕ), 
    x > 1 ∧ y > 1 ∧
    x ∣ n ∧ y ∣ n ∧
    n ≠ x ∧ n ≠ y ∧
    (∀ z : ℕ, z > 1 ∧ z ∣ n ∧ n ≠ z → z ≥ x) ∧
    (∀ z : ℕ, z > 1 ∧ z ∣ n ∧ n ≠ z → z ≤ y) ∧
    (y = x^3 + 3 ∨ y = x^3 - 3)) ↔
  (n = 10 ∨ n = 22) :=
sorry

end proper_divisor_cube_difference_l1025_102590


namespace f_has_two_zeros_l1025_102583

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2*x - 6 + Real.log x

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
by sorry

end f_has_two_zeros_l1025_102583


namespace rented_movie_cost_l1025_102539

theorem rented_movie_cost (ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) :
  ticket_price = 10.62 →
  num_tickets = 2 →
  bought_movie_price = 13.95 →
  total_spent = 36.78 →
  total_spent - (ticket_price * num_tickets + bought_movie_price) = 1.59 :=
by sorry

end rented_movie_cost_l1025_102539


namespace inequality_theorem_l1025_102510

theorem inequality_theorem (m n : ℕ) (h : m > n) :
  (1 + 1 / m : ℝ) ^ m > (1 + 1 / n : ℝ) ^ n ∧
  (1 + 1 / m : ℝ) ^ (m + 1) < (1 + 1 / n : ℝ) ^ (n + 1) := by
  sorry

end inequality_theorem_l1025_102510


namespace blue_candies_count_l1025_102592

/-- The number of blue candies in a bag, given the following conditions:
    - There are 5 green candies and 4 red candies
    - The probability of picking a blue candy is 25% -/
def num_blue_candies : ℕ :=
  let green_candies : ℕ := 5
  let red_candies : ℕ := 4
  let prob_blue : ℚ := 1/4
  3

theorem blue_candies_count :
  let green_candies : ℕ := 5
  let red_candies : ℕ := 4
  let prob_blue : ℚ := 1/4
  let total_candies : ℕ := green_candies + red_candies + num_blue_candies
  (num_blue_candies : ℚ) / total_candies = prob_blue :=
by sorry

end blue_candies_count_l1025_102592


namespace total_subjects_theorem_l1025_102560

/-- The number of subjects taken by Monica, Marius, and Millie -/
def total_subjects (monica : ℕ) (marius_extra : ℕ) (millie_extra : ℕ) : ℕ :=
  monica + (monica + marius_extra) + (monica + marius_extra + millie_extra)

/-- Theorem stating the total number of subjects taken by the three students -/
theorem total_subjects_theorem :
  total_subjects 10 4 3 = 41 := by
  sorry

end total_subjects_theorem_l1025_102560


namespace john_annual_maintenance_expenses_l1025_102508

/-- Represents John's annual car maintenance expenses --/
def annual_maintenance_expenses (
  annual_mileage : ℕ)
  (oil_change_interval : ℕ)
  (free_oil_changes : ℕ)
  (oil_change_cost : ℕ)
  (tire_rotation_interval : ℕ)
  (tire_rotation_cost : ℕ)
  (brake_pad_interval : ℕ)
  (brake_pad_cost : ℕ) : ℕ :=
  let paid_oil_changes := annual_mileage / oil_change_interval - free_oil_changes
  let annual_oil_change_cost := paid_oil_changes * oil_change_cost
  let annual_tire_rotation_cost := (annual_mileage / tire_rotation_interval) * tire_rotation_cost
  let annual_brake_pad_cost := (annual_mileage * brake_pad_cost) / brake_pad_interval
  annual_oil_change_cost + annual_tire_rotation_cost + annual_brake_pad_cost

/-- Theorem stating John's annual maintenance expenses --/
theorem john_annual_maintenance_expenses :
  annual_maintenance_expenses 12000 3000 1 50 6000 40 24000 200 = 330 := by
  sorry

end john_annual_maintenance_expenses_l1025_102508


namespace doctors_lawyers_ratio_l1025_102528

theorem doctors_lawyers_ratio (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (40 * (m + n) = 35 * m + 50 * n) → (m : ℚ) / n = 2 := by
  sorry

end doctors_lawyers_ratio_l1025_102528


namespace arithmetic_sequence_problem_l1025_102570

theorem arithmetic_sequence_problem (x : ℚ) :
  let a₁ := 3 * x - 4
  let a₂ := 6 * x - 14
  let a₃ := 4 * x + 2
  let d := a₂ - a₁  -- common difference
  let a_n (n : ℕ) := a₁ + (n - 1) * d  -- general term
  ∃ n : ℕ, a_n n = 4018 ∧ n = 716 :=
by sorry

end arithmetic_sequence_problem_l1025_102570


namespace arithmetic_mistakes_calculation_difference_l1025_102505

theorem arithmetic_mistakes (x : ℤ) : 
  ((-1 - 8) * 2 - x = -24) → (x = 6) :=
by sorry

theorem calculation_difference : 
  ((-1 - 8) + 2 - 5) - ((-1 - 8) * 2 - 5) = 11 :=
by sorry

end arithmetic_mistakes_calculation_difference_l1025_102505


namespace carter_cake_difference_l1025_102521

def regular_cheesecakes : ℕ := 6
def regular_muffins : ℕ := 5
def regular_red_velvet : ℕ := 8

def regular_total : ℕ := regular_cheesecakes + regular_muffins + regular_red_velvet

def triple_total : ℕ := 3 * regular_total

theorem carter_cake_difference : triple_total - regular_total = 38 := by
  sorry

end carter_cake_difference_l1025_102521


namespace scenario_proof_l1025_102558

theorem scenario_proof (a b c d : ℝ) 
  (h1 : a * b * c * d > 0) 
  (h2 : a < c) 
  (h3 : b * c * d < 0) : 
  a < 0 ∧ b > 0 ∧ c < 0 ∧ d > 0 :=
by sorry

end scenario_proof_l1025_102558


namespace body_part_count_l1025_102597

theorem body_part_count (suspension_days_per_instance : ℕ) 
                        (total_bullying_instances : ℕ) 
                        (body_part_count : ℕ) : 
  suspension_days_per_instance = 3 →
  total_bullying_instances = 20 →
  suspension_days_per_instance * total_bullying_instances = 3 * body_part_count →
  body_part_count = 20 := by
  sorry

end body_part_count_l1025_102597


namespace square_of_x_minus_three_l1025_102555

theorem square_of_x_minus_three (x : ℝ) (h : x = -3) : (x - 3)^2 = 36 := by
  sorry

end square_of_x_minus_three_l1025_102555


namespace new_average_production_l1025_102524

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) :
  n = 12 ∧ past_avg = 50 ∧ today_prod = 115 →
  (n * past_avg + today_prod) / (n + 1) = 55 := by
  sorry

end new_average_production_l1025_102524


namespace average_age_combined_l1025_102501

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 10 →
  avg_age_parents = 40 →
  let total_individuals := num_students + num_parents
  let total_age := num_students * avg_age_students + num_parents * avg_age_parents
  (total_age / total_individuals : ℝ) = 28 := by
sorry

end average_age_combined_l1025_102501


namespace unique_triples_l1025_102565

theorem unique_triples : 
  ∀ (a b c : ℕ+), 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
    (∃ (k₁ : ℕ+), (2 * a - 1 : ℤ) = k₁ * b) →
    (∃ (k₂ : ℕ+), (2 * b - 1 : ℤ) = k₂ * c) →
    (∃ (k₃ : ℕ+), (2 * c - 1 : ℤ) = k₃ * a) →
    ((a = 7 ∧ b = 13 ∧ c = 25) ∨
     (a = 13 ∧ b = 25 ∧ c = 7) ∨
     (a = 25 ∧ b = 7 ∧ c = 13)) :=
by sorry

end unique_triples_l1025_102565


namespace digit_120th_of_7_26th_l1025_102540

theorem digit_120th_of_7_26th : ∃ (seq : ℕ → ℕ), 
  (∀ n, seq n < 10) ∧ 
  (∀ n, seq (n + 9) = seq n) ∧
  (∀ n, (7 * 10^n) % 26 = (seq n * 10^8 + seq (n+1) * 10^7 + seq (n+2) * 10^6 + 
                           seq (n+3) * 10^5 + seq (n+4) * 10^4 + seq (n+5) * 10^3 + 
                           seq (n+6) * 10^2 + seq (n+7) * 10 + seq (n+8)) % 26) ∧
  seq 2 = 9 := by
  sorry

end digit_120th_of_7_26th_l1025_102540


namespace lucas_units_digit_l1025_102562

-- Define Lucas numbers
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem lucas_units_digit :
  unitsDigit (lucas (lucas 9)) = 7 := by
  sorry

end lucas_units_digit_l1025_102562


namespace mike_work_hours_l1025_102598

/-- Given that Mike worked 3 hours each day for 5 days, prove that his total work hours is 15. -/
theorem mike_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days_worked = 5 → total_hours = hours_per_day * days_worked → total_hours = 15 := by
  sorry

end mike_work_hours_l1025_102598


namespace circle_intersection_ratio_l1025_102518

theorem circle_intersection_ratio (m : ℝ) (h : 0 < m ∧ m < 1) :
  let R : ℝ := 1  -- We can set R = 1 without loss of generality
  let common_area := 2 * R^2 * (Real.arccos m - m * Real.sqrt (1 - m^2))
  let third_circle_area := π * (m * R)^2
  common_area / third_circle_area = 2 * (Real.arccos m - m * Real.sqrt (1 - m^2)) / (π * m^2) := by
  sorry

end circle_intersection_ratio_l1025_102518


namespace fraction_sum_bounds_l1025_102561

theorem fraction_sum_bounds (a b c d : ℕ+) 
  (sum_num : a + c = 1000)
  (sum_denom : b + d = 1000) :
  (999 : ℚ) / 969 + 1 / 31 ≤ (a : ℚ) / b + (c : ℚ) / d ∧ 
  (a : ℚ) / b + (c : ℚ) / d ≤ 999 + 1 / 999 := by
sorry

end fraction_sum_bounds_l1025_102561


namespace prism_21_edges_l1025_102575

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ := sorry

/-- The number of vertices in a prism -/
def num_vertices (p : Prism) : ℕ := sorry

/-- Theorem: A prism with 21 edges has 9 faces and 7 vertices -/
theorem prism_21_edges (p : Prism) (h : p.edges = 21) : 
  num_faces p = 9 ∧ num_vertices p = 7 := by sorry

end prism_21_edges_l1025_102575


namespace valid_pairings_count_l1025_102586

def number_of_bowls : ℕ := 5
def number_of_glasses : ℕ := 5
def number_of_colors : ℕ := 5

def total_pairings : ℕ := number_of_bowls * number_of_glasses

def invalid_pairings : ℕ := 1

theorem valid_pairings_count : 
  total_pairings - invalid_pairings = 24 :=
sorry

end valid_pairings_count_l1025_102586


namespace correct_statements_l1025_102574

theorem correct_statements :
  (abs (-5) = 5) ∧ (-(- 3) = 3) :=
by sorry

end correct_statements_l1025_102574


namespace cube_vertex_shapes_l1025_102530

-- Define a cube
structure Cube where
  vertices : Fin 8 → Point3D

-- Define a selection of 4 vertices from a cube
def VertexSelection (c : Cube) := Fin 4 → Fin 8

-- Define geometric shapes that can be formed by 4 vertices
inductive Shape
  | Rectangle
  | TetrahedronIsoscelesRight
  | TetrahedronEquilateral
  | TetrahedronRight

-- Function to check if a selection of vertices forms a specific shape
def formsShape (c : Cube) (s : VertexSelection c) (shape : Shape) : Prop :=
  match shape with
  | Shape.Rectangle => sorry
  | Shape.TetrahedronIsoscelesRight => sorry
  | Shape.TetrahedronEquilateral => sorry
  | Shape.TetrahedronRight => sorry

-- Theorem stating that all these shapes can be formed by selecting 4 vertices from a cube
theorem cube_vertex_shapes (c : Cube) :
  ∃ (s₁ s₂ s₃ s₄ : VertexSelection c),
    formsShape c s₁ Shape.Rectangle ∧
    formsShape c s₂ Shape.TetrahedronIsoscelesRight ∧
    formsShape c s₃ Shape.TetrahedronEquilateral ∧
    formsShape c s₄ Shape.TetrahedronRight :=
  sorry

end cube_vertex_shapes_l1025_102530


namespace triangle_problem_l1025_102537

/-- Given a triangle ABC with circumradius 1 and the relation between sides, prove the value of a and the area when b = 1. -/
theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (2 * Real.sin A) = 1 ∧  -- circumradius = 1
  b = a * Real.cos C - (Real.sqrt 3 / 6) * a * c →
  -- Conclusions
  a = Real.sqrt 3 ∧
  (b = 1 → Real.sqrt 3 / 4 = 1/2 * b * c * Real.sin A) :=
by sorry

end triangle_problem_l1025_102537


namespace sin_squared_50_over_1_plus_sin_10_l1025_102517

theorem sin_squared_50_over_1_plus_sin_10 :
  (Real.sin (50 * π / 180))^2 / (1 + Real.sin (10 * π / 180)) = 1 / 2 := by
  sorry

end sin_squared_50_over_1_plus_sin_10_l1025_102517


namespace tenth_term_of_sequence_l1025_102542

/-- The general term of the sequence -/
def sequenceTerm (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

/-- The 10th term of the sequence is 20/21 -/
theorem tenth_term_of_sequence : sequenceTerm 10 = 20 / 21 := by
  sorry

end tenth_term_of_sequence_l1025_102542


namespace complex_on_line_l1025_102525

theorem complex_on_line (a : ℝ) : 
  (∃ z : ℂ, z = (a - Complex.I) / (1 + Complex.I) ∧ 
   z.re - z.im + 1 = 0) ↔ a = -1 := by
  sorry

end complex_on_line_l1025_102525


namespace person_a_age_l1025_102538

/-- The ages of two people, A and B, satisfy certain conditions. -/
structure AgeProblem where
  /-- Age of Person A this year -/
  a : ℕ
  /-- Age of Person B this year -/
  b : ℕ
  /-- The sum of their ages this year is 43 -/
  sum_constraint : a + b = 43
  /-- In 4 years, A will be 3 years older than B -/
  future_constraint : a + 4 = (b + 4) + 3

/-- Given the age constraints, Person A's age this year is 23 -/
theorem person_a_age (p : AgeProblem) : p.a = 23 := by
  sorry

end person_a_age_l1025_102538


namespace sin_cos_sixth_power_sum_l1025_102512

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := by
  sorry

end sin_cos_sixth_power_sum_l1025_102512


namespace expression_simplification_l1025_102513

theorem expression_simplification (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a/b)^(b-a) := by
  sorry

end expression_simplification_l1025_102513


namespace cheryl_strawberries_l1025_102563

theorem cheryl_strawberries (total : ℕ) (buckets : ℕ) (left_in_each : ℕ) 
  (h1 : total = 300)
  (h2 : buckets = 5)
  (h3 : left_in_each = 40) :
  total / buckets - left_in_each = 20 := by
  sorry

end cheryl_strawberries_l1025_102563


namespace triangle_problem_l1025_102593

theorem triangle_problem (a b c A B C : ℝ) (h1 : c * Real.cos A - Real.sqrt 3 * a * Real.sin C - c = 0)
  (h2 : a = 2) (h3 : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end triangle_problem_l1025_102593


namespace sum_of_quadratic_solutions_l1025_102557

theorem sum_of_quadratic_solutions : 
  let f (x : ℝ) := x^2 - 4*x - 14 - (3*x + 16)
  let solutions := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 7 :=
by
  sorry

end sum_of_quadratic_solutions_l1025_102557


namespace queue_probabilities_l1025_102576

/-- Probabilities of different numbers of people queuing -/
structure QueueProbabilities where
  p0 : ℝ  -- Probability of 0 people
  p1 : ℝ  -- Probability of 1 person
  p2 : ℝ  -- Probability of 2 people
  p3 : ℝ  -- Probability of 3 people
  p4 : ℝ  -- Probability of 4 people
  p5 : ℝ  -- Probability of 5 or more people
  sum_to_one : p0 + p1 + p2 + p3 + p4 + p5 = 1
  all_nonneg : 0 ≤ p0 ∧ 0 ≤ p1 ∧ 0 ≤ p2 ∧ 0 ≤ p3 ∧ 0 ≤ p4 ∧ 0 ≤ p5

/-- The probabilities for the specific scenario -/
def scenario : QueueProbabilities where
  p0 := 0.1
  p1 := 0.16
  p2 := 0.3
  p3 := 0.3
  p4 := 0.1
  p5 := 0.04
  sum_to_one := by sorry
  all_nonneg := by sorry

theorem queue_probabilities (q : QueueProbabilities) :
  (q.p0 + q.p1 + q.p2 = 0.56) ∧ 
  (q.p3 + q.p4 + q.p5 = 0.44) :=
by sorry

end queue_probabilities_l1025_102576


namespace first_year_more_rabbits_l1025_102520

def squirrels (k : ℕ) : ℕ := 2020 * 2^k - 2019

def rabbits (k : ℕ) : ℕ := (4^k + 2) / 3

def more_rabbits_than_squirrels (k : ℕ) : Prop :=
  rabbits k > squirrels k

theorem first_year_more_rabbits : 
  (∀ n < 13, ¬(more_rabbits_than_squirrels n)) ∧ 
  more_rabbits_than_squirrels 13 := by
  sorry

#check first_year_more_rabbits

end first_year_more_rabbits_l1025_102520


namespace inequality_proof_l1025_102573

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end inequality_proof_l1025_102573


namespace expression_simplification_and_evaluation_specific_evaluation_l1025_102588

theorem expression_simplification_and_evaluation (a b : ℚ) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) - 2*a*a = 4*a*b :=
by sorry

theorem specific_evaluation :
  let a : ℚ := -1
  let b : ℚ := 1/2
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) - 2*a*a = -2 :=
by sorry

end expression_simplification_and_evaluation_specific_evaluation_l1025_102588


namespace range_of_a_l1025_102549

def A (a : ℝ) := {x : ℝ | 1 ≤ x ∧ x ≤ a}

def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = 5*x - 6}

def C (a : ℝ) := {m : ℝ | ∃ x ∈ A a, m = x^2}

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ A a → (1 ≤ x ∧ x ≤ a)) → 
  (∀ y : ℝ, y ∈ B a ↔ ∃ x ∈ A a, y = 5*x - 6) → 
  (∀ m : ℝ, m ∈ C a ↔ ∃ x ∈ A a, m = x^2) → 
  (B a ∩ C a = C a) → 
  (2 ≤ a ∧ a ≤ 3) :=
by sorry

end range_of_a_l1025_102549


namespace isosceles_right_triangle_ratio_l1025_102500

theorem isosceles_right_triangle_ratio (a c : ℝ) : 
  a > 0 → -- Ensure a is positive
  c^2 = 2 * a^2 → -- Pythagorean theorem for isosceles right triangle
  (2 * a) / c = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_ratio_l1025_102500


namespace grade_calculation_l1025_102533

/-- Represents the weighted average calculation for a student's grades --/
def weighted_average (math history science geography : ℝ) : ℝ :=
  0.3 * math + 0.3 * history + 0.2 * science + 0.2 * geography

/-- Theorem stating the conditions and the result to be proven --/
theorem grade_calculation (math history science geography : ℝ) :
  math = 74 →
  history = 81 →
  science = geography + 5 →
  science ≥ 75 →
  science = 86.25 →
  geography = 81.25 →
  weighted_average math history science geography = 80 :=
by
  sorry

#eval weighted_average 74 81 86.25 81.25

end grade_calculation_l1025_102533


namespace sequence_eventually_periodic_l1025_102587

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10

def is_eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ n₀ p, p > 0 ∧ ∀ k ≥ n₀, a (k + p) = a k

theorem sequence_eventually_periodic (a : ℕ → ℕ) (h : is_valid_sequence a) :
  is_eventually_periodic a := by
  sorry

#check sequence_eventually_periodic

end sequence_eventually_periodic_l1025_102587


namespace toms_incorrect_calculation_correct_calculation_l1025_102514

/-- The original number Tom was working with -/
def y : ℤ := 114

/-- Tom's incorrect calculation -/
theorem toms_incorrect_calculation : (y - 14) / 2 = 50 := by sorry

/-- The correct calculation -/
theorem correct_calculation : ((y - 5) / 7 : ℚ).floor = 15 := by sorry

end toms_incorrect_calculation_correct_calculation_l1025_102514


namespace f_positive_iff_x_range_l1025_102529

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_positive_iff_x_range (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, f a x > 0) ↔ (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end f_positive_iff_x_range_l1025_102529


namespace gcd_13247_36874_l1025_102511

theorem gcd_13247_36874 : Nat.gcd 13247 36874 = 1 := by
  sorry

end gcd_13247_36874_l1025_102511


namespace consecutive_integers_product_l1025_102568

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 2520 →
  e = 7 := by
sorry

end consecutive_integers_product_l1025_102568


namespace inequality_equivalence_l1025_102509

theorem inequality_equivalence (x y : ℝ) : y - x^2 < |x| ↔ y < x^2 + |x| := by
  sorry

end inequality_equivalence_l1025_102509


namespace quadratic_equation_solutions_l1025_102534

theorem quadratic_equation_solutions (a b : ℝ) :
  (∀ x : ℝ, x = -1 ∨ x = 2 → -a * x^2 + b * x = -2) →
  (-a * (-1)^2 + b * (-1) + 2 = 0) ∧ (-a * 2^2 + b * 2 + 2 = 0) :=
by sorry

end quadratic_equation_solutions_l1025_102534


namespace max_intersection_quadrilateral_pentagon_l1025_102584

/-- A polygon in the plane -/
structure Polygon :=
  (sides : ℕ)

/-- The number of intersection points between two polygons -/
def intersection_points (p1 p2 : Polygon) : ℕ := sorry

theorem max_intersection_quadrilateral_pentagon :
  ∃ (quad pent : Polygon),
    quad.sides = 4 ∧
    pent.sides = 5 ∧
    (∀ (q p : Polygon), q.sides = 4 → p.sides = 5 →
      intersection_points q p ≤ intersection_points quad pent) ∧
    intersection_points quad pent = 20 :=
sorry

end max_intersection_quadrilateral_pentagon_l1025_102584


namespace tetrahedron_sum_l1025_102541

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to define any fields, as we're only interested in its properties

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- The number of corners in a regular tetrahedron -/
def num_corners (t : RegularTetrahedron) : ℕ := 4

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- Theorem: The sum of edges, corners, and faces of a regular tetrahedron is 14 -/
theorem tetrahedron_sum (t : RegularTetrahedron) : 
  num_edges t + num_corners t + num_faces t = 14 := by
  sorry

end tetrahedron_sum_l1025_102541


namespace average_of_xyz_l1025_102544

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) :
  (x + y + z) / 3 = 4 := by
sorry

end average_of_xyz_l1025_102544


namespace number_equation_solution_l1025_102566

theorem number_equation_solution : ∃ (x : ℝ), x + 3 * x = 20 ∧ x = 5 := by sorry

end number_equation_solution_l1025_102566


namespace coffee_beans_per_cup_l1025_102523

/-- Represents the coffee consumption and cost scenario for Maddie's mom --/
structure CoffeeScenario where
  cups_per_day : ℕ
  coffee_bag_cost : ℚ
  coffee_bag_ounces : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  weekly_coffee_expense : ℚ

/-- Calculates the ounces of coffee beans per cup --/
def ounces_per_cup (scenario : CoffeeScenario) : ℚ :=
  sorry

/-- Theorem stating that the ounces of coffee beans per cup is 1.5 --/
theorem coffee_beans_per_cup (scenario : CoffeeScenario) 
  (h1 : scenario.cups_per_day = 2)
  (h2 : scenario.coffee_bag_cost = 8)
  (h3 : scenario.coffee_bag_ounces = 21/2)
  (h4 : scenario.milk_gallons_per_week = 1/2)
  (h5 : scenario.milk_cost_per_gallon = 4)
  (h6 : scenario.weekly_coffee_expense = 18) :
  ounces_per_cup scenario = 3/2 :=
sorry

end coffee_beans_per_cup_l1025_102523


namespace triangle_property_l1025_102552

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles of the triangle
  (a b c : ℝ)  -- Sides of the triangle opposite to angles A, B, C respectively

-- Define the property that makes a triangle right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : Real.sin t.C = Real.sin t.A * Real.cos t.B) : 
  isRightTriangle t :=
sorry

end triangle_property_l1025_102552


namespace unique_solution_trig_equation_l1025_102571

theorem unique_solution_trig_equation :
  ∃! (n : ℕ+), Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (2 * n.val) / 3 :=
by sorry

end unique_solution_trig_equation_l1025_102571


namespace y_value_l1025_102564

theorem y_value (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := by
  sorry

end y_value_l1025_102564


namespace quadratic_function_properties_l1025_102548

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end quadratic_function_properties_l1025_102548


namespace triangle_inequality_theorem_l1025_102551

/-- Checks if three numbers can form a triangle --/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  (¬ canFormTriangle 2 5 7) ∧
  (¬ canFormTriangle 9 3 5) ∧
  (canFormTriangle 4 5 6) ∧
  (¬ canFormTriangle 4 5 10) :=
sorry

end triangle_inequality_theorem_l1025_102551


namespace stephanies_remaining_payment_l1025_102536

/-- Calculates the remaining amount to pay for Stephanie's bills -/
def remaining_payment (electricity_bill gas_bill water_bill internet_bill : ℚ)
  (gas_paid_fraction : ℚ) (gas_additional_payment : ℚ)
  (water_paid_fraction : ℚ) (internet_payments : ℕ) (internet_payment_amount : ℚ) : ℚ :=
  let gas_remaining := gas_bill - (gas_paid_fraction * gas_bill + gas_additional_payment)
  let water_remaining := water_bill - (water_paid_fraction * water_bill)
  let internet_remaining := internet_bill - (internet_payments : ℚ) * internet_payment_amount
  gas_remaining + water_remaining + internet_remaining

/-- Stephanie's remaining bill payment is $30 -/
theorem stephanies_remaining_payment :
  remaining_payment 60 40 40 25 (3/4) 5 (1/2) 4 5 = 30 := by
  sorry

end stephanies_remaining_payment_l1025_102536


namespace additional_distance_for_average_speed_l1025_102567

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (increased_speed : ℝ)
  (target_average_speed : ℝ)
  (h1 : initial_distance = 15)
  (h2 : initial_speed = 30)
  (h3 : increased_speed = 55)
  (h4 : target_average_speed = 50) :
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / increased_speed)) = target_average_speed ∧
    additional_distance = 110 :=
by sorry

end additional_distance_for_average_speed_l1025_102567


namespace greatest_integer_problem_l1025_102578

theorem greatest_integer_problem : 
  (∃ m : ℕ, 
    0 < m ∧ 
    m < 150 ∧ 
    (∃ a : ℤ, m = 10 * a - 2) ∧ 
    (∃ b : ℤ, m = 9 * b - 4) ∧
    (∀ n : ℕ, 
      (0 < n ∧ 
       n < 150 ∧ 
       (∃ a' : ℤ, n = 10 * a' - 2) ∧ 
       (∃ b' : ℤ, n = 9 * b' - 4)) → 
      n ≤ m)) ∧
  (∀ m : ℕ, 
    (0 < m ∧ 
     m < 150 ∧ 
     (∃ a : ℤ, m = 10 * a - 2) ∧ 
     (∃ b : ℤ, m = 9 * b - 4) ∧
     (∀ n : ℕ, 
       (0 < n ∧ 
        n < 150 ∧ 
        (∃ a' : ℤ, n = 10 * a' - 2) ∧ 
        (∃ b' : ℤ, n = 9 * b' - 4)) → 
       n ≤ m)) → 
    m = 68) :=
sorry

end greatest_integer_problem_l1025_102578


namespace circle_central_symmetry_l1025_102527

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Central symmetry for a figure in 2D plane --/
def CentralSymmetry (F : Set (ℝ × ℝ)) :=
  ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, p ∈ F → (2 * c.1 - p.1, 2 * c.2 - p.2) ∈ F

/-- The set of points in a circle --/
def CirclePoints (c : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

/-- Theorem: A circle has central symmetry --/
theorem circle_central_symmetry (c : Circle) : CentralSymmetry (CirclePoints c) := by
  sorry


end circle_central_symmetry_l1025_102527


namespace max_visible_sum_l1025_102594

def cube_numbers : List ℕ := [1, 3, 9, 27, 81, 243]

def is_valid_cube (c : List ℕ) : Prop :=
  c.length = 6 ∧ c.toFinset = cube_numbers.toFinset

def visible_sum (bottom middle top : List ℕ) : ℕ :=
  (bottom.take 5).sum + (middle.take 5).sum + (top.take 5).sum

def is_valid_stack (bottom middle top : List ℕ) : Prop :=
  is_valid_cube bottom ∧ is_valid_cube middle ∧ is_valid_cube top

theorem max_visible_sum :
  ∀ bottom middle top : List ℕ,
    is_valid_stack bottom middle top →
    visible_sum bottom middle top ≤ 1087 :=
sorry

end max_visible_sum_l1025_102594


namespace writing_outlining_difference_l1025_102572

/-- Represents the time spent on different activities for a speech --/
structure SpeechTime where
  outlining : ℕ
  writing : ℕ
  practicing : ℕ

/-- Defines the conditions for Javier's speech preparation --/
def javierSpeechConditions (t : SpeechTime) : Prop :=
  t.outlining = 30 ∧
  t.writing > t.outlining ∧
  t.practicing = t.writing / 2 ∧
  t.outlining + t.writing + t.practicing = 117

/-- Theorem stating the difference between writing and outlining time --/
theorem writing_outlining_difference (t : SpeechTime) 
  (h : javierSpeechConditions t) : t.writing - t.outlining = 28 := by
  sorry

#check writing_outlining_difference

end writing_outlining_difference_l1025_102572


namespace inequality_abc_l1025_102569

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end inequality_abc_l1025_102569


namespace max_guaranteed_points_is_34_l1025_102543

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- The specific tournament described in the problem -/
def tournament : FootballTournament :=
  { num_teams := 15
  , points_for_win := 3
  , points_for_draw := 1
  , points_for_loss := 0 }

/-- The maximum number of points that can be guaranteed for each of 6 teams -/
def max_guaranteed_points (t : FootballTournament) : Nat :=
  34

/-- Theorem stating that 34 is the maximum number of points that can be guaranteed for each of 6 teams -/
theorem max_guaranteed_points_is_34 :
  ∀ n : Nat, n > max_guaranteed_points tournament →
  ¬(∃ points : Fin tournament.num_teams → Nat,
    (∀ i j : Fin tournament.num_teams, i ≠ j →
      points i + points j ≤ tournament.points_for_win) ∧
    (∃ top_6 : Finset (Fin tournament.num_teams),
      top_6.card = 6 ∧ ∀ i ∈ top_6, points i ≥ n)) :=
by sorry

end max_guaranteed_points_is_34_l1025_102543
