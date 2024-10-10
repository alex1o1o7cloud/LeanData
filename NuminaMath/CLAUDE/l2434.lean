import Mathlib

namespace printer_time_calculation_l2434_243475

def pages_to_print : ℕ := 300
def pages_per_minute : ℕ := 25
def pages_before_maintenance : ℕ := 50

theorem printer_time_calculation :
  let print_time := pages_to_print / pages_per_minute
  let maintenance_breaks := pages_to_print / pages_before_maintenance
  print_time + maintenance_breaks = 18 := by
  sorry

end printer_time_calculation_l2434_243475


namespace cubic_fraction_simplification_l2434_243421

theorem cubic_fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + b + 2 * c = 0) : 
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) := by
  sorry

end cubic_fraction_simplification_l2434_243421


namespace prob_four_or_full_house_after_reroll_l2434_243454

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 6

-- Define a function to represent the probability of a specific outcome when rolling a die
def prob_specific_outcome (sides : ℕ) : ℚ := 1 / sides

-- Define the probability of getting a four-of-a-kind or a full house after re-rolling
def prob_four_or_full_house : ℚ := 1 / 3

-- State the theorem
theorem prob_four_or_full_house_after_reroll 
  (h1 : ∃ (triple pair : ℕ), triple ≠ pair ∧ triple ≤ die_sides ∧ pair ≤ die_sides) 
  (h2 : ¬ ∃ (four : ℕ), four ≤ die_sides) :
  prob_four_or_full_house = prob_specific_outcome die_sides + prob_specific_outcome die_sides :=
sorry

end prob_four_or_full_house_after_reroll_l2434_243454


namespace percentage_enrolled_in_biology_l2434_243458

theorem percentage_enrolled_in_biology (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 594) :
  (((total_students - not_enrolled : ℝ) / total_students) * 100) = 
    (880 - 594 : ℝ) / 880 * 100 := by
  sorry

end percentage_enrolled_in_biology_l2434_243458


namespace molecular_weight_3_moles_HBrO3_l2434_243423

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Hydrogen atoms in HBrO3 -/
def num_H : ℕ := 1

/-- Number of Bromine atoms in HBrO3 -/
def num_Br : ℕ := 1

/-- Number of Oxygen atoms in HBrO3 -/
def num_O : ℕ := 3

/-- Number of moles of HBrO3 -/
def num_moles : ℝ := 3

/-- Calculates the molecular weight of HBrO3 in g/mol -/
def molecular_weight_HBrO3 : ℝ := 
  num_H * atomic_weight_H + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem: The molecular weight of 3 moles of HBrO3 is 386.73 grams -/
theorem molecular_weight_3_moles_HBrO3 : 
  num_moles * molecular_weight_HBrO3 = 386.73 := by
  sorry

end molecular_weight_3_moles_HBrO3_l2434_243423


namespace group_size_proof_l2434_243438

/-- The number of people in the group -/
def n : ℕ := 10

/-- The weight increase of the group when the new person joins -/
def weight_increase : ℕ := 40

/-- The weight of the person being replaced -/
def old_weight : ℕ := 70

/-- The weight of the new person joining the group -/
def new_weight : ℕ := 110

/-- The average weight increase per person -/
def avg_increase : ℕ := 4

theorem group_size_proof :
  n * old_weight + weight_increase = n * (old_weight + avg_increase) :=
by sorry

end group_size_proof_l2434_243438


namespace set_relationship_l2434_243400

/-- Definition of set E -/
def E : Set ℚ := { e | ∃ m : ℤ, e = m + 1/6 }

/-- Definition of set F -/
def F : Set ℚ := { f | ∃ n : ℤ, f = n/2 - 1/3 }

/-- Definition of set G -/
def G : Set ℚ := { g | ∃ p : ℤ, g = p/2 + 1/6 }

/-- Theorem stating the relationship among sets E, F, and G -/
theorem set_relationship : E ⊆ F ∧ F = G := by
  sorry

end set_relationship_l2434_243400


namespace christina_rearrangements_l2434_243491

theorem christina_rearrangements (n : ℕ) (rate1 rate2 : ℕ) (h1 : n = 9) (h2 : rate1 = 12) (h3 : rate2 = 18) :
  (n.factorial / 2 / rate1 + n.factorial / 2 / rate2) / 60 = 420 := by
  sorry

end christina_rearrangements_l2434_243491


namespace selling_price_after_markup_and_discount_l2434_243412

/-- The selling price of a commodity after markup and discount -/
theorem selling_price_after_markup_and_discount (a : ℝ) : 
  let markup_rate : ℝ := 0.5
  let discount_rate : ℝ := 0.3
  let marked_price : ℝ := a * (1 + markup_rate)
  let final_price : ℝ := marked_price * (1 - discount_rate)
  final_price = 1.05 * a :=
by sorry

end selling_price_after_markup_and_discount_l2434_243412


namespace rectangular_solid_surface_area_l2434_243482

/-- Given a rectangular solid with length l, width w, and height h, 
    prove that if it satisfies certain volume change conditions, 
    its surface area is 290 square cm. -/
theorem rectangular_solid_surface_area 
  (l w h : ℝ) 
  (h1 : (l - 2) * w * h = l * w * h - 48)
  (h2 : l * (w + 3) * h = l * w * h + 99)
  (h3 : l * w * (h + 4) = l * w * h + 352)
  : 2 * (l * w + l * h + w * h) = 290 := by
  sorry

end rectangular_solid_surface_area_l2434_243482


namespace rectangle_diagonal_l2434_243455

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * (l + w) = 72) →  -- perimeter condition
  (l = 5/2 * w) →       -- ratio condition
  Real.sqrt (l^2 + w^2) = 194/7 := by
sorry

end rectangle_diagonal_l2434_243455


namespace geometric_sequence_problem_l2434_243447

/-- Given a geometric sequence {a_n} with a₁ = 1 and common ratio q = 3,
    if the sum of the first t terms S_t = 364, then the t-th term a_t = 243 -/
theorem geometric_sequence_problem (t : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 3 * a n) →  -- geometric sequence with q = 3
  a 1 = 1 →                    -- a₁ = 1
  (∀ n, S n = (a 1) * (1 - 3^n) / (1 - 3)) →  -- sum formula for geometric sequence
  S t = 364 →                  -- S_t = 364
  a t = 243 := by              -- a_t = 243
sorry

end geometric_sequence_problem_l2434_243447


namespace arithmetic_sequence_11th_term_l2434_243470

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 10)
  (h_diff : a 8 ^ 2 - a 2 ^ 2 = 36) :
  a 11 = 11 := by
sorry

end arithmetic_sequence_11th_term_l2434_243470


namespace container_max_volume_l2434_243401

theorem container_max_volume :
  let total_length : ℝ := 24
  let volume (x : ℝ) : ℝ := x^2 * (total_length / 4 - x / 2)
  ∀ x > 0, x < total_length / 4 → volume x ≤ 8 ∧
  ∃ x > 0, x < total_length / 4 ∧ volume x = 8 :=
by sorry

end container_max_volume_l2434_243401


namespace amare_fabric_needed_l2434_243446

/-- The amount of fabric Amare needs for the dresses -/
def fabric_needed (fabric_per_dress : ℝ) (num_dresses : ℕ) (fabric_owned : ℝ) : ℝ :=
  fabric_per_dress * num_dresses * 3 - fabric_owned

/-- Theorem stating the amount of fabric Amare needs -/
theorem amare_fabric_needed :
  fabric_needed 5.5 4 7 = 59 := by
  sorry

end amare_fabric_needed_l2434_243446


namespace binomial_expansion_sum_l2434_243403

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₂ + a₃ = 40 := by
sorry

end binomial_expansion_sum_l2434_243403


namespace max_area_at_45_degrees_l2434_243453

/-- A screen in a room corner --/
structure Screen where
  length : ℝ
  angle : ℝ

/-- Configuration of two screens in a room corner --/
structure CornerScreens where
  screen1 : Screen
  screen2 : Screen

/-- The area enclosed by two screens in a room corner --/
noncomputable def enclosedArea (cs : CornerScreens) : ℝ := sorry

/-- Theorem: The area enclosed by two equal-length screens in a right-angled corner
    is maximized when each screen forms a 45° angle with its adjacent wall --/
theorem max_area_at_45_degrees (l : ℝ) (h : l > 0) :
  ∃ (cs : CornerScreens),
    cs.screen1.length = l ∧
    cs.screen2.length = l ∧
    cs.screen1.angle = π/4 ∧
    cs.screen2.angle = π/4 ∧
    ∀ (other : CornerScreens),
      other.screen1.length = l →
      other.screen2.length = l →
      enclosedArea other ≤ enclosedArea cs :=
sorry

end max_area_at_45_degrees_l2434_243453


namespace roots_product_theorem_l2434_243497

theorem roots_product_theorem (α β : ℝ) : 
  (α^2 + 2017*α + 1 = 0) → 
  (β^2 + 2017*β + 1 = 0) → 
  (1 + 2020*α + α^2) * (1 + 2020*β + β^2) = 9 := by
  sorry

end roots_product_theorem_l2434_243497


namespace backpack_price_relation_l2434_243460

theorem backpack_price_relation (x : ℝ) : x > 0 →
  (810 : ℝ) / (x + 20) = (600 : ℝ) / x * (1 - 0.1) := by
  sorry

end backpack_price_relation_l2434_243460


namespace polynomial_product_sum_l2434_243461

theorem polynomial_product_sum (x : ℝ) : ∃ (a b c d e : ℝ),
  (2 * x^3 - 3 * x^2 + 5 * x - 1) * (8 - 3 * x) = 
    a * x^4 + b * x^3 + c * x^2 + d * x + e ∧
  16 * a + 8 * b + 4 * c + 2 * d + e = 26 := by
  sorry

end polynomial_product_sum_l2434_243461


namespace aarons_age_l2434_243495

theorem aarons_age (aaron julie : ℕ) 
  (h1 : julie = 4 * aaron)
  (h2 : julie + 10 = 2 * (aaron + 10)) :
  aaron = 5 := by
sorry

end aarons_age_l2434_243495


namespace area_of_bounded_region_l2434_243402

/-- The equation of the graph that partitions the plane -/
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50*abs x = 500

/-- The bounded region formed by the graph -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ 
    ((y = 25 - 2*x ∧ x ≥ 0) ∨ (y = -25 - 2*x ∧ x < 0)) ∧
    -25 ≤ y ∧ y ≤ 25}

/-- The area of the bounded region is 1250 -/
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 1250 := by sorry

end area_of_bounded_region_l2434_243402


namespace terms_before_zero_l2434_243409

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem terms_before_zero (a : ℤ) (d : ℤ) (h1 : a = 102) (h2 : d = -6) :
  ∃ n : ℕ, n = 17 ∧ arithmetic_sequence a d (n + 1) = 0 :=
sorry

end terms_before_zero_l2434_243409


namespace lines_perpendicular_to_plane_are_parallel_l2434_243492

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l2434_243492


namespace quadratic_inequality_solution_set_l2434_243498

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) := by
  sorry

end quadratic_inequality_solution_set_l2434_243498


namespace grid_coloring_theorem_l2434_243433

theorem grid_coloring_theorem (n : ℕ) :
  (∀ (grid : Fin 25 → Fin n → Fin 8),
    ∃ (cols : Fin 4 → Fin n) (rows : Fin 4 → Fin 25),
      ∀ (i j : Fin 4), grid (rows i) (cols j) = grid (rows 0) (cols 0)) ↔
  n ≥ 303601 :=
by sorry

end grid_coloring_theorem_l2434_243433


namespace line_equation_l2434_243469

-- Define the point M
def M : ℝ × ℝ := (1, -2)

-- Define the line l
def l : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P is on the x-axis
axiom P_on_x_axis : P.2 = 0

-- State that Q is on the y-axis
axiom Q_on_y_axis : Q.1 = 0

-- State that M is the midpoint of PQ
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- State that P, Q, and M are on the line l
axiom P_on_l : P ∈ l
axiom Q_on_l : Q ∈ l
axiom M_on_l : M ∈ l

-- Theorem: The equation of line PQ is 2x - y - 4 = 0
theorem line_equation : ∀ (x y : ℝ), (x, y) ∈ l ↔ 2 * x - y - 4 = 0 :=
sorry

end line_equation_l2434_243469


namespace negation_of_quadratic_inequality_l2434_243442

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := by sorry

end negation_of_quadratic_inequality_l2434_243442


namespace ray_AB_bisects_PAQ_l2434_243473

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 5/2)^2 = 25/4

-- Define points A and B on the y-axis
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (0, 1)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2/8 + y^2/4 = 1

-- Define line l passing through B
def line_l (x y : ℝ) (k : ℝ) : Prop :=
  y = k * x + 1

-- Theorem statement
theorem ray_AB_bisects_PAQ :
  ∀ (P Q : ℝ × ℝ) (k : ℝ),
    circle_C 2 0 →  -- Circle C is tangent to x-axis at T(2,0)
    |point_A.2 - point_B.2| = 3 →  -- |AB| = 3
    line_l P.1 P.2 k →  -- P is on line l
    line_l Q.1 Q.2 k →  -- Q is on line l
    ellipse P.1 P.2 →  -- P is on the ellipse
    ellipse Q.1 Q.2 →  -- Q is on the ellipse
    -- Ray AB bisects angle PAQ
    (P.2 - point_A.2) / (P.1 - point_A.1) + (Q.2 - point_A.2) / (Q.1 - point_A.1) = 0 :=
by
  sorry


end ray_AB_bisects_PAQ_l2434_243473


namespace stewart_farm_sheep_count_l2434_243459

/-- Stewart Farm problem -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (sheep_food horse_food : ℝ),
  sheep * 7 = horses →
  horse_food = 230 →
  horses * horse_food = 12880 →
  sheep_food = 150 →
  sheep * sheep_food = 6300 →
  sheep = 8 := by
sorry

end stewart_farm_sheep_count_l2434_243459


namespace arianna_work_hours_l2434_243417

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Arianna spends on chores -/
def hours_on_chores : ℕ := 5

/-- Represents the number of hours Arianna spends sleeping -/
def hours_sleeping : ℕ := 13

/-- Theorem stating that Arianna spends 6 hours at work -/
theorem arianna_work_hours :
  hours_in_day - (hours_on_chores + hours_sleeping) = 6 := by
  sorry

end arianna_work_hours_l2434_243417


namespace quadratic_equation_solution_l2434_243450

theorem quadratic_equation_solution (a k : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - k = 0) → 
  (k = 44) → 
  (a * 4^2 + 3 * 4 - k = 0) → 
  a = 2 := by
sorry

end quadratic_equation_solution_l2434_243450


namespace inches_in_foot_l2434_243468

theorem inches_in_foot (room_side : ℝ) (room_area_sq_inches : ℝ) :
  room_side = 10 →
  room_area_sq_inches = 14400 →
  ∃ (inches_per_foot : ℝ), inches_per_foot = 12 ∧ 
    (room_side * inches_per_foot)^2 = room_area_sq_inches :=
by sorry

end inches_in_foot_l2434_243468


namespace three_number_problem_l2434_243404

theorem three_number_problem (a b c : ℚ) : 
  a + b + c = 220 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a →
  b = 60 := by sorry

end three_number_problem_l2434_243404


namespace least_integer_with_divisibility_conditions_l2434_243413

def is_prime (n : ℕ) : Prop := sorry

def is_consecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

theorem least_integer_with_divisibility_conditions (N : ℕ) : 
  (∀ k ∈ Finset.range 31, k ≠ 0 → ∃ (a b : ℕ), a ≠ b ∧ is_consecutive a b ∧ 
    (is_prime a ∨ is_prime b) ∧ 
    (∀ i ∈ Finset.range 31, i ≠ 0 ∧ i ≠ a ∧ i ≠ b → N % i = 0) ∧
    N % a ≠ 0 ∧ N % b ≠ 0) →
  N ≥ 8923714800 :=
sorry

end least_integer_with_divisibility_conditions_l2434_243413


namespace quadratic_roots_l2434_243426

/-- The quadratic equation kx^2 - (2k-3)x + k-2 = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 - (2*k - 3) * x + (k - 2) = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  9 - 4*k

theorem quadratic_roots :
  (∃! x : ℝ, quadratic_equation 0 x) ∧
  (∀ k : ℝ, 0 < k → k ≤ 9/4 → ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) :=
sorry

end quadratic_roots_l2434_243426


namespace bears_per_shelf_l2434_243449

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 6)
  (h2 : new_shipment = 18)
  (h3 : num_shelves = 4) :
  (initial_stock + new_shipment) / num_shelves = 6 :=
by sorry

end bears_per_shelf_l2434_243449


namespace rs_equals_240_l2434_243415

-- Define the triangle DEF
structure Triangle (DE EF : ℝ) where
  de_positive : DE > 0
  ef_positive : EF > 0

-- Define points Q, R, S, N
structure Points (D E F Q R S N : ℝ × ℝ) where
  q_on_de : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • D + t • E
  r_on_df : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • D + t • F
  s_on_fq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • F + t • Q
  s_on_er : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • E + t • R
  n_on_fq : ∃ t : ℝ, t > 1 ∧ N = (1 - t) • F + t • Q

-- Define the conditions
def Conditions (D E F Q R S N : ℝ × ℝ) (triangle : Triangle 600 400) (points : Points D E F Q R S N) : Prop :=
  let de := ‖E - D‖
  let dq := ‖Q - D‖
  let qe := ‖E - Q‖
  let dn := ‖N - D‖
  let sn := ‖N - S‖
  let sq := ‖Q - S‖
  de = 600 ∧ dq = qe ∧ dn = 240 ∧ sn = sq

-- Theorem statement
theorem rs_equals_240 (D E F Q R S N : ℝ × ℝ) 
  (triangle : Triangle 600 400) (points : Points D E F Q R S N) 
  (h : Conditions D E F Q R S N triangle points) : 
  ‖R - S‖ = 240 := by sorry

end rs_equals_240_l2434_243415


namespace five_balls_three_boxes_l2434_243466

/-- The number of ways to place n distinguishable balls into k indistinguishable boxes -/
def ball_distribution (n k : ℕ) : ℕ := sorry

/-- The number of ways to place 5 distinguishable balls into 3 indistinguishable boxes is 36 -/
theorem five_balls_three_boxes : ball_distribution 5 3 = 36 := by sorry

end five_balls_three_boxes_l2434_243466


namespace triangle_perimeter_l2434_243419

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define the properties of the triangle
def TriangleProperties (A B C : ℝ × ℝ) :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  BC = AC - 1 ∧ AC = AB - 1 ∧ (AB^2 + AC^2 - BC^2) / (2 * AB * AC) = 3/5

-- Theorem statement
theorem triangle_perimeter (A B C : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hp : TriangleProperties A B C) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB + BC + AC = 42 := by
  sorry

end triangle_perimeter_l2434_243419


namespace speeding_percentage_l2434_243429

/-- The percentage of motorists who exceed the speed limit and receive tickets -/
def ticketed_speeders : ℝ := 20

/-- The percentage of speeding motorists who do not receive tickets -/
def unticketed_speeder_percentage : ℝ := 20

/-- The total percentage of motorists who exceed the speed limit -/
def total_speeders : ℝ := 25

theorem speeding_percentage :
  ticketed_speeders * (100 - unticketed_speeder_percentage) / 100 = total_speeders * (100 - unticketed_speeder_percentage) / 100 := by
  sorry

end speeding_percentage_l2434_243429


namespace tens_digit_of_2023_pow_2024_minus_2025_l2434_243478

theorem tens_digit_of_2023_pow_2024_minus_2025 :
  (2023^2024 - 2025) % 100 / 10 = 5 := by
  sorry

end tens_digit_of_2023_pow_2024_minus_2025_l2434_243478


namespace sixth_employee_salary_l2434_243437

def employee_salaries : List ℝ := [1000, 2500, 3650, 1500, 2000]
def mean_salary : ℝ := 2291.67
def num_employees : ℕ := 6

theorem sixth_employee_salary :
  let total_salary := (mean_salary * num_employees)
  let known_salaries_sum := employee_salaries.sum
  total_salary - known_salaries_sum = 2100 := by
  sorry

end sixth_employee_salary_l2434_243437


namespace equation_solution_l2434_243481

theorem equation_solution :
  ∃ x : ℝ, (10 : ℝ)^x * 500^x = 1000000^3 ∧ x = 18 / 3.699 := by
  sorry

end equation_solution_l2434_243481


namespace arithmetic_sequence_special_property_l2434_243428

/-- Given an arithmetic sequence {a_n} with common difference d (d ≠ 0) and sum of first n terms S_n,
    if {√(S_n + n)} is also an arithmetic sequence with common difference d, then d = 1/2. -/
theorem arithmetic_sequence_special_property (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) - a n = d) ∧
  (∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * d) ∧
  (∀ n : ℕ, Real.sqrt (S (n + 1) + (n + 1)) - Real.sqrt (S n + n) = d) →
  d = 1 / 2 := by
sorry

end arithmetic_sequence_special_property_l2434_243428


namespace thabo_hardcover_count_l2434_243479

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 160 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 20 ∧
  bc.paperback_fiction = 2 * bc.paperback_nonfiction

theorem thabo_hardcover_count (bc : BookCollection) 
  (h : is_valid_collection bc) : bc.hardcover_nonfiction = 25 := by
  sorry

end thabo_hardcover_count_l2434_243479


namespace unique_triple_solution_l2434_243430

theorem unique_triple_solution :
  ∃! (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y = 3 ∧ x * y - z^2 = 4 ∧ x = 1 ∧ y = 2 ∧ z = 0 := by
  sorry

end unique_triple_solution_l2434_243430


namespace remaining_digits_average_l2434_243457

theorem remaining_digits_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 20 →
  subset = 14 →
  total_avg = 500 →
  subset_avg = 390 →
  let remaining := total - subset
  let remaining_sum := total * total_avg - subset * subset_avg
  remaining_sum / remaining = 756.67 := by
  sorry

end remaining_digits_average_l2434_243457


namespace juan_number_puzzle_l2434_243406

theorem juan_number_puzzle (n : ℝ) : ((n + 3) * 3 - 3) * 2 / 3 = 10 → n = 3 := by
  sorry

end juan_number_puzzle_l2434_243406


namespace cubic_expression_value_l2434_243480

theorem cubic_expression_value : 
  let x : ℤ := -2
  (-2)^3 + (-2)^2 + 3*(-2) - 6 = -16 := by sorry

end cubic_expression_value_l2434_243480


namespace f_of_two_equals_eleven_l2434_243490

/-- A function f satisfying the given conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b * x + 3

/-- The theorem stating that f(2) = 11 under the given conditions -/
theorem f_of_two_equals_eleven (a b : ℝ) 
  (h1 : f a b 1 = 7) 
  (h3 : f a b 3 = 15) : 
  f a b 2 = 11 := by
  sorry

end f_of_two_equals_eleven_l2434_243490


namespace remainder_theorem_l2434_243411

theorem remainder_theorem (x y u v : ℤ) : 
  x > 0 → y > 0 → x = u * y + v → 0 ≤ v → v < y → 
  (x - u * y + 3 * v) % y = 4 * v % y := by
sorry

end remainder_theorem_l2434_243411


namespace max_students_equal_distribution_l2434_243425

theorem max_students_equal_distribution (pens pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) :
  (Nat.gcd pens pencils : ℕ) = 4 :=
sorry

end max_students_equal_distribution_l2434_243425


namespace median_length_in_right_triangle_l2434_243486

theorem median_length_in_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let c := Real.sqrt (a^2 + b^2)  -- hypotenuse
  let m := Real.sqrt (b^2 + (a/2)^2)  -- median
  (∃ k : ℝ, k = 0.51 ∧ m = k * c) ∧ ¬(∃ k : ℝ, k = 0.49 ∧ m = k * c) :=
by sorry

end median_length_in_right_triangle_l2434_243486


namespace perfect_square_polynomial_l2434_243424

theorem perfect_square_polynomial (k : ℝ) : 
  (∀ a : ℝ, ∃ b : ℝ, a^2 + 2*k*a + 1 = b^2) → (k = 1 ∨ k = -1) := by
  sorry

end perfect_square_polynomial_l2434_243424


namespace christen_peeled_twenty_potatoes_l2434_243494

/-- Calculates the number of potatoes Christen peeled -/
def christenPotatoesPeeled (initialPotatoes : ℕ) (homerRate : ℕ) (christenRate : ℕ) (timeBeforeJoining : ℕ) : ℕ :=
  let potatoesLeftWhenChristenJoins := initialPotatoes - homerRate * timeBeforeJoining
  let combinedRate := homerRate + christenRate
  let timeToFinish := potatoesLeftWhenChristenJoins / combinedRate
  christenRate * timeToFinish

/-- Theorem stating that Christen peeled 20 potatoes given the initial conditions -/
theorem christen_peeled_twenty_potatoes :
  christenPotatoesPeeled 44 3 5 4 = 20 := by
  sorry

end christen_peeled_twenty_potatoes_l2434_243494


namespace normal_distribution_probability_l2434_243436

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for the normal distribution -/
noncomputable def probability (X : NormalDistribution) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability (X : NormalDistribution) 
  (h1 : X.μ = 4)
  (h2 : X.σ = 1)
  (h3 : probability X (X.μ - 2 * X.σ) (X.μ + 2 * X.σ) = 0.9544)
  (h4 : probability X (X.μ - X.σ) (X.μ + X.σ) = 0.6826) :
  probability X 5 6 = 0.1359 := by
  sorry

end normal_distribution_probability_l2434_243436


namespace emily_beads_count_l2434_243487

/-- The number of necklaces Emily made -/
def necklaces : ℕ := 26

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 2

/-- The total number of beads Emily had -/
def total_beads : ℕ := necklaces * beads_per_necklace

/-- Theorem stating that the total number of beads Emily had is 52 -/
theorem emily_beads_count : total_beads = 52 := by
  sorry

end emily_beads_count_l2434_243487


namespace min_value_theorem_min_value_explicit_l2434_243410

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = 1/4) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 3)⁻¹ + (y + 3)⁻¹ = 1/4 → a + 3*b ≤ x + 3*y :=
by sorry

theorem min_value_explicit (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = 1/4) : 
  a + 3*b = 4 + 8*Real.sqrt 3 :=
by sorry

end min_value_theorem_min_value_explicit_l2434_243410


namespace sum_of_ages_in_5_years_l2434_243496

-- Define the current ages
def will_current_age : ℕ := 7
def diane_current_age : ℕ := 2 * will_current_age
def janet_current_age : ℕ := diane_current_age + 3

-- Define the ages in 5 years
def will_future_age : ℕ := will_current_age + 5
def diane_future_age : ℕ := diane_current_age + 5
def janet_future_age : ℕ := janet_current_age + 5

-- Theorem to prove
theorem sum_of_ages_in_5_years :
  will_future_age + diane_future_age + janet_future_age = 53 := by
  sorry


end sum_of_ages_in_5_years_l2434_243496


namespace exists_non_convex_polyhedron_with_no_visible_vertices_l2434_243493

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_valid : True  -- Additional conditions for a valid polyhedron would go here

/-- A point in 3D space -/
def Point3D := Fin 3 → ℝ

/-- Predicate to check if a polyhedron is non-convex -/
def is_non_convex (P : Polyhedron) : Prop := sorry

/-- Predicate to check if a point is outside a polyhedron -/
def is_outside (M : Point3D) (P : Polyhedron) : Prop := sorry

/-- Predicate to check if a vertex is visible from a point -/
def is_visible (v : Point3D) (M : Point3D) (P : Polyhedron) : Prop := sorry

/-- Theorem stating the existence of a non-convex polyhedron with no visible vertices from an exterior point -/
theorem exists_non_convex_polyhedron_with_no_visible_vertices :
  ∃ (P : Polyhedron) (M : Point3D),
    is_non_convex P ∧
    is_outside M P ∧
    ∀ v ∈ P.vertices, ¬is_visible v M P := by
  sorry

end exists_non_convex_polyhedron_with_no_visible_vertices_l2434_243493


namespace soccer_team_winning_percentage_l2434_243477

/-- Calculates the winning percentage of a soccer team -/
def winning_percentage (games_played : ℕ) (games_won : ℕ) : ℚ :=
  (games_won : ℚ) / (games_played : ℚ) * 100

/-- Theorem stating that a team with 280 games played and 182 wins has a 65% winning percentage -/
theorem soccer_team_winning_percentage :
  let games_played : ℕ := 280
  let games_won : ℕ := 182
  winning_percentage games_played games_won = 65 := by
  sorry

end soccer_team_winning_percentage_l2434_243477


namespace shells_added_l2434_243488

/-- Given that Jovana initially had 5 pounds of shells and now has 28 pounds,
    prove that she added 23 pounds of shells. -/
theorem shells_added (initial : ℕ) (final : ℕ) (h1 : initial = 5) (h2 : final = 28) :
  final - initial = 23 := by
  sorry

end shells_added_l2434_243488


namespace celestia_badges_l2434_243489

def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def luna_badges : ℕ := 17

theorem celestia_badges : 
  total_badges - hermione_badges - luna_badges = 52 := by
  sorry

end celestia_badges_l2434_243489


namespace johnson_family_seating_l2434_243435

/-- The number of ways to arrange 5 sons and 3 daughters in a row of 8 chairs
    such that at least 2 sons are next to each other -/
def seating_arrangements (num_sons : Nat) (num_daughters : Nat) : Nat :=
  Nat.factorial (num_sons + num_daughters) - 
  (Nat.factorial num_daughters * Nat.factorial num_sons)

theorem johnson_family_seating :
  seating_arrangements 5 3 = 39600 := by
  sorry

end johnson_family_seating_l2434_243435


namespace first_divisor_of_square_plus_164_l2434_243405

theorem first_divisor_of_square_plus_164 : 
  ∀ n ∈ [3, 4, 5, 6, 7, 8, 9, 10, 11], 
    (n ∣ (166^2 + 164)) → 
    n = 3 := by sorry

end first_divisor_of_square_plus_164_l2434_243405


namespace sum_digits_1944_base9_l2434_243420

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits of 1944 in base 9 is 8 -/
theorem sum_digits_1944_base9 : sumDigits (toBase9 1944) = 8 := by
  sorry

end sum_digits_1944_base9_l2434_243420


namespace alexa_katerina_weight_l2434_243432

/-- The combined weight of Alexa and Katerina is 92 pounds -/
theorem alexa_katerina_weight (total_weight : ℕ) (alexa_weight : ℕ) (michael_weight : ℕ)
  (h1 : total_weight = 154)
  (h2 : alexa_weight = 46)
  (h3 : michael_weight = 62) :
  total_weight - michael_weight = 92 :=
by sorry

end alexa_katerina_weight_l2434_243432


namespace moon_radius_scientific_notation_l2434_243464

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (hx : x > 0) : ScientificNotation :=
  sorry

theorem moon_radius_scientific_notation :
  toScientificNotation 1738000 (by norm_num) =
    ScientificNotation.mk 1.738 6 (by norm_num) (by norm_num) :=
  sorry

end moon_radius_scientific_notation_l2434_243464


namespace certain_number_proof_l2434_243407

theorem certain_number_proof : ∃ x : ℝ, x * 7 = (35 / 100) * 900 ∧ x = 45 := by sorry

end certain_number_proof_l2434_243407


namespace largest_n_satisfying_conditions_l2434_243434

theorem largest_n_satisfying_conditions : ∃ (n : ℕ), n = 313 ∧ 
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧ 
  (∃ (a : ℕ), 5*n + 103 = a^2) ∧
  (∀ (k : ℕ), k > n → ¬(∃ (m : ℤ), k^2 = (m+1)^3 - m^3) ∨ ¬(∃ (a : ℕ), 5*k + 103 = a^2)) :=
sorry

end largest_n_satisfying_conditions_l2434_243434


namespace product_expansion_sum_l2434_243476

theorem product_expansion_sum (a b c d e : ℝ) :
  (∀ x : ℝ, (5*x^3 - 3*x^2 + x - 8)*(8 - 3*x) = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  16*a + 8*b + 4*c + 2*d + e = 44 := by
sorry

end product_expansion_sum_l2434_243476


namespace percentage_problem_l2434_243440

theorem percentage_problem (P : ℝ) : 
  (100 : ℝ) = (P / 100) * 100 + 84 → P = 16 := by
  sorry

end percentage_problem_l2434_243440


namespace name_tag_area_l2434_243483

/-- The area of a square name tag with side length 11 cm is 121 cm² -/
theorem name_tag_area : 
  let side_length : ℝ := 11
  let area : ℝ := side_length * side_length
  area = 121 := by sorry

end name_tag_area_l2434_243483


namespace polynomial_coefficients_sum_l2434_243484

theorem polynomial_coefficients_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1 ∧ a₀ + a₂ + a₄ + a₆ = 365) :=
by sorry

end polynomial_coefficients_sum_l2434_243484


namespace largest_solution_of_equation_l2434_243471

theorem largest_solution_of_equation (a : ℝ) : 
  (3 * a + 4) * (a - 2) = 8 * a → a ≤ 4 := by
  sorry

end largest_solution_of_equation_l2434_243471


namespace remainder_problem_l2434_243462

theorem remainder_problem (x : ℤ) : x % 82 = 5 → (x + 7) % 41 = 12 := by
  sorry

end remainder_problem_l2434_243462


namespace sqrt_three_squared_equals_three_l2434_243474

theorem sqrt_three_squared_equals_three : (Real.sqrt 3)^2 = 3 := by
  sorry

end sqrt_three_squared_equals_three_l2434_243474


namespace total_dolls_l2434_243452

def doll_problem (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ) : Prop :=
  vera_dolls = 20 ∧
  sophie_dolls = 2 * vera_dolls ∧
  aida_dolls = 2 * sophie_dolls ∧
  aida_dolls + sophie_dolls + vera_dolls = 140

theorem total_dolls :
  ∃ (vera_dolls sophie_dolls aida_dolls : ℕ),
    doll_problem vera_dolls sophie_dolls aida_dolls :=
by
  sorry

end total_dolls_l2434_243452


namespace seokjin_position_relative_to_jungkook_l2434_243427

/-- Given the positions of Jungkook, Yoojeong, and Seokjin on a staircase,
    prove that Seokjin stands 3 steps above Jungkook. -/
theorem seokjin_position_relative_to_jungkook 
  (jungkook_stair : ℕ) 
  (yoojeong_above_jungkook : ℕ) 
  (seokjin_below_yoojeong : ℕ) 
  (h1 : jungkook_stair = 19)
  (h2 : yoojeong_above_jungkook = 8)
  (h3 : seokjin_below_yoojeong = 5) :
  (jungkook_stair + yoojeong_above_jungkook - seokjin_below_yoojeong) - jungkook_stair = 3 :=
by sorry

end seokjin_position_relative_to_jungkook_l2434_243427


namespace absolute_value_inequality_l2434_243414

theorem absolute_value_inequality (x : ℝ) : 3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := by
  sorry

end absolute_value_inequality_l2434_243414


namespace raine_steps_theorem_l2434_243456

/-- The number of steps Raine takes to walk to school -/
def steps_to_school : ℕ := 150

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- The total number of steps Raine takes in the given number of days -/
def total_steps : ℕ := 2 * steps_to_school * days

theorem raine_steps_theorem : total_steps = 1500 := by
  sorry

end raine_steps_theorem_l2434_243456


namespace village_population_equality_l2434_243467

/-- The initial population of Village X -/
def initial_population_X : ℕ := 78000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The yearly increase in population of Village Y -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 18

/-- The initial population of Village Y -/
def initial_population_Y : ℕ := 42000

theorem village_population_equality :
  initial_population_X - decrease_rate_X * years = 
  initial_population_Y + increase_rate_Y * years :=
by sorry

#check village_population_equality

end village_population_equality_l2434_243467


namespace tommys_tomato_profit_l2434_243445

/-- Represents the problem of calculating Tommy's profit from selling tomatoes --/
theorem tommys_tomato_profit :
  let crate_capacity : ℕ := 20  -- kg
  let num_crates : ℕ := 3
  let purchase_cost : ℕ := 330  -- $
  let selling_price : ℕ := 6    -- $ per kg
  let rotten_tomatoes : ℕ := 3  -- kg
  
  let total_tomatoes : ℕ := crate_capacity * num_crates
  let sellable_tomatoes : ℕ := total_tomatoes - rotten_tomatoes
  let revenue : ℕ := sellable_tomatoes * selling_price
  let profit : ℤ := revenue - purchase_cost
  
  profit = 12 := by sorry

end tommys_tomato_profit_l2434_243445


namespace computer_multiplications_l2434_243431

/-- Represents the number of multiplications a computer can perform per second -/
def multiplications_per_second : ℕ := 15000

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Represents the number of hours we're calculating for -/
def hours : ℕ := 2

/-- Theorem stating that the computer will perform 108 million multiplications in two hours -/
theorem computer_multiplications :
  multiplications_per_second * seconds_per_hour * hours = 108000000 := by
  sorry

#eval multiplications_per_second * seconds_per_hour * hours

end computer_multiplications_l2434_243431


namespace rotation_of_point_N_l2434_243439

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem rotation_of_point_N : 
  let N : Point := ⟨-1, -2⟩
  rotate180 N = ⟨1, 2⟩ := by
  sorry

end rotation_of_point_N_l2434_243439


namespace exists_valid_assignment_l2434_243444

/-- Represents a rectangular parallelepiped with dimensions a, b, and c -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents an assignment of numbers to the unit squares on the surface of a parallelepiped -/
def SurfaceAssignment (p : Parallelepiped) := (ℕ × ℕ × ℕ) → ℝ

/-- Calculates the sum of numbers in a 1-width band surrounding the parallelepiped -/
def bandSum (p : Parallelepiped) (assignment : SurfaceAssignment p) : ℝ := sorry

/-- Theorem stating the existence of a valid assignment for a 3 × 4 × 5 parallelepiped -/
theorem exists_valid_assignment :
  ∃ (assignment : SurfaceAssignment ⟨3, 4, 5⟩),
    bandSum ⟨3, 4, 5⟩ assignment = 120 := by sorry

end exists_valid_assignment_l2434_243444


namespace trajectory_is_ellipse_chord_length_l2434_243441

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + Real.sqrt 3)^2) + Real.sqrt (x^2 + (y - Real.sqrt 3)^2) = 4

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + y^2/4 = 1

-- Define the line y = 1/2x
def line (x y : ℝ) : Prop :=
  y = 1/2 * x

-- Theorem 1: The trajectory C is equivalent to the ellipse equation
theorem trajectory_is_ellipse :
  ∀ x y : ℝ, trajectory_C x y ↔ ellipse_equation x y :=
sorry

-- Theorem 2: The length of the chord AB is 4
theorem chord_length :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_equation x₁ y₁ ∧
    ellipse_equation x₂ y₂ ∧
    line x₁ y₁ ∧
    line x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 :=
sorry

end trajectory_is_ellipse_chord_length_l2434_243441


namespace all_points_in_triangle_satisfy_condition_probability_a_minus_b_positive_is_zero_l2434_243485

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.1 ≤ 4 ∧ p.2 ≥ 0 ∧ 4 * p.2 ≤ 10 * p.1}

-- Theorem statement
theorem all_points_in_triangle_satisfy_condition :
  ∀ p : ℝ × ℝ, p ∈ triangle → p.1 - p.2 ≤ 0 :=
by
  sorry

-- Probability statement
theorem probability_a_minus_b_positive_is_zero :
  ∀ p : ℝ × ℝ, p ∈ triangle → (p.1 - p.2 > 0) = false :=
by
  sorry

end all_points_in_triangle_satisfy_condition_probability_a_minus_b_positive_is_zero_l2434_243485


namespace cubic_function_properties_l2434_243463

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_properties :
  ∀ (a b c : ℝ),
  (∀ x : ℝ, f a b c x ≤ f a b c (-1)) ∧  -- Maximum at x = -1
  (f a b c (-1) = 7) ∧                   -- Maximum value is 7
  (∀ x : ℝ, f a b c x ≥ f a b c 3) →     -- Minimum at x = 3
  (a = -3 ∧ b = -9 ∧ c = 2 ∧ f a b c 3 = -25) :=
by sorry

end cubic_function_properties_l2434_243463


namespace exists_square_with_2018_l2434_243448

theorem exists_square_with_2018 : ∃ n : ℕ, ∃ a b : ℕ, n^2 = a * 10000 + 2018 * b :=
  sorry

end exists_square_with_2018_l2434_243448


namespace min_tiles_to_pave_courtyard_l2434_243418

def courtyard_length : ℕ := 378
def courtyard_width : ℕ := 525

def tile_side_length : ℕ := Nat.gcd courtyard_length courtyard_width

def courtyard_area : ℕ := courtyard_length * courtyard_width
def tile_area : ℕ := tile_side_length * tile_side_length

def number_of_tiles : ℕ := courtyard_area / tile_area

theorem min_tiles_to_pave_courtyard :
  number_of_tiles = 450 := by sorry

end min_tiles_to_pave_courtyard_l2434_243418


namespace no_real_roots_min_value_is_three_l2434_243451

-- Define the quadratic function
def f (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + 3

-- Theorem 1: The quadratic equation has no real solutions for any m
theorem no_real_roots (m : ℝ) : ∀ x : ℝ, f m x ≠ 0 := by sorry

-- Theorem 2: The minimum value of the function is 3 for all m
theorem min_value_is_three (m : ℝ) : ∀ x : ℝ, f m x ≥ 3 := by sorry

end no_real_roots_min_value_is_three_l2434_243451


namespace carly_swimming_time_l2434_243422

/-- Calculates the total swimming practice time in a month -/
def monthly_swimming_time (butterfly_hours_per_day : ℕ) (butterfly_days_per_week : ℕ)
                          (backstroke_hours_per_day : ℕ) (backstroke_days_per_week : ℕ)
                          (weeks_in_month : ℕ) : ℕ :=
  ((butterfly_hours_per_day * butterfly_days_per_week) +
   (backstroke_hours_per_day * backstroke_days_per_week)) * weeks_in_month

/-- Proves that Carly spends 96 hours practicing swimming in a month -/
theorem carly_swimming_time :
  monthly_swimming_time 3 4 2 6 4 = 96 :=
by sorry

end carly_swimming_time_l2434_243422


namespace right_triangle_increased_sides_is_acute_l2434_243472

/-- 
Given a right-angled triangle with sides a, b, and c (where c is the hypotenuse),
and a positive real number d, prove that the triangle with sides (a+d), (b+d), and (c+d)
is an acute-angled triangle.
-/
theorem right_triangle_increased_sides_is_acute 
  (a b c d : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Original triangle is right-angled
  (h_pos : d > 0)              -- Increase is positive
  : (a+d)^2 + (b+d)^2 > (c+d)^2 := by
  sorry

end right_triangle_increased_sides_is_acute_l2434_243472


namespace empty_solution_set_range_l2434_243443

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) → 
  -1 < a ∧ a < 0 := by
sorry

end empty_solution_set_range_l2434_243443


namespace bridget_bakery_profit_l2434_243465

/-- Calculates the profit for Bridget's bakery given the specified conditions. -/
def bakery_profit (total_loaves : ℕ) (morning_price afternoon_price late_price : ℚ)
  (operational_cost production_cost : ℚ) : ℚ :=
  let morning_sales := (2 : ℚ) / 5 * total_loaves
  let afternoon_sales := (1 : ℚ) / 2 * (total_loaves - morning_sales)
  let late_sales := (2 : ℚ) / 3 * (total_loaves - morning_sales - afternoon_sales)
  
  let revenue := morning_sales * morning_price + 
                 afternoon_sales * afternoon_price + 
                 late_sales * late_price
  
  let cost := (total_loaves : ℚ) * production_cost + operational_cost
  
  revenue - cost

/-- Theorem stating that under the given conditions, Bridget's bakery profit is $53. -/
theorem bridget_bakery_profit :
  bakery_profit 60 3 (3/2) 2 10 1 = 53 := by
  sorry

#eval bakery_profit 60 3 (3/2) 2 10 1

end bridget_bakery_profit_l2434_243465


namespace circle_equation_proof_l2434_243416

theorem circle_equation_proof (x y : ℝ) : 
  let equation := x^2 + y^2 - 10*y
  let center := (0, 5)
  let radius := 5
  -- The circle's equation
  (equation = 0) →
  -- Center is on the y-axis
  (center.1 = 0) ∧
  -- Circle is tangent to x-axis (distance from center to x-axis equals radius)
  (center.2 = radius) ∧
  -- Circle passes through (3, 1)
  ((3 - center.1)^2 + (1 - center.2)^2 = radius^2) :=
by sorry

end circle_equation_proof_l2434_243416


namespace roberto_outfits_l2434_243408

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ := trousers * shirts * jackets

/-- Theorem: Roberto can create 84 different outfits -/
theorem roberto_outfits :
  let trousers : ℕ := 4
  let shirts : ℕ := 7
  let jackets : ℕ := 3
  number_of_outfits trousers shirts jackets = 84 := by
  sorry

#eval number_of_outfits 4 7 3

end roberto_outfits_l2434_243408


namespace triangle_shape_l2434_243499

theorem triangle_shape (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h3 : a^2 * c^2 + b^2 * c^2 = a^4 - b^4) : 
  a^2 = b^2 + c^2 := by sorry

end triangle_shape_l2434_243499
