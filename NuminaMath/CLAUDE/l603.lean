import Mathlib

namespace equation_solutions_l603_60344

theorem equation_solutions (n : ℕ+) :
  (∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    s.card = 15 ∧ 
    (∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ 3*x + 3*y + z = n)) →
  n = 19 := by
sorry

end equation_solutions_l603_60344


namespace crayons_remaining_l603_60346

/-- Given a drawer with 7 crayons initially, prove that after removing 3 crayons, 4 crayons remain. -/
theorem crayons_remaining (initial : ℕ) (removed : ℕ) (remaining : ℕ) : 
  initial = 7 → removed = 3 → remaining = initial - removed → remaining = 4 := by sorry

end crayons_remaining_l603_60346


namespace simplify_fraction_l603_60309

theorem simplify_fraction : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end simplify_fraction_l603_60309


namespace problem_statement_l603_60375

theorem problem_statement :
  ∀ x y : ℝ,
  x = 98 * 1.2 →
  y = (x + 35) * 0.9 →
  2 * y - 3 * x = -78.12 :=
by
  sorry

end problem_statement_l603_60375


namespace f_2_equals_5_l603_60370

def f (x : ℝ) : ℝ := 2 * (x - 1) + 3

theorem f_2_equals_5 : f 2 = 5 := by
  sorry

end f_2_equals_5_l603_60370


namespace bus_journey_fraction_l603_60308

theorem bus_journey_fraction (total_journey : ℝ) (rail_fraction : ℝ) (foot_distance : ℝ) :
  total_journey = 130 →
  rail_fraction = 3/5 →
  foot_distance = 6.5 →
  (total_journey - (rail_fraction * total_journey + foot_distance)) / total_journey = 45.5 / 130 := by
  sorry

end bus_journey_fraction_l603_60308


namespace ellipse_triangle_perimeter_l603_60340

/-- Given an ellipse with minor axis length 8 and eccentricity 3/5,
    prove that the perimeter of a triangle formed by two points where a line
    through one focus intersects the ellipse and the other focus is 20. -/
theorem ellipse_triangle_perimeter (b : ℝ) (e : ℝ) (a : ℝ) (c : ℝ) 
    (h1 : b = 4)  -- Half of the minor axis length
    (h2 : e = 3/5)  -- Eccentricity
    (h3 : e = c/a)  -- Definition of eccentricity
    (h4 : a^2 = b^2 + c^2)  -- Ellipse equation
    : 4 * a = 20 := by
  sorry

end ellipse_triangle_perimeter_l603_60340


namespace certain_number_proof_l603_60389

theorem certain_number_proof : ∃ x : ℕ, 9873 + x = 13200 ∧ x = 3327 := by
  sorry

end certain_number_proof_l603_60389


namespace foci_coordinates_l603_60342

/-- The curve equation -/
def curve (a x y : ℝ) : Prop :=
  x^2 / (a - 4) + y^2 / (a + 5) = 1

/-- The foci are fixed points -/
def fixed_foci (a : ℝ) : Prop :=
  ∃ x y : ℝ, ∀ b : ℝ, curve b x y → (x, y) = (0, 3) ∨ (x, y) = (0, -3)

/-- Theorem: If the foci of the curve are fixed points, then their coordinates are (0, ±3) -/
theorem foci_coordinates (a : ℝ) :
  fixed_foci a → ∃ x y : ℝ, curve a x y ∧ ((x, y) = (0, 3) ∨ (x, y) = (0, -3)) :=
sorry

end foci_coordinates_l603_60342


namespace bicycle_cost_price_l603_60321

/-- The cost price of a bicycle for seller A, given the selling conditions and final price. -/
theorem bicycle_cost_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 225 →
  ∃ (cost_price_A : ℝ), cost_price_A = 150 :=
by
  sorry

end bicycle_cost_price_l603_60321


namespace quartic_real_root_l603_60352

theorem quartic_real_root 
  (A B C D E : ℝ) 
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
sorry

end quartic_real_root_l603_60352


namespace line_intersects_plane_l603_60319

theorem line_intersects_plane (α : Subspace ℝ (Fin 3 → ℝ)) 
  (a b u : Fin 3 → ℝ) 
  (ha : a ∈ α) (hb : b ∈ α)
  (ha_def : a = ![1, 1/2, 3])
  (hb_def : b = ![1/2, 1, 1])
  (hu_def : u = ![1/2, 0, 1]) :
  ∃ (t : ℝ), (t • u) ∈ α ∧ t • u ≠ 0 := by
  sorry

end line_intersects_plane_l603_60319


namespace double_reflection_of_F_l603_60383

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem double_reflection_of_F (F : ℝ × ℝ) (h : F = (-2, 1)) :
  (reflect_over_x_axis ∘ reflect_over_y_axis) F = (2, -1) := by
  sorry

end double_reflection_of_F_l603_60383


namespace sams_initial_money_l603_60348

/-- Calculates the initial amount of money given the number of books bought, 
    cost per book, and money left after purchase. -/
def initial_money (num_books : ℕ) (cost_per_book : ℕ) (money_left : ℕ) : ℕ :=
  num_books * cost_per_book + money_left

/-- Theorem stating that given the specific conditions of Sam's purchase,
    his initial amount of money was 79 dollars. -/
theorem sams_initial_money : 
  initial_money 9 7 16 = 79 := by
  sorry

#eval initial_money 9 7 16

end sams_initial_money_l603_60348


namespace division_simplification_l603_60339

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by sorry

end division_simplification_l603_60339


namespace line_passes_through_point_l603_60357

/-- The value of b for which the line 2bx + (3b - 2)y = 5b + 6 passes through the point (6, -10) -/
theorem line_passes_through_point : 
  ∃ b : ℚ, b = 14/23 ∧ 2*b*6 + (3*b - 2)*(-10) = 5*b + 6 := by
  sorry

end line_passes_through_point_l603_60357


namespace builder_purchase_cost_l603_60320

/-- Calculates the total cost of a builder's purchase with specific items, taxes, and discounts --/
theorem builder_purchase_cost : 
  let drill_bits_cost : ℚ := 5 * 6
  let hammers_cost : ℚ := 3 * 8
  let toolbox_cost : ℚ := 25
  let nails_cost : ℚ := (50 / 2) * 0.1
  let drill_bits_tax : ℚ := drill_bits_cost * 0.1
  let toolbox_tax : ℚ := toolbox_cost * 0.15
  let hammers_discount : ℚ := hammers_cost * 0.05
  let total_before_discount : ℚ := drill_bits_cost + drill_bits_tax + hammers_cost - hammers_discount + toolbox_cost + toolbox_tax + nails_cost
  let overall_discount : ℚ := if total_before_discount > 60 then total_before_discount * 0.05 else 0
  let final_total : ℚ := total_before_discount - overall_discount
  ∃ (rounded_total : ℚ), (rounded_total ≥ final_total) ∧ (rounded_total < final_total + 0.005) ∧ (rounded_total = 82.70) :=
by sorry


end builder_purchase_cost_l603_60320


namespace min_hits_in_square_l603_60380

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of points in a square -/
def SquareConfiguration := List Point

/-- Function to determine if a point is hit -/
def isHit (config : SquareConfiguration) (p : Point) : Bool := sorry

/-- Function to count the number of hits in a configuration -/
def countHits (config : SquareConfiguration) : Nat :=
  (config.filter (isHit config)).length

/-- Theorem stating the existence of a configuration with minimum 10 hits -/
theorem min_hits_in_square (n : Nat) (h : n = 50) :
  ∃ (config : SquareConfiguration),
    config.length = n ∧
    countHits config = 10 ∧
    ∀ (other_config : SquareConfiguration),
      other_config.length = n →
      countHits other_config ≥ 10 := by
  sorry

end min_hits_in_square_l603_60380


namespace parity_of_p_and_q_l603_60349

theorem parity_of_p_and_q (m n p q : ℤ) :
  Odd m →
  Even n →
  p - 1998 * q = n →
  1999 * p + 3 * q = m →
  (Even p ∧ Odd q) :=
by sorry

end parity_of_p_and_q_l603_60349


namespace parabola_vertex_in_second_quadrant_l603_60347

/-- Represents a parabola of the form y = 2(x-m-1)^2 + 2m + 4 -/
def Parabola (m : ℝ) := λ x : ℝ => 2 * (x - m - 1)^2 + 2 * m + 4

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x (m : ℝ) : ℝ := m + 1

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (m : ℝ) : ℝ := 2 * m + 4

/-- Predicate for a point being in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem parabola_vertex_in_second_quadrant (m : ℝ) :
  in_second_quadrant (vertex_x m) (vertex_y m) ↔ -2 < m ∧ m < -1 :=
sorry

end parabola_vertex_in_second_quadrant_l603_60347


namespace point_on_line_l603_60391

/-- Given a line defined by x = (y / 2) - (2 / 5), if (m, n) and (m + p, n + 4) both lie on this line, then p = 2 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 2 - 2 / 5) →
  (m + p = (n + 4) / 2 - 2 / 5) →
  p = 2 := by sorry

end point_on_line_l603_60391


namespace total_frisbee_distance_l603_60350

/-- The distance Bess can throw the Frisbee -/
def bess_throw_distance : ℕ := 20

/-- The number of times Bess throws the Frisbee -/
def bess_throw_count : ℕ := 4

/-- The distance Holly can throw the Frisbee -/
def holly_throw_distance : ℕ := 8

/-- The number of times Holly throws the Frisbee -/
def holly_throw_count : ℕ := 5

/-- Theorem stating the total distance traveled by both Frisbees -/
theorem total_frisbee_distance : 
  2 * bess_throw_distance * bess_throw_count + holly_throw_distance * holly_throw_count = 200 := by
  sorry

end total_frisbee_distance_l603_60350


namespace line_tangent_to_circle_l603_60318

/-- The line equation √3x - y + m = 0 is tangent to the circle x² + y² - 2x - 2 = 0 
    if and only if m = √3 or m = -3√3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, (Real.sqrt 3 * x - y + m = 0) → 
   (x^2 + y^2 - 2*x - 2 = 0) → 
   (∀ ε > 0, ∃ x' y' : ℝ, 
     x' ≠ x ∧ y' ≠ y ∧ 
     (Real.sqrt 3 * x' - y' + m = 0) ∧ 
     (x'^2 + y'^2 - 2*x' - 2 ≠ 0) ∧
     ((x' - x)^2 + (y' - y)^2 < ε^2))) ↔ 
  (m = Real.sqrt 3 ∨ m = -3 * Real.sqrt 3) :=
sorry

end line_tangent_to_circle_l603_60318


namespace line_equation_through_point_with_inclination_l603_60343

/-- Proves that the equation of a line passing through point (2, -3) with an inclination angle of 45° is x - y - 5 = 0 -/
theorem line_equation_through_point_with_inclination 
  (M : ℝ × ℝ) 
  (h_M : M = (2, -3)) 
  (α : ℝ) 
  (h_α : α = π / 4) : 
  ∀ x y : ℝ, (x - M.1) = (y - M.2) → x - y - 5 = 0 := by
  sorry

end line_equation_through_point_with_inclination_l603_60343


namespace circumcircle_of_triangle_ABC_l603_60364

def A : ℝ × ℝ := (5, 1)
def B : ℝ × ℝ := (7, -3)
def C : ℝ × ℝ := (2, -8)

def circumcircle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 = 25

theorem circumcircle_of_triangle_ABC :
  circumcircle_equation A.1 A.2 ∧
  circumcircle_equation B.1 B.2 ∧
  circumcircle_equation C.1 C.2 := by
  sorry

end circumcircle_of_triangle_ABC_l603_60364


namespace salary_calculation_l603_60354

theorem salary_calculation (salary : ℝ) : 
  salary * (1/5 + 1/10 + 3/5) + 16000 = salary → salary = 160000 := by
  sorry

end salary_calculation_l603_60354


namespace distinct_arrangements_l603_60338

/-- The number of members in the committee -/
def total_members : ℕ := 10

/-- The number of women (rocking chairs) -/
def num_women : ℕ := 7

/-- The number of men (stools) -/
def num_men : ℕ := 2

/-- The number of children (benches) -/
def num_children : ℕ := 1

/-- The number of distinct arrangements of seats -/
def num_arrangements : ℕ := total_members * (total_members - 1) * (total_members - 2) / 2

theorem distinct_arrangements :
  num_arrangements = 360 :=
sorry

end distinct_arrangements_l603_60338


namespace parallel_lines_iff_coplanar_l603_60372

-- Define the types for points and planes
variable (Point Plane : Type*)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the "on_plane" relation for points and planes
variable (on_plane : Point → Plane → Prop)

-- Define the parallel relation for lines (represented by two points each)
variable (parallel_lines : (Point × Point) → (Point × Point) → Prop)

-- Define the coplanar relation for four points
variable (coplanar : Point → Point → Point → Point → Prop)

-- State the theorem
theorem parallel_lines_iff_coplanar
  (α β : Plane) (A B C D : Point)
  (h_planes_parallel : parallel_planes α β)
  (h_A_on_α : on_plane A α)
  (h_C_on_α : on_plane C α)
  (h_B_on_β : on_plane B β)
  (h_D_on_β : on_plane D β) :
  parallel_lines (A, C) (B, D) ↔ coplanar A B C D :=
sorry

end parallel_lines_iff_coplanar_l603_60372


namespace arithmetic_sequence_problem_l603_60387

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h1 : isArithmeticSequence a) 
    (h2 : a 1 + 3 * a 6 + a 11 = 120) : 
  2 * a 7 - a 8 = 24 := by
  sorry

end arithmetic_sequence_problem_l603_60387


namespace number_of_americans_l603_60385

theorem number_of_americans (total : ℕ) (chinese : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : chinese = 22)
  (h3 : australians = 11) :
  total - chinese - australians = 16 := by
  sorry

end number_of_americans_l603_60385


namespace not_all_exp_increasing_l603_60327

-- Define the exponential function
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem not_all_exp_increasing :
  ¬ (∀ (a : ℝ), a > 0 → (∀ (x y : ℝ), x < y → exp a x < exp a y)) :=
by sorry

end not_all_exp_increasing_l603_60327


namespace additional_push_ups_l603_60300

def push_ups (x : ℕ) : ℕ → ℕ
  | 1 => 10
  | 2 => 10 + x
  | 3 => 10 + 2*x
  | _ => 0

theorem additional_push_ups :
  ∃ x : ℕ, (push_ups x 1 + push_ups x 2 + push_ups x 3 = 45) ∧ x = 5 := by
  sorry

end additional_push_ups_l603_60300


namespace cone_base_radius_l603_60353

/-- Given a cone with slant height 6 cm and central angle of unfolded lateral surface 120°,
    prove that the radius of its base is 2 cm. -/
theorem cone_base_radius (slant_height : ℝ) (central_angle : ℝ) :
  slant_height = 6 →
  central_angle = 120 * π / 180 →
  2 * π * slant_height * (central_angle / (2 * π)) = 2 * π * 2 :=
by sorry

end cone_base_radius_l603_60353


namespace luke_coin_piles_l603_60378

theorem luke_coin_piles (piles_quarters piles_dimes : ℕ) 
  (h1 : piles_quarters = piles_dimes)
  (h2 : 3 * piles_quarters + 3 * piles_dimes = 30) : 
  piles_quarters = 5 := by
  sorry

end luke_coin_piles_l603_60378


namespace infinitely_many_divisible_pairs_l603_60303

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

theorem infinitely_many_divisible_pairs :
  ∀ n : ℕ, ∃ a b : ℕ,
    a = fib (2 * n + 1) ∧
    b = fib (2 * n + 3) ∧
    a > 0 ∧
    b > 0 ∧
    a ∣ (b^2 + 1) ∧
    b ∣ (a^2 + 1) :=
by sorry

end infinitely_many_divisible_pairs_l603_60303


namespace jersey_profit_calculation_l603_60315

-- Define the given conditions
def tshirt_profit : ℝ := 25
def tshirts_sold : ℕ := 113
def jerseys_sold : ℕ := 78
def jersey_price_difference : ℝ := 90

-- Define the theorem to be proved
theorem jersey_profit_calculation :
  let jersey_profit := tshirt_profit + jersey_price_difference
  jersey_profit = 115 := by sorry

end jersey_profit_calculation_l603_60315


namespace video_games_spending_l603_60326

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/7
def video_games_fraction : ℚ := 2/7
def snacks_fraction : ℚ := 1/2
def clothes_fraction : ℚ := 3/14

def video_games_spent : ℚ := total_allowance * video_games_fraction

theorem video_games_spending :
  video_games_spent = 7.15 := by sorry

end video_games_spending_l603_60326


namespace bells_toll_together_l603_60351

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 9) (hb : b = 10) (hc : c = 14) (hd : d = 18) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 630 := by
  sorry

end bells_toll_together_l603_60351


namespace smallest_ellipse_area_l603_60333

theorem smallest_ellipse_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
  ((x - 1/2)^2 + y^2 ≥ 1/4 ∧ (x + 1/2)^2 + y^2 ≥ 1/4)) :
  ∃ k : ℝ, k = 4 ∧ π * a * b ≥ k * π := by
  sorry

end smallest_ellipse_area_l603_60333


namespace triple_345_is_right_triangle_l603_60386

/-- A triple of natural numbers representing the sides of a triangle -/
structure TripleNat where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if a triple of natural numbers satisfies the Pythagorean theorem -/
def is_right_triangle (t : TripleNat) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2

/-- The specific triple (3, 4, 5) -/
def triple_345 : TripleNat :=
  { a := 3, b := 4, c := 5 }

/-- Theorem stating that (3, 4, 5) forms a right triangle -/
theorem triple_345_is_right_triangle : is_right_triangle triple_345 := by
  sorry

end triple_345_is_right_triangle_l603_60386


namespace number_puzzle_l603_60302

theorem number_puzzle :
  ∀ (a b : ℤ),
  a + b = 72 →
  a = b + 12 →
  (a = 30 ∨ b = 30) →
  (a = 18 ∨ b = 18) :=
by
  sorry

end number_puzzle_l603_60302


namespace divide_friends_among_teams_l603_60330

theorem divide_friends_among_teams (n : ℕ) (k : ℕ) : 
  n = 8 ∧ k = 3 →
  (k^n : ℕ) - k * ((k-1)^n : ℕ) + (k * (k-1) * (k-2) * 1^n) / 2 = 5796 :=
by sorry

end divide_friends_among_teams_l603_60330


namespace binomial_coefficient_10_3_l603_60314

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by sorry

end binomial_coefficient_10_3_l603_60314


namespace sqrt_450_simplification_l603_60358

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l603_60358


namespace solve_baseball_card_problem_l603_60336

def baseball_card_problem (patricia_money : ℕ) (card_price : ℕ) : Prop :=
  let lisa_money := 5 * patricia_money
  let charlotte_money := lisa_money / 2
  let james_money := 10 + charlotte_money + lisa_money
  let total_money := patricia_money + lisa_money + charlotte_money + james_money
  card_price - total_money = 144

theorem solve_baseball_card_problem :
  baseball_card_problem 6 250 := by
  sorry

end solve_baseball_card_problem_l603_60336


namespace classics_section_books_l603_60307

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- The total number of books in the classics section -/
def total_books : ℕ := num_authors * books_per_author

theorem classics_section_books :
  total_books = 198 := by sorry

end classics_section_books_l603_60307


namespace f_two_equals_two_thirds_l603_60306

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ x / (x + 1)

-- State the theorem
theorem f_two_equals_two_thirds :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  f 2 = 2 / 3 := by
  sorry

end f_two_equals_two_thirds_l603_60306


namespace pages_revised_once_is_35_l603_60334

/-- Represents the manuscript typing problem -/
structure ManuscriptTyping where
  total_pages : ℕ
  pages_revised_twice : ℕ
  first_typing_cost : ℕ
  revision_cost : ℕ
  total_cost : ℕ

/-- Calculates the number of pages revised once -/
def pages_revised_once (m : ManuscriptTyping) : ℕ :=
  ((m.total_cost - m.first_typing_cost * m.total_pages - 
    m.revision_cost * m.pages_revised_twice * 2) / m.revision_cost)

/-- Theorem stating that the number of pages revised once is 35 -/
theorem pages_revised_once_is_35 (m : ManuscriptTyping) 
  (h1 : m.total_pages = 100)
  (h2 : m.pages_revised_twice = 15)
  (h3 : m.first_typing_cost = 6)
  (h4 : m.revision_cost = 4)
  (h5 : m.total_cost = 860) :
  pages_revised_once m = 35 := by
  sorry

#eval pages_revised_once ⟨100, 15, 6, 4, 860⟩

end pages_revised_once_is_35_l603_60334


namespace parallel_neither_sufficient_nor_necessary_l603_60393

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_neither_sufficient_nor_necessary 
  (a b : Line) (α : Plane) 
  (h : line_in_plane b α) :
  ¬(∀ a b α, parallel_lines a b → parallel_line_plane a α) ∧ 
  ¬(∀ a b α, parallel_line_plane a α → parallel_lines a b) :=
sorry

end parallel_neither_sufficient_nor_necessary_l603_60393


namespace element_in_set_l603_60356

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_l603_60356


namespace inequality_proof_l603_60312

open Real

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq_a : exp a = 2 * a * exp (1/2))
  (eq_b : exp b = 3 * b * exp (1/3))
  (eq_c : exp c = 5 * c * exp (1/5)) :
  b * c * exp a < c * a * exp b ∧ c * a * exp b < a * b * exp c :=
by sorry

end inequality_proof_l603_60312


namespace volunteer_distribution_l603_60362

/-- The number of ways to distribute volunteers to pavilions -/
def distribute_volunteers (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- Two specific volunteers cannot be in the same pavilion -/
def separate_volunteers (n : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution :
  distribute_volunteers 5 3 2 - separate_volunteers 3 = 114 :=
sorry

end volunteer_distribution_l603_60362


namespace rhombus_count_in_divided_triangle_l603_60394

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents a rhombus made of smaller equilateral triangles --/
structure Rhombus where
  smallTriangles : ℕ

/-- Counts the number of rhombuses in an equilateral triangle --/
def countRhombuses (triangle : EquilateralTriangle) (rhombus : Rhombus) : ℕ :=
  sorry

/-- Theorem statement --/
theorem rhombus_count_in_divided_triangle 
  (largeTriangle : EquilateralTriangle) 
  (smallTriangle : EquilateralTriangle) 
  (rhombus : Rhombus) :
  largeTriangle.sideLength = 10 →
  smallTriangle.sideLength = 1 →
  rhombus.smallTriangles = 8 →
  (largeTriangle.sideLength / smallTriangle.sideLength) ^ 2 = 100 →
  countRhombuses largeTriangle rhombus = 84 :=
sorry

end rhombus_count_in_divided_triangle_l603_60394


namespace calculation_proof_l603_60360

theorem calculation_proof : (((15^15 / 15^14)^3 * 8^3) / 2^9) = 3375 := by sorry

end calculation_proof_l603_60360


namespace michael_candy_distribution_l603_60337

def minimum_additional_candies (initial_candies : Nat) (num_friends : Nat) : Nat :=
  (num_friends - initial_candies % num_friends) % num_friends

theorem michael_candy_distribution :
  minimum_additional_candies 25 4 = 1 := by
  sorry

end michael_candy_distribution_l603_60337


namespace parabola_rotation_l603_60379

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a point (x, y) by 180 degrees around the origin -/
def rotate180 (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- The original parabola y = x^2 - 6x -/
def original_parabola : Parabola := { a := 1, b := -6, c := 0 }

/-- The rotated parabola y = -(x+3)^2 + 9 -/
def rotated_parabola : Parabola := { a := -1, b := -6, c := 9 }

theorem parabola_rotation :
  ∀ x y : ℝ,
  y = original_parabola.a * x^2 + original_parabola.b * x + original_parabola.c →
  let (x', y') := rotate180 x y
  y' = rotated_parabola.a * x'^2 + rotated_parabola.b * x' + rotated_parabola.c :=
by sorry

end parabola_rotation_l603_60379


namespace solve_age_problem_l603_60355

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 12)

theorem solve_age_problem :
  ∀ a b : ℕ, age_problem a b → b = 42 :=
by
  sorry

end solve_age_problem_l603_60355


namespace joan_pinball_spending_l603_60304

def half_dollar_value : ℚ := 0.5

theorem joan_pinball_spending (wednesday_spent : ℕ) (total_spent : ℚ) 
  (h1 : wednesday_spent = 4)
  (h2 : total_spent = 9)
  : ℕ := by
  sorry

#check joan_pinball_spending

end joan_pinball_spending_l603_60304


namespace dividend_calculation_l603_60313

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 131 := by
  sorry

end dividend_calculation_l603_60313


namespace sally_and_jolly_money_l603_60345

theorem sally_and_jolly_money (total : ℕ) (jolly_plus_20 : ℕ) :
  total = 150 →
  jolly_plus_20 = 70 →
  ∃ (sally : ℕ) (jolly : ℕ),
    sally + jolly = total ∧
    jolly + 20 = jolly_plus_20 ∧
    sally = 100 ∧
    jolly = 50 :=
by sorry

end sally_and_jolly_money_l603_60345


namespace floor_divisibility_l603_60311

theorem floor_divisibility (n : ℕ) : 
  ∃ k : ℤ, (⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ = 2^(n+1) * k) ∧ 
           ¬∃ m : ℤ, (⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ = 2^(n+2) * m) :=
by sorry

end floor_divisibility_l603_60311


namespace cookies_divisible_by_bags_l603_60365

/-- Represents the number of snack bags Destiny can make -/
def num_bags : ℕ := 6

/-- Represents the total number of chocolate candy bars -/
def total_candy_bars : ℕ := 18

/-- Represents the number of cookies Destiny received -/
def num_cookies : ℕ := sorry

/-- Theorem stating that the number of cookies is divisible by the number of bags -/
theorem cookies_divisible_by_bags : num_bags ∣ num_cookies := by sorry

end cookies_divisible_by_bags_l603_60365


namespace max_interesting_in_five_l603_60392

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for prime numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Predicate for interesting numbers -/
def is_interesting (n : ℕ) : Prop := is_prime (sum_of_digits n)

/-- Theorem: At most 4 out of 5 consecutive natural numbers can be interesting -/
theorem max_interesting_in_five (n : ℕ) : 
  ∃ (k : Fin 5), ¬is_interesting (n + k) :=
sorry

end max_interesting_in_five_l603_60392


namespace nonright_angle_is_45_l603_60377

/-- A right isosceles triangle with specific properties -/
structure RightIsoscelesTriangle where
  -- The length of the hypotenuse
  h : ℝ
  -- The height from the right angle to the hypotenuse
  a : ℝ
  -- The product of the hypotenuse and the square of the height is 90
  hyp_height_product : h * a^2 = 90
  -- The triangle is right-angled (implied by being right isosceles)
  right_angled : True
  -- The triangle is isosceles
  isosceles : True

/-- The measure of one of the non-right angles in the triangle -/
def nonRightAngle (t : RightIsoscelesTriangle) : ℝ := 45

/-- Theorem: In a right isosceles triangle where the product of the hypotenuse
    and the square of the height is 90, one of the non-right angles is 45° -/
theorem nonright_angle_is_45 (t : RightIsoscelesTriangle) :
  nonRightAngle t = 45 := by sorry

end nonright_angle_is_45_l603_60377


namespace younger_brother_age_l603_60317

theorem younger_brother_age 
  (older younger : ℕ) 
  (sum_condition : older + younger = 46)
  (age_relation : younger = older / 3 + 10) :
  younger = 19 := by
  sorry

end younger_brother_age_l603_60317


namespace remaining_land_to_clean_l603_60329

theorem remaining_land_to_clean 
  (total_land : ℕ) 
  (lizzie_group : ℕ) 
  (other_group : ℕ) 
  (h1 : total_land = 900) 
  (h2 : lizzie_group = 250) 
  (h3 : other_group = 265) : 
  total_land - (lizzie_group + other_group) = 385 := by
sorry

end remaining_land_to_clean_l603_60329


namespace min_value_inequality_l603_60373

theorem min_value_inequality (a b m n : ℝ) : 
  a > 0 → b > 0 → m > 0 → n > 0 → 
  a + b = 1 → m * n = 2 → 
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end min_value_inequality_l603_60373


namespace complex_fourth_power_l603_60361

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end complex_fourth_power_l603_60361


namespace negative_square_nonpositive_l603_60390

theorem negative_square_nonpositive (a : ℚ) : -a^2 ≤ 0 := by
  sorry

end negative_square_nonpositive_l603_60390


namespace douglas_vote_county_y_l603_60325

theorem douglas_vote_county_y (total_vote_percent : ℝ) (county_x_percent : ℝ) (ratio_x_to_y : ℝ) :
  total_vote_percent = 60 ∧ 
  county_x_percent = 72 ∧ 
  ratio_x_to_y = 2 →
  let county_y_percent := (3 * total_vote_percent - 2 * county_x_percent) / 1
  county_y_percent = 36 := by
sorry

end douglas_vote_county_y_l603_60325


namespace ben_pea_picking_l603_60366

/-- Ben's pea-picking problem -/
theorem ben_pea_picking (P : ℕ) : ∃ (T : ℚ), T = P / 8 :=
  by
  -- Define Ben's picking rates
  have rate1 : (56 : ℚ) / 7 = 8 := by sorry
  have rate2 : (72 : ℚ) / 9 = 8 := by sorry

  -- Prove the theorem
  sorry

end ben_pea_picking_l603_60366


namespace distinctNumbers_eq_2001_l603_60316

/-- The number of distinct numbers in the list [⌊1²/500⌋, ⌊2²/500⌋, ⌊3²/500⌋, ..., ⌊1000²/500⌋] -/
def distinctNumbers : ℕ :=
  let list := List.range 1000
  let floorList := list.map (fun n => Int.floor ((n + 1)^2 / 500 : ℚ))
  floorList.eraseDups.length

/-- The theorem stating that the number of distinct numbers in the list is 2001 -/
theorem distinctNumbers_eq_2001 : distinctNumbers = 2001 := by
  sorry

end distinctNumbers_eq_2001_l603_60316


namespace apple_pie_division_l603_60388

/-- The number of apple pies Sedrach has -/
def total_pies : ℕ := 13

/-- The number of bite-size samples each part of an apple pie can be split into -/
def samples_per_part : ℕ := 5

/-- The total number of people who can taste the pies -/
def total_tasters : ℕ := 130

/-- The number of parts each apple pie is divided into -/
def parts_per_pie : ℕ := 2

theorem apple_pie_division :
  total_pies * parts_per_pie * samples_per_part = total_tasters := by sorry

end apple_pie_division_l603_60388


namespace triangle_inequality_l603_60395

theorem triangle_inequality (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : 5 * a * b * c > a^3 + b^3 + c^3) :
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end triangle_inequality_l603_60395


namespace train_length_problem_l603_60384

/-- Given two trains running in opposite directions, calculate the length of the second train. -/
theorem train_length_problem (length_A : ℝ) (speed_A speed_B : ℝ) (crossing_time : ℝ) :
  length_A = 230 →
  speed_A = 120 * 1000 / 3600 →
  speed_B = 80 * 1000 / 3600 →
  crossing_time = 9 →
  ∃ length_B : ℝ, abs (length_B - 269.95) < 0.01 ∧
    length_A + length_B = (speed_A + speed_B) * crossing_time :=
by sorry

end train_length_problem_l603_60384


namespace train_passing_jogger_time_l603_60332

/-- Time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 190 →
  train_length = 120 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 31 := by
  sorry

#check train_passing_jogger_time

end train_passing_jogger_time_l603_60332


namespace inequality_equivalence_l603_60359

theorem inequality_equivalence (x : ℝ) : (x - 2) / 3 ≤ x ↔ x ≥ -1 := by sorry

end inequality_equivalence_l603_60359


namespace discount_calculation_l603_60369

theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) :
  list_price = 65 →
  final_price = 57.33 →
  first_discount = 10 →
  ∃ second_discount : ℝ,
    second_discount = 2 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end discount_calculation_l603_60369


namespace modulo_equivalence_l603_60374

theorem modulo_equivalence : ∃ n : ℕ, 173 * 927 ≡ n [ZMOD 50] ∧ n < 50 ∧ n = 21 := by
  sorry

end modulo_equivalence_l603_60374


namespace fourth_vertex_not_in_third_quadrant_l603_60301

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem -/
theorem fourth_vertex_not_in_third_quadrant :
  ∀ (p : Parallelogram),
    p.A = ⟨2, 0⟩ →
    p.B = ⟨-1/2, 0⟩ →
    p.C = ⟨0, 1⟩ →
    ¬(isInThirdQuadrant p.D) :=
by sorry

end fourth_vertex_not_in_third_quadrant_l603_60301


namespace abs_z_equals_one_l603_60371

theorem abs_z_equals_one (r : ℝ) (z : ℂ) (h1 : |r| < Real.sqrt 8) (h2 : z + 1/z = r) : 
  Complex.abs z = 1 := by
sorry

end abs_z_equals_one_l603_60371


namespace subcommittees_with_experts_count_l603_60367

def committee_size : ℕ := 12
def expert_count : ℕ := 5
def subcommittee_size : ℕ := 5

theorem subcommittees_with_experts_count : 
  (Nat.choose committee_size subcommittee_size) - 
  (Nat.choose (committee_size - expert_count) subcommittee_size) = 771 := by
  sorry

end subcommittees_with_experts_count_l603_60367


namespace height_equals_base_l603_60323

/-- An isosceles triangle with constant perimeter for inscribed rectangles -/
structure ConstantPerimeterTriangle where
  -- The base of the triangle
  base : ℝ
  -- The height of the triangle
  height : ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The perimeter of any inscribed rectangle is constant
  constantPerimeter : True

/-- Theorem: In a ConstantPerimeterTriangle, the height equals the base -/
theorem height_equals_base (t : ConstantPerimeterTriangle) : t.height = t.base := by
  sorry

end height_equals_base_l603_60323


namespace log_division_simplification_l603_60331

theorem log_division_simplification :
  Real.log 64 / Real.log (1/64) = -1 := by
  sorry

end log_division_simplification_l603_60331


namespace georges_work_hours_l603_60399

/-- George's work problem -/
theorem georges_work_hours (hourly_rate : ℕ) (tuesday_hours : ℕ) (total_earnings : ℕ) :
  hourly_rate = 5 →
  tuesday_hours = 2 →
  total_earnings = 45 →
  ∃ (monday_hours : ℕ), monday_hours = 7 ∧ hourly_rate * (monday_hours + tuesday_hours) = total_earnings :=
by sorry

end georges_work_hours_l603_60399


namespace exterior_angle_of_regular_polygon_l603_60382

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : n > 2) :
  (180 * (n - 2) = 720) → (360 / n = 60) := by
  sorry

end exterior_angle_of_regular_polygon_l603_60382


namespace solve_m_l603_60335

def g (n : Int) : Int :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_m (m : Int) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 15) : m = 55 := by
  sorry

end solve_m_l603_60335


namespace probability_mean_greater_than_median_l603_60396

/-- A fair six-sided die --/
def Die : Type := Fin 6

/-- The result of rolling three dice --/
structure ThreeDiceRoll :=
  (d1 d2 d3 : Die)

/-- The sample space of all possible outcomes when rolling three dice --/
def sampleSpace : Finset ThreeDiceRoll := sorry

/-- The mean of a three dice roll --/
def mean (roll : ThreeDiceRoll) : ℚ := sorry

/-- The median of a three dice roll --/
def median (roll : ThreeDiceRoll) : ℚ := sorry

/-- The event where the mean is greater than the median --/
def meanGreaterThanMedian : Finset ThreeDiceRoll := sorry

theorem probability_mean_greater_than_median :
  (meanGreaterThanMedian.card : ℚ) / sampleSpace.card = 29 / 72 := by sorry

end probability_mean_greater_than_median_l603_60396


namespace complex_fraction_eq_i_l603_60310

def complex (a b : ℝ) := a + b * Complex.I

theorem complex_fraction_eq_i (a b : ℝ) (h : complex a b = Complex.I * (2 - Complex.I)) :
  (complex b a) / (complex a (-b)) = Complex.I := by sorry

end complex_fraction_eq_i_l603_60310


namespace second_part_multiplier_l603_60328

theorem second_part_multiplier (total : ℕ) (first_part : ℕ) (k : ℕ) : 
  total = 36 →
  first_part = 19 →
  8 * first_part + k * (total - first_part) = 203 →
  k = 3 := by sorry

end second_part_multiplier_l603_60328


namespace total_profit_calculation_l603_60397

-- Define the investments and c's profit share
def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def c_profit_share : ℕ := 3000

-- Theorem statement
theorem total_profit_calculation :
  let total_investment := investment_a + investment_b + investment_c
  let profit_ratio_c := investment_c / total_investment
  let total_profit := c_profit_share / profit_ratio_c
  total_profit = 5000 := by sorry

end total_profit_calculation_l603_60397


namespace intersection_complement_l603_60305

def U : Set ℕ := {1, 2, 3, 4}

theorem intersection_complement (A B : Set ℕ) 
  (h1 : (A ∪ B)ᶜ = {4})
  (h2 : B = {1, 2}) : 
  A ∩ Bᶜ = {3} := by
  sorry

end intersection_complement_l603_60305


namespace odd_function_implies_a_equals_two_l603_60363

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a

theorem odd_function_implies_a_equals_two (a : ℝ) :
  (∀ x, f a (x + 1) = -f a (-x + 1)) → a = 2 := by
  sorry

end odd_function_implies_a_equals_two_l603_60363


namespace quadratic_equation_roots_l603_60322

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + m*y - 6 = 0 ∧ y = 3) :=
by sorry

end quadratic_equation_roots_l603_60322


namespace third_term_is_four_l603_60341

/-- A geometric sequence with specific terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  sixth_term : a 6 = 6
  ninth_term : a 9 = 9

/-- The third term of the geometric sequence is 4 -/
theorem third_term_is_four (seq : GeometricSequence) : seq.a 3 = 4 := by
  sorry

end third_term_is_four_l603_60341


namespace max_consecutive_semi_primes_correct_l603_60376

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_semi_prime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

def max_consecutive_semi_primes : ℕ := 5

theorem max_consecutive_semi_primes_correct :
  (∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧
    (∀ k : ℕ, k < max_consecutive_semi_primes → is_semi_prime (m + k))) ∧
  (∀ n : ℕ, ¬∃ m : ℕ, 
    (∀ k : ℕ, k < max_consecutive_semi_primes + 1 → is_semi_prime (m + k))) :=
sorry

end max_consecutive_semi_primes_correct_l603_60376


namespace jill_sandy_make_jack_misses_l603_60398

-- Define the probabilities of making a basket for each person
def jack_prob : ℚ := 1/6
def jill_prob : ℚ := 1/7
def sandy_prob : ℚ := 1/8

-- Define the probability of the desired outcome
def desired_outcome_prob : ℚ := (1 - jack_prob) * jill_prob * sandy_prob

-- Theorem statement
theorem jill_sandy_make_jack_misses :
  desired_outcome_prob = 5/336 := by
  sorry

end jill_sandy_make_jack_misses_l603_60398


namespace math_homework_pages_l603_60324

theorem math_homework_pages (reading : ℕ) (math : ℕ) : 
  math = reading + 3 →
  reading + math = 13 →
  math = 8 := by
sorry

end math_homework_pages_l603_60324


namespace lucy_shell_count_l603_60381

/-- Lucy's shell counting problem -/
theorem lucy_shell_count (initial_shells final_shells : ℕ) 
  (h1 : initial_shells = 68) 
  (h2 : final_shells = 89) : 
  final_shells - initial_shells = 21 := by
  sorry

end lucy_shell_count_l603_60381


namespace quadratic_roots_property_l603_60368

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 5 = 0 → 
  x₂^2 - 4*x₂ - 5 = 0 → 
  (x₁ - 1) * (x₂ - 1) = -8 := by
sorry

end quadratic_roots_property_l603_60368
