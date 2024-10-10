import Mathlib

namespace gavins_green_shirts_l3174_317414

theorem gavins_green_shirts (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) 
  (h1 : total_shirts = 23)
  (h2 : blue_shirts = 6)
  (h3 : green_shirts = total_shirts - blue_shirts) :
  green_shirts = 17 :=
by sorry

end gavins_green_shirts_l3174_317414


namespace limit_to_infinity_l3174_317449

theorem limit_to_infinity (M : ℝ) (h : M > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (2 * n^2 - 3 * n + 2) / (n + 2) > M := by
  sorry

end limit_to_infinity_l3174_317449


namespace wendys_pastries_l3174_317424

/-- Wendy's pastry problem -/
theorem wendys_pastries (cupcakes cookies sold : ℕ) 
  (h1 : cupcakes = 4)
  (h2 : cookies = 29)
  (h3 : sold = 9) :
  cupcakes + cookies - sold = 24 := by
  sorry

end wendys_pastries_l3174_317424


namespace tournament_claim_inconsistency_l3174_317477

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)
  (games_played : ℕ)

/-- Calculates the number of games in a single-elimination tournament -/
def games_in_tournament (t : Tournament) : ℕ := t.participants - 1

/-- Represents the claim made by some players -/
structure Claim :=
  (num_players : ℕ)
  (games_per_player : ℕ)

/-- Calculates the minimum number of games implied by a claim -/
def min_games_from_claim (c : Claim) : ℕ :=
  c.num_players * (c.games_per_player - 1)

/-- The main theorem -/
theorem tournament_claim_inconsistency (t : Tournament) (c : Claim) 
  (h1 : t.participants = 18)
  (h2 : c.num_players = 6)
  (h3 : c.games_per_player = 4) :
  min_games_from_claim c > games_in_tournament t :=
by sorry

end tournament_claim_inconsistency_l3174_317477


namespace x_equals_six_l3174_317404

def floor (y : ℤ) : ℤ :=
  if y % 2 = 0 then y / 2 + 1 else 2 * y + 1

theorem x_equals_six :
  ∃ x : ℤ, floor x * floor 3 = 28 ∧ x = 6 :=
by
  sorry

end x_equals_six_l3174_317404


namespace root_product_expression_l3174_317451

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - p*α + 2 = 0) → 
  (β^2 - p*β + 2 = 0) → 
  (γ^2 + q*γ - 2 = 0) → 
  (δ^2 + q*δ - 2 = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = -2*(p-q)^2 - 4*p*q + 4*q^2 + 16 := by
sorry

end root_product_expression_l3174_317451


namespace value_of_expression_l3174_317461

theorem value_of_expression (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 := by
  sorry

end value_of_expression_l3174_317461


namespace remainder_of_s_1012_l3174_317439

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (x^1012 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^3 + x^2 + x + 1

-- Define s(x) as the polynomial remainder
noncomputable def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_1012 : |s 1012| % 100 = 2 := by
  sorry

end remainder_of_s_1012_l3174_317439


namespace proposition_b_is_true_l3174_317400

theorem proposition_b_is_true : ∀ (a b : ℝ), a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3 := by
  sorry

end proposition_b_is_true_l3174_317400


namespace racing_game_cost_l3174_317473

/-- The cost of the racing game given the total spent and the cost of the basketball game -/
theorem racing_game_cost (total_spent basketball_cost : ℚ) 
  (h1 : total_spent = 9.43)
  (h2 : basketball_cost = 5.2) : 
  total_spent - basketball_cost = 4.23 := by
  sorry

end racing_game_cost_l3174_317473


namespace total_teachers_count_l3174_317422

/-- Given a school with major and minor departments, calculate the total number of teachers -/
theorem total_teachers_count (total_departments : Nat) (major_departments : Nat) (minor_departments : Nat)
  (teachers_per_major : Nat) (teachers_per_minor : Nat)
  (h1 : total_departments = major_departments + minor_departments)
  (h2 : total_departments = 17)
  (h3 : major_departments = 9)
  (h4 : minor_departments = 8)
  (h5 : teachers_per_major = 45)
  (h6 : teachers_per_minor = 29) :
  major_departments * teachers_per_major + minor_departments * teachers_per_minor = 637 := by
  sorry

#check total_teachers_count

end total_teachers_count_l3174_317422


namespace square_equality_l3174_317463

theorem square_equality (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end square_equality_l3174_317463


namespace rachel_plant_arrangement_count_l3174_317499

/-- Represents the types of plants Rachel has -/
inductive Plant
| Basil
| Aloe
| Cactus

/-- Represents the colors of lamps Rachel has -/
inductive LampColor
| White
| Red
| Blue

/-- A configuration of plants under lamps -/
structure Configuration where
  plantUnderLamp : Plant → LampColor

/-- Checks if a configuration is valid according to the given conditions -/
def isValidConfiguration (config : Configuration) : Prop :=
  -- Each plant is under exactly one lamp
  (∀ p : Plant, ∃! l : LampColor, config.plantUnderLamp p = l) ∧
  -- No lamp is used for just one plant unless it's the red one
  (∀ l : LampColor, l ≠ LampColor.Red → (∃ p₁ p₂ : Plant, p₁ ≠ p₂ ∧ config.plantUnderLamp p₁ = l ∧ config.plantUnderLamp p₂ = l))

/-- The number of valid configurations -/
def validConfigurationsCount : ℕ := sorry

theorem rachel_plant_arrangement_count :
  validConfigurationsCount = 4 := by sorry

end rachel_plant_arrangement_count_l3174_317499


namespace max_xy_value_l3174_317420

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a/3 + b/4 = 1 → x*y ≥ a*b ∧ x*y ≤ 3 :=
by sorry

end max_xy_value_l3174_317420


namespace not_equivalent_fraction_l3174_317415

theorem not_equivalent_fraction (x : ℝ) : x = 0.00000325 → x ≠ 1 / 308000000 := by
  sorry

end not_equivalent_fraction_l3174_317415


namespace gift_purchase_solution_l3174_317441

/-- Pricing function based on quantity --/
def price (q : ℕ) : ℚ :=
  if q ≤ 120 then 3.5
  else if q ≤ 300 then 3.2
  else 3

/-- Total cost for a given quantity --/
def total_cost (q : ℕ) : ℚ :=
  if q ≤ 120 then q * price q
  else if q ≤ 300 then 120 * 3.5 + (q - 120) * price q
  else 120 * 3.5 + 180 * 3.2 + (q - 300) * price q

/-- Theorem stating the correctness of the solution --/
theorem gift_purchase_solution :
  let xiaoli_units : ℕ := 290
  let xiaowang_units : ℕ := 110
  xiaoli_units + xiaowang_units = 400 ∧
  xiaoli_units > 280 ∧
  total_cost xiaoli_units + total_cost xiaowang_units = 1349 :=
by sorry

end gift_purchase_solution_l3174_317441


namespace stickers_after_birthday_l3174_317423

def initial_stickers : ℕ := 39
def birthday_stickers : ℕ := 22

theorem stickers_after_birthday :
  initial_stickers + birthday_stickers = 61 := by
  sorry

end stickers_after_birthday_l3174_317423


namespace triangle_height_inequality_l3174_317488

/-- Given a triangle ABC with sides a, b, c and heights h_a, h_b, h_c, 
    the sum of squares of heights divided by squares of sides is at most 9/2. -/
theorem triangle_height_inequality (a b c h_a h_b h_c : ℝ) 
    (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_pos_ha : h_a > 0) (h_pos_hb : h_b > 0) (h_pos_hc : h_c > 0)
    (h_triangle : a * h_a = b * h_b ∧ b * h_b = c * h_c) : 
    (h_b^2 + h_c^2) / a^2 + (h_c^2 + h_a^2) / b^2 + (h_a^2 + h_b^2) / c^2 ≤ 9/2 := by
  sorry

end triangle_height_inequality_l3174_317488


namespace min_value_a_squared_plus_4b_squared_l3174_317406

theorem min_value_a_squared_plus_4b_squared (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 2/a + 1/b = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 1 → a^2 + 4*b^2 ≤ x^2 + 4*y^2 :=
by sorry

end min_value_a_squared_plus_4b_squared_l3174_317406


namespace kayla_apples_l3174_317447

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 200 →
  total = kylie + kayla →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end kayla_apples_l3174_317447


namespace flea_landing_product_l3174_317407

/-- The number of circles in the arrangement -/
def num_circles : ℕ := 12

/-- The number of steps the red flea takes clockwise -/
def red_steps : ℕ := 1991

/-- The number of steps the black flea takes counterclockwise -/
def black_steps : ℕ := 1949

/-- The final position of a flea after taking a number of steps -/
def final_position (steps : ℕ) : ℕ :=
  steps % num_circles

/-- The position of the black flea, adjusted for counterclockwise movement -/
def black_position : ℕ :=
  num_circles - (final_position black_steps)

theorem flea_landing_product :
  final_position red_steps * black_position = 77 := by
  sorry

end flea_landing_product_l3174_317407


namespace copy_pages_proof_l3174_317492

/-- The cost in cents to copy a single page -/
def cost_per_page : ℚ := 25/10

/-- The amount of money available in dollars -/
def available_money : ℚ := 20

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of pages that can be copied with the available money -/
def pages_copied : ℕ := 800

theorem copy_pages_proof : 
  (available_money * cents_per_dollar) / cost_per_page = pages_copied := by
  sorry

end copy_pages_proof_l3174_317492


namespace sqrt_sum_comparison_l3174_317479

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 6 > Real.sqrt 2 + Real.sqrt 7 := by
  sorry

end sqrt_sum_comparison_l3174_317479


namespace book_student_difference_l3174_317437

/-- Proves that in 5 classrooms, where each classroom has 18 students and each student has 3 books,
    the difference between the total number of books and the total number of students is 180. -/
theorem book_student_difference :
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 18
  let books_per_student : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_books : ℕ := total_students * books_per_student
  total_books - total_students = 180 :=
by
  sorry


end book_student_difference_l3174_317437


namespace most_cost_effective_option_l3174_317455

/-- Represents the cost calculation for tea sets and bowls under different offers -/
def cost_calculation (tea_set_price : ℕ) (tea_bowl_price : ℕ) (num_sets : ℕ) (num_bowls : ℕ) : ℕ → ℕ
| 1 => tea_set_price * num_sets + tea_bowl_price * (num_bowls - num_sets)  -- Offer 1
| 2 => (tea_set_price * num_sets * 95 + tea_bowl_price * num_bowls * 95) / 100  -- Offer 2
| _ => 0  -- Invalid offer

theorem most_cost_effective_option 
  (tea_set_price : ℕ) 
  (tea_bowl_price : ℕ) 
  (num_sets : ℕ) 
  (num_bowls : ℕ) 
  (h1 : tea_set_price = 200)
  (h2 : tea_bowl_price = 20)
  (h3 : num_sets = 30)
  (h4 : num_bowls = 40)
  (h5 : num_bowls > num_sets) :
  let offer1_cost := cost_calculation tea_set_price tea_bowl_price num_sets num_bowls 1
  let offer2_cost := cost_calculation tea_set_price tea_bowl_price num_sets num_bowls 2
  let combined_offer_cost := tea_set_price * num_sets + 
                             (cost_calculation tea_set_price tea_bowl_price (num_bowls - num_sets) (num_bowls - num_sets) 2)
  combined_offer_cost < min offer1_cost offer2_cost ∧ combined_offer_cost = 6190 :=
by sorry

end most_cost_effective_option_l3174_317455


namespace bus_distribution_solution_l3174_317416

/-- Represents the problem of distributing passengers among buses --/
structure BusDistribution where
  k : ℕ  -- Original number of buses
  n : ℕ  -- Number of passengers per bus after redistribution
  max_capacity : ℕ  -- Maximum capacity of each bus

/-- The conditions of the bus distribution problem --/
def valid_distribution (bd : BusDistribution) : Prop :=
  bd.k ≥ 2 ∧
  bd.n ≤ bd.max_capacity ∧
  22 * bd.k + 1 = bd.n * (bd.k - 1)

/-- The theorem stating the solution to the bus distribution problem --/
theorem bus_distribution_solution :
  ∃ (bd : BusDistribution),
    bd.max_capacity = 32 ∧
    valid_distribution bd ∧
    bd.k = 24 ∧
    bd.n * (bd.k - 1) = 529 :=
sorry


end bus_distribution_solution_l3174_317416


namespace min_distance_squared_l3174_317485

noncomputable def e : ℝ := Real.exp 1

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : b = a - 2 * e^a) 
  (h2 : c + d = 4) : 
  ∃ (min : ℝ), min = 18 ∧ ∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ min :=
sorry

end min_distance_squared_l3174_317485


namespace expression_simplification_l3174_317419

/-- Proves that the given expression simplifies to 1 when a = 1 and b = -2 -/
theorem expression_simplification (a b : ℤ) (ha : a = 1) (hb : b = -2) :
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 := by
  sorry


end expression_simplification_l3174_317419


namespace N_is_k_times_sum_of_digits_l3174_317430

/-- A number consisting of k nines -/
def N (k : ℕ) : ℕ := 10^k - 1

/-- The sum of digits of a number consisting of k nines -/
def sum_of_digits (k : ℕ) : ℕ := 9 * k

/-- Theorem stating that N(k) is k times greater than the sum of its digits for all natural k -/
theorem N_is_k_times_sum_of_digits (k : ℕ) :
  N k = k * (sum_of_digits k) :=
sorry

end N_is_k_times_sum_of_digits_l3174_317430


namespace train_length_l3174_317460

theorem train_length (pole_time : ℝ) (tunnel_length tunnel_time : ℝ) :
  pole_time = 20 →
  tunnel_length = 500 →
  tunnel_time = 40 →
  ∃ (train_length : ℝ),
    train_length = pole_time * (train_length + tunnel_length) / tunnel_time ∧
    train_length = 500 := by
  sorry

end train_length_l3174_317460


namespace max_third_term_in_arithmetic_sequence_l3174_317432

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem max_third_term_in_arithmetic_sequence :
  ∀ a b c d : ℕ,
  a > 0 → b > 0 → c > 0 → d > 0 →
  is_arithmetic_sequence a b c d →
  a + b + c + d = 50 →
  c ≤ 16 :=
by sorry

end max_third_term_in_arithmetic_sequence_l3174_317432


namespace nuts_left_l3174_317421

theorem nuts_left (total : ℕ) (eaten_fraction : ℚ) (h1 : total = 30) (h2 : eaten_fraction = 5/6) :
  total - (total * eaten_fraction).floor = 5 := by
  sorry

end nuts_left_l3174_317421


namespace hyperbola_triangle_area_l3174_317417

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (P : ℝ × ℝ) : Prop :=
  let F₁ := left_focus
  let F₂ := right_focus
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Theorem statement
theorem hyperbola_triangle_area (P : ℝ × ℝ) :
  point_on_hyperbola P → right_angle P → 
  let F₁ := left_focus
  let F₂ := right_focus
  let area := (1/2) * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt * 
              ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt
  area = 16 := by
  sorry

end hyperbola_triangle_area_l3174_317417


namespace quadratic_equation_real_root_l3174_317464

theorem quadratic_equation_real_root (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by sorry

end quadratic_equation_real_root_l3174_317464


namespace range_of_u_l3174_317444

theorem range_of_u (x y : ℝ) (h : x^2/3 + y^2 = 1) :
  1 ≤ |2*x + y - 4| + |3 - x - 2*y| ∧ |2*x + y - 4| + |3 - x - 2*y| ≤ 13 := by
  sorry

end range_of_u_l3174_317444


namespace alcohol_concentration_in_mixture_l3174_317466

/-- Calculates the new concentration of alcohol in a mixture --/
theorem alcohol_concentration_in_mixture
  (vessel1_capacity : ℝ)
  (vessel1_concentration : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_concentration : ℝ)
  (total_liquid : ℝ)
  (new_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_concentration = 0.4)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_concentration = 0.6)
  (h5 : total_liquid = 8)
  (h6 : new_vessel_capacity = 10)
  (h7 : total_liquid ≤ new_vessel_capacity) :
  let alcohol1 := vessel1_capacity * vessel1_concentration
  let alcohol2 := vessel2_capacity * vessel2_concentration
  let total_alcohol := alcohol1 + alcohol2
  let water_added := new_vessel_capacity - total_liquid
  let new_concentration := total_alcohol / new_vessel_capacity
  new_concentration = 0.44 := by
  sorry

#check alcohol_concentration_in_mixture

end alcohol_concentration_in_mixture_l3174_317466


namespace expression_value_l3174_317412

theorem expression_value : (40 + 15)^2 - 15^2 = 2800 := by
  sorry

end expression_value_l3174_317412


namespace base_subtraction_l3174_317467

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Express 343₈ - 265₇ as a base 10 integer -/
theorem base_subtraction : 
  let base_8_num := to_base_10 [3, 4, 3] 8
  let base_7_num := to_base_10 [2, 6, 5] 7
  base_8_num - base_7_num = 82 := by
sorry

end base_subtraction_l3174_317467


namespace election_majority_l3174_317490

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6900 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1380 :=
by sorry

end election_majority_l3174_317490


namespace sphere_volume_from_surface_area_l3174_317431

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * Real.pi * r^2 = 36 * Real.pi →
  (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end sphere_volume_from_surface_area_l3174_317431


namespace rectangle_area_l3174_317484

/-- A rectangle with one side of length 4 and a diagonal of length 5 has an area of 12. -/
theorem rectangle_area (w l d : ℝ) (hw : w = 4) (hd : d = 5) (h_pythagorean : w^2 + l^2 = d^2) : w * l = 12 := by
  sorry

end rectangle_area_l3174_317484


namespace translated_parabola_vertex_l3174_317459

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The translation amount to the right -/
def translation : ℝ := 2

/-- The new parabola function after translation -/
def f_translated (x : ℝ) : ℝ := f (x - translation)

/-- Theorem stating the coordinates of the vertex of the translated parabola -/
theorem translated_parabola_vertex :
  ∃ (x y : ℝ), x = 4 ∧ y = -1 ∧
  ∀ (t : ℝ), f_translated t ≥ f_translated x :=
sorry

end translated_parabola_vertex_l3174_317459


namespace line_perpendicular_to_plane_l3174_317474

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (lies_in : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (intersection_line : Plane → Plane → Line)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α β : Plane) (a c : Line) :
  perpendicular_planes α β →
  lies_in a α →
  c = intersection_line α β →
  perpendicular_lines a c →
  perpendicular_line_plane a β :=
sorry

end line_perpendicular_to_plane_l3174_317474


namespace max_median_redistribution_l3174_317482

theorem max_median_redistribution (x : ℕ) :
  let initial_amounts : List ℕ := [28, 72, 98, x]
  let total : ℕ := initial_amounts.sum
  let redistributed : ℚ := (total : ℚ) / 4
  (∀ (a : ℕ), a ∈ initial_amounts → (a : ℚ) ≤ redistributed) →
  redistributed ≤ 98 →
  x ≤ 194 →
  (x = 194 → redistributed = 98) :=
by sorry

end max_median_redistribution_l3174_317482


namespace simplify_expression_l3174_317468

theorem simplify_expression (a : ℝ) : 6*a - 5*a + 4*a - 3*a + 2*a - a = 3*a := by
  sorry

end simplify_expression_l3174_317468


namespace ben_spending_correct_l3174_317427

/-- Calculates Ben's spending at the bookstore with given prices and discounts --/
def benSpending (notebookPrice magazinePrice penPrice bookPrice : ℚ)
                (notebookCount magazineCount penCount bookCount : ℕ)
                (penDiscount membershipDiscount membershipThreshold : ℚ) : ℚ :=
  let subtotal := notebookPrice * notebookCount +
                  magazinePrice * magazineCount +
                  penPrice * (1 - penDiscount) * penCount +
                  bookPrice * bookCount
  if subtotal ≥ membershipThreshold then
    subtotal - membershipDiscount
  else
    subtotal

/-- Theorem stating that Ben's spending matches the calculated amount --/
theorem ben_spending_correct :
  benSpending 2 6 1.5 12 4 3 5 2 0.25 10 50 = 45.625 := by sorry

end ben_spending_correct_l3174_317427


namespace prob_at_least_twice_eq_target_l3174_317497

/-- The probability of hitting a target in one shot -/
def p : ℝ := 0.6

/-- The number of shots taken -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots -/
def prob_at_least_twice (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * p^2 * (1 - p) + (n.choose 3) * p^3

theorem prob_at_least_twice_eq_target : 
  prob_at_least_twice p n = 0.648 := by
sorry

end prob_at_least_twice_eq_target_l3174_317497


namespace unique_maintaining_value_interval_for_square_maintaining_value_intervals_for_square_plus_constant_l3174_317487

/-- Definition of a "maintaining value" interval for a function f on [a,b] --/
def is_maintaining_value_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  Monotone f ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

/-- The square function --/
def f (x : ℝ) : ℝ := x^2

/-- The square function with constant --/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + m

/-- Theorem: [0,1] is the only "maintaining value" interval for f(x) = x^2 --/
theorem unique_maintaining_value_interval_for_square :
  ∀ a b : ℝ, is_maintaining_value_interval f a b ↔ a = 0 ∧ b = 1 :=
sorry

/-- Theorem: Characterization of "maintaining value" intervals for g(x) = x^2 + m --/
theorem maintaining_value_intervals_for_square_plus_constant :
  ∀ m : ℝ, m ≠ 0 →
  (∃ a b : ℝ, is_maintaining_value_interval (g m) a b) ↔ 
  (m ∈ Set.Icc (-1) (-3/4) ∪ Set.Ioc 0 (1/4)) :=
sorry

end unique_maintaining_value_interval_for_square_maintaining_value_intervals_for_square_plus_constant_l3174_317487


namespace cos_135_degrees_l3174_317446

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_degrees_l3174_317446


namespace gcd_16_12_l3174_317486

def operation : List (ℕ × ℕ) := [(16, 12), (12, 4), (8, 4), (4, 4)]

theorem gcd_16_12 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_l3174_317486


namespace elevator_theorem_l3174_317483

/-- Represents the elevator system described in the problem -/
structure ElevatorSystem where
  /-- The probability of moving up on the nth press is current_floor / (n-1) -/
  move_up_prob : (current_floor : ℕ) → (n : ℕ) → ℚ
  move_up_prob_def : ∀ (current_floor n : ℕ), move_up_prob current_floor n = current_floor / (n - 1)

/-- The expected number of pairs of consecutive presses that both move up -/
def expected_consecutive_up_pairs (system : ElevatorSystem) (start_press end_press : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem elevator_theorem (system : ElevatorSystem) :
  expected_consecutive_up_pairs system 3 100 = 97 / 3 := by sorry

end elevator_theorem_l3174_317483


namespace quadratic_equations_solutions_l3174_317411

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ 
   x₁^2 - 8*x₁ + 1 = 0 ∧ x₂^2 - 8*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 ∧ y₂ = 1 ∧
   y₁*(y₁ - 2) - y₁ + 2 = 0 ∧ y₂*(y₂ - 2) - y₂ + 2 = 0) :=
by sorry

end quadratic_equations_solutions_l3174_317411


namespace trash_bin_charge_is_10_l3174_317443

/-- Represents the garbage bill calculation -/
def garbage_bill (T : ℚ) : Prop :=
  let weeks : ℕ := 4
  let trash_bins : ℕ := 2
  let recycling_bins : ℕ := 1
  let recycling_charge : ℚ := 5
  let discount_rate : ℚ := 0.18
  let fine : ℚ := 20
  let final_bill : ℚ := 102

  let pre_discount := weeks * (trash_bins * T + recycling_bins * recycling_charge)
  let discount := discount_rate * pre_discount
  let post_discount := pre_discount - discount
  let total_bill := post_discount + fine

  total_bill = final_bill

/-- Theorem stating that the charge per trash bin is $10 -/
theorem trash_bin_charge_is_10 : garbage_bill 10 := by
  sorry

end trash_bin_charge_is_10_l3174_317443


namespace absolute_value_equation_l3174_317462

theorem absolute_value_equation (y : ℝ) :
  (|y - 25| + |y - 23| = |2*y - 46|) → y = 24 := by
  sorry

end absolute_value_equation_l3174_317462


namespace largest_non_expressible_not_expressible_83_largest_non_expressible_is_83_l3174_317472

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ k c, k > 0 ∧ is_composite c ∧ n = 36 * k + c

theorem largest_non_expressible : ∀ n : ℕ, n > 83 → is_expressible n :=
  sorry

theorem not_expressible_83 : ¬ is_expressible 83 :=
  sorry

theorem largest_non_expressible_is_83 :
  (∀ n : ℕ, n > 83 → is_expressible n) ∧ ¬ is_expressible 83 :=
  sorry

end largest_non_expressible_not_expressible_83_largest_non_expressible_is_83_l3174_317472


namespace power_comparison_l3174_317448

theorem power_comparison : (2 : ℕ)^16 / (16 : ℕ)^2 = 256 := by sorry

end power_comparison_l3174_317448


namespace prime_factorization_property_l3174_317405

theorem prime_factorization_property (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ y : ℕ, y ≤ p / 2 ∧ ¬∃ (a b : ℕ), a > y ∧ b > y ∧ p * y + 1 = a * b :=
by sorry

end prime_factorization_property_l3174_317405


namespace arrange_5_balls_4_boxes_l3174_317442

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def arrange_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to put 5 distinguishable balls into 4 distinguishable boxes -/
theorem arrange_5_balls_4_boxes : arrange_balls 5 4 = 1024 := by
  sorry

end arrange_5_balls_4_boxes_l3174_317442


namespace bank_teller_coins_l3174_317495

theorem bank_teller_coins (rolls_per_teller : ℕ) (coins_per_roll : ℕ) (num_tellers : ℕ) :
  rolls_per_teller = 10 →
  coins_per_roll = 25 →
  num_tellers = 4 →
  rolls_per_teller * coins_per_roll * num_tellers = 1000 :=
by
  sorry

end bank_teller_coins_l3174_317495


namespace f_symmetric_about_origin_l3174_317433

def f (x : ℝ) : ℝ := x^3 + x

theorem f_symmetric_about_origin : ∀ x : ℝ, f (-x) = -f x := by sorry

end f_symmetric_about_origin_l3174_317433


namespace simplify_expression_l3174_317457

theorem simplify_expression (y : ℝ) : 3 * y + 4.5 * y + 7 * y = 14.5 * y := by
  sorry

end simplify_expression_l3174_317457


namespace f_increasing_on_positive_reals_l3174_317476

def f (x : ℝ) : ℝ := x^2 + x

theorem f_increasing_on_positive_reals :
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end f_increasing_on_positive_reals_l3174_317476


namespace cube_difference_l3174_317402

theorem cube_difference (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x - y = 3) (h4 : x + y = 5) : x^3 - y^3 = 63 := by
  sorry

end cube_difference_l3174_317402


namespace pages_per_chapter_l3174_317408

theorem pages_per_chapter 
  (total_pages : ℕ) 
  (num_chapters : ℕ) 
  (h1 : total_pages = 555) 
  (h2 : num_chapters = 5) 
  (h3 : total_pages % num_chapters = 0) : 
  total_pages / num_chapters = 111 := by
  sorry

end pages_per_chapter_l3174_317408


namespace perpendicular_necessary_not_sufficient_l3174_317438

structure GeometricSpace where
  Line : Type
  Plane : Type
  perpendicular : Line → Line → Prop
  parallel : Line → Plane → Prop
  perpendicular_plane : Line → Plane → Prop

variable (S : GeometricSpace)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem perpendicular_necessary_not_sufficient
  (l m : S.Line) (α : S.Plane)
  (h1 : l ≠ m)
  (h2 : S.perpendicular_plane m α) :
  necessary_not_sufficient (S.perpendicular l m) (S.parallel l α) := by
  sorry

end perpendicular_necessary_not_sufficient_l3174_317438


namespace equality_of_absolute_value_sums_l3174_317435

theorem equality_of_absolute_value_sums (a b c d : ℝ) 
  (h : ∀ x : ℝ, |2*x + 4| + |a*x + b| = |c*x + d|) : 
  d = 2*c := by sorry

end equality_of_absolute_value_sums_l3174_317435


namespace tan_seventeen_pi_over_four_l3174_317470

theorem tan_seventeen_pi_over_four : Real.tan (17 * π / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l3174_317470


namespace percent_problem_l3174_317428

theorem percent_problem (x : ℝ) (h : 120 = 0.75 * x) : x = 160 := by
  sorry

end percent_problem_l3174_317428


namespace calculation_equality_algebraic_simplification_l3174_317494

-- Part 1
theorem calculation_equality : (-(1/3))⁻¹ + (2015 - Real.sqrt 3)^0 - 4 * Real.sin (60 * π / 180) + |(- Real.sqrt 12)| = -2 := by sorry

-- Part 2
theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  ((1 / (a + b) - 1 / (a - b)) / (b / (a^2 - 2*a*b + b^2))) = -2*(a - b)/(a + b) := by sorry

end calculation_equality_algebraic_simplification_l3174_317494


namespace connor_score_l3174_317478

theorem connor_score (connor amy jason : ℕ) : 
  (amy = connor + 4) →
  (jason = 2 * amy) →
  (connor + amy + jason = 20) →
  connor = 2 := by sorry

end connor_score_l3174_317478


namespace evaluate_expression_l3174_317480

theorem evaluate_expression (x y : ℚ) (hx : x = 4/8) (hy : y = 5/6) :
  (8*x + 6*y) / (72*x*y) = 3/10 := by
  sorry

end evaluate_expression_l3174_317480


namespace probability_n_less_than_m_plus_one_l3174_317429

/-- The number of balls in the bag -/
def num_balls : ℕ := 4

/-- The set of possible ball numbers -/
def ball_numbers : Finset ℕ := Finset.range num_balls

/-- The sample space of all possible outcomes (m, n) -/
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product ball_numbers ball_numbers

/-- The event where n < m + 1 -/
def event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.2 < p.1 + 1)

/-- The probability of the event -/
noncomputable def probability : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

theorem probability_n_less_than_m_plus_one :
  probability = 5/8 := by sorry

end probability_n_less_than_m_plus_one_l3174_317429


namespace range_of_x_l3174_317489

theorem range_of_x (a : ℝ) (h1 : a > 1) :
  {x : ℝ | a^(2*x + 1) > (1/a)^(2*x)} = {x : ℝ | x > -1/4} := by sorry

end range_of_x_l3174_317489


namespace problem_statement_l3174_317471

variables (a b c : ℝ)

def f (x : ℝ) := a * x^2 + b * x + c
def g (x : ℝ) := a * x + b

theorem problem_statement :
  (∀ x : ℝ, abs x ≤ 1 → abs (f a b c x) ≤ 1) →
  (abs c ≤ 1 ∧ ∀ x : ℝ, abs x ≤ 1 → abs (g a b x) ≤ 2) :=
by sorry

end problem_statement_l3174_317471


namespace quadratic_roots_expression_l3174_317491

theorem quadratic_roots_expression (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) →
  (3 * q ^ 2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 14 := by
sorry

end quadratic_roots_expression_l3174_317491


namespace parabola_translation_l3174_317481

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- Theorem stating that the translated parabola is correct
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end parabola_translation_l3174_317481


namespace max_sum_of_factors_l3174_317410

theorem max_sum_of_factors (p q : ℕ+) (h : p * q = 100) : 
  ∃ (a b : ℕ+), a * b = 100 ∧ a + b ≤ p + q ∧ a + b = 101 :=
sorry

end max_sum_of_factors_l3174_317410


namespace value_of_4x2y2_l3174_317426

theorem value_of_4x2y2 (x y : ℤ) (h : y^2 + 4*x^2*y^2 = 40*x^2 + 817) : 
  4*x^2*y^2 = 3484 := by sorry

end value_of_4x2y2_l3174_317426


namespace smallest_dual_base_palindrome_l3174_317465

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
    n > 5 →
    isPalindrome n 2 →
    isPalindrome n 4 →
    (∀ m : ℕ, m > 5 ∧ m < n → ¬(isPalindrome m 2 ∧ isPalindrome m 4)) →
    n = 15 :=
  sorry

end smallest_dual_base_palindrome_l3174_317465


namespace divisibility_condition_l3174_317456

theorem divisibility_condition (M : ℕ) : 
  0 < M ∧ M < 10 → (5 ∣ 1989^M + M^1889 ↔ M = 1 ∨ M = 4) := by
  sorry

end divisibility_condition_l3174_317456


namespace mans_rowing_speed_l3174_317409

/-- Proves that a man's rowing speed in still water is 15 kmph given the conditions of downstream rowing --/
theorem mans_rowing_speed (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 3 →
  distance = 70 →
  time = 13.998880089592832 →
  (distance / time - current_speed * 1000 / 3600) * 3.6 = 15 := by
  sorry

end mans_rowing_speed_l3174_317409


namespace largest_constant_inequality_l3174_317425

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (D : ℝ), D = Real.sqrt (12 / 17) ∧
  (∀ (x y : ℝ), x^2 + 2*y^2 + 3 ≥ D*(3*x + 4*y)) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 + 3 ≥ D'*(3*x + 4*y)) → D' ≤ D) :=
sorry

end largest_constant_inequality_l3174_317425


namespace xyz_product_l3174_317475

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -25)
  (eq2 : y * z + 5 * z = -25)
  (eq3 : z * x + 5 * x = -25) :
  x * y * z = 125 := by
sorry

end xyz_product_l3174_317475


namespace triangle_theorem_l3174_317440

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * t.b - t.a) * Real.cos (t.A + t.B) = -t.c * Real.cos t.A)
  (h2 : t.c = 3)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (4 * Real.sqrt 3) / 3) :
  t.C = π/3 ∧ t.a + t.b = 5 := by
  sorry


end triangle_theorem_l3174_317440


namespace sum_squares_consecutive_even_numbers_l3174_317469

/-- Given 6 consecutive even numbers with a sum of 72, prove that the sum of their squares is 1420 -/
theorem sum_squares_consecutive_even_numbers :
  ∀ (a : ℕ), 
  (∃ (n : ℕ), a = 2*n) →  -- a is even
  (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) + (a + 10) = 72) →  -- sum is 72
  (a^2 + (a + 2)^2 + (a + 4)^2 + (a + 6)^2 + (a + 8)^2 + (a + 10)^2 = 1420) :=
by sorry


end sum_squares_consecutive_even_numbers_l3174_317469


namespace angle_sum_is_pi_over_two_l3174_317403

theorem angle_sum_is_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end angle_sum_is_pi_over_two_l3174_317403


namespace expression_simplification_and_evaluation_l3174_317454

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4*a + 4) / (a + 1)) = -(a + 2) / (a - 2) ∧
  (-(1 + 2) / (1 - 2) = 3) := by
sorry

end expression_simplification_and_evaluation_l3174_317454


namespace angle_measure_proof_l3174_317452

/-- Two angles are supplementary if their measures sum to 180 degrees -/
def Supplementary (a b : ℝ) : Prop := a + b = 180

theorem angle_measure_proof (A B : ℝ) 
  (h1 : Supplementary A B) 
  (h2 : A = 8 * B) : 
  A = 160 := by
  sorry

end angle_measure_proof_l3174_317452


namespace greatest_divisor_630_under_60_and_factor_90_l3174_317493

def is_greatest_divisor (n : ℕ) : Prop :=
  n ∣ 630 ∧ n < 60 ∧ n ∣ 90 ∧
  ∀ m : ℕ, m ∣ 630 → m < 60 → m ∣ 90 → m ≤ n

theorem greatest_divisor_630_under_60_and_factor_90 :
  is_greatest_divisor 45 := by
  sorry

end greatest_divisor_630_under_60_and_factor_90_l3174_317493


namespace a_lt_2_necessary_not_sufficient_for_a_sq_lt_4_l3174_317496

theorem a_lt_2_necessary_not_sufficient_for_a_sq_lt_4 :
  (∀ a : ℝ, a^2 < 4 → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 4) := by
  sorry

end a_lt_2_necessary_not_sufficient_for_a_sq_lt_4_l3174_317496


namespace log_equation_solution_l3174_317450

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = 3^(10/3) := by
  sorry

end log_equation_solution_l3174_317450


namespace sequence_formula_l3174_317436

theorem sequence_formula (n : ℕ+) (S : ℕ+ → ℝ) (a : ℕ+ → ℝ) 
  (h : ∀ k : ℕ+, S k = a k - 3) : 
  a n = 2 * 3^(n : ℝ) := by
sorry

end sequence_formula_l3174_317436


namespace cubic_root_product_l3174_317434

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if the product of any two roots equals 3, then c = 3a -/
theorem cubic_root_product (a b c d : ℝ) (ha : a ≠ 0) :
  (∃ r s t : ℝ, r * s = 3 ∧ r * t = 3 ∧ s * t = 3 ∧
    a * r^3 + b * r^2 + c * r + d = 0 ∧
    a * s^3 + b * s^2 + c * s + d = 0 ∧
    a * t^3 + b * t^2 + c * t + d = 0) →
  c = 3 * a :=
by sorry

end cubic_root_product_l3174_317434


namespace soaps_in_package_l3174_317453

/-- Given a number of boxes, packages per box, and total soaps, calculates soaps per package -/
def soaps_per_package (num_boxes : ℕ) (packages_per_box : ℕ) (total_soaps : ℕ) : ℕ :=
  total_soaps / (num_boxes * packages_per_box)

/-- Theorem: There are 192 soaps in one package -/
theorem soaps_in_package :
  soaps_per_package 2 6 2304 = 192 := by sorry

end soaps_in_package_l3174_317453


namespace polynomial_multiplication_simplification_l3174_317401

theorem polynomial_multiplication_simplification :
  ∀ (x : ℝ),
  (3 * x - 2) * (5 * x^12 + 3 * x^11 - 4 * x^9 + x^8) =
  15 * x^13 - x^12 - 6 * x^11 - 12 * x^10 + 11 * x^9 - 2 * x^8 :=
by sorry

end polynomial_multiplication_simplification_l3174_317401


namespace smallest_sum_of_coefficients_l3174_317413

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 8*x^2 + a*x - b = 0 ∧
    y^3 - 8*y^2 + a*y - b = 0 ∧
    z^3 - 8*z^2 + a*z - b = 0) →
  (∀ a' b' : ℝ, (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 8*x^2 + a'*x - b' = 0 ∧
    y^3 - 8*y^2 + a'*y - b' = 0 ∧
    z^3 - 8*z^2 + a'*z - b' = 0) →
  a + b ≤ a' + b') →
  a + b = 27 :=
by sorry

end smallest_sum_of_coefficients_l3174_317413


namespace carnation_tulip_difference_l3174_317458

theorem carnation_tulip_difference :
  let carnations : ℕ := 13
  let tulips : ℕ := 7
  carnations - tulips = 6 :=
by sorry

end carnation_tulip_difference_l3174_317458


namespace factorization_of_2x_squared_minus_8_l3174_317445

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l3174_317445


namespace count_distinct_n_values_l3174_317498

/-- Given a quadratic equation x² - nx + 36 = 0 with integer roots,
    there are exactly 10 distinct possible values for n. -/
theorem count_distinct_n_values : ∃ (S : Finset ℤ),
  (∀ n ∈ S, ∃ x₁ x₂ : ℤ, x₁ * x₂ = 36 ∧ x₁ + x₂ = n) ∧
  (∀ n : ℤ, (∃ x₁ x₂ : ℤ, x₁ * x₂ = 36 ∧ x₁ + x₂ = n) → n ∈ S) ∧
  Finset.card S = 10 :=
sorry

end count_distinct_n_values_l3174_317498


namespace wendy_score_l3174_317418

/-- Wendy's video game scoring system -/
structure GameScore where
  points_per_treasure : ℕ
  treasures_level1 : ℕ
  treasures_level2 : ℕ

/-- Calculate the total score for Wendy's game -/
def total_score (game : GameScore) : ℕ :=
  (game.treasures_level1 + game.treasures_level2) * game.points_per_treasure

/-- Theorem: Wendy's total score is 35 points -/
theorem wendy_score : 
  ∀ (game : GameScore), 
  game.points_per_treasure = 5 → 
  game.treasures_level1 = 4 → 
  game.treasures_level2 = 3 → 
  total_score game = 35 := by
  sorry

end wendy_score_l3174_317418
