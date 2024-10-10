import Mathlib

namespace prime_divides_power_plus_one_l3616_361664

theorem prime_divides_power_plus_one (n b p : ℕ) :
  n ≠ 0 →
  b ≠ 0 →
  Nat.Prime p →
  Odd p →
  p ∣ b^(2^n) + 1 →
  ∃ m : ℕ, p = 2^(n+1) * m + 1 := by
  sorry

end prime_divides_power_plus_one_l3616_361664


namespace cone_base_radius_l3616_361617

/-- Given a sector paper with radius 24 cm and area 120π cm², 
    prove that the radius of the circular base of the cone formed by this sector is 5 cm -/
theorem cone_base_radius (sector_radius : ℝ) (sector_area : ℝ) (base_radius : ℝ) : 
  sector_radius = 24 →
  sector_area = 120 * Real.pi →
  sector_area = Real.pi * base_radius * sector_radius →
  base_radius = 5 := by
sorry

end cone_base_radius_l3616_361617


namespace hotdogs_served_during_dinner_l3616_361690

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := 11

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := total_hotdogs - lunch_hotdogs

theorem hotdogs_served_during_dinner : dinner_hotdogs = 2 := by
  sorry

end hotdogs_served_during_dinner_l3616_361690


namespace triangle_angles_from_exterior_ratio_l3616_361610

/-- Proves that a triangle with exterior angles in the ratio 12:13:15 has interior angles of 45°, 63°, and 72° -/
theorem triangle_angles_from_exterior_ratio :
  ∀ (E₁ E₂ E₃ : ℝ),
  E₁ > 0 ∧ E₂ > 0 ∧ E₃ > 0 →
  E₁ / 12 = E₂ / 13 ∧ E₂ / 13 = E₃ / 15 →
  E₁ + E₂ + E₃ = 360 →
  ∃ (I₁ I₂ I₃ : ℝ),
    I₁ = 180 - E₁ ∧
    I₂ = 180 - E₂ ∧
    I₃ = 180 - E₃ ∧
    I₁ + I₂ + I₃ = 180 ∧
    I₁ = 45 ∧ I₂ = 63 ∧ I₃ = 72 :=
by sorry


end triangle_angles_from_exterior_ratio_l3616_361610


namespace trey_decorations_l3616_361640

theorem trey_decorations (total : ℕ) (nails thumbtacks sticky : ℕ) : 
  (nails = (2 * total) / 3) →
  (thumbtacks = (2 * (total - nails)) / 5) →
  (sticky = total - nails - thumbtacks) →
  (sticky = 15) →
  (nails = 50) := by
  sorry

end trey_decorations_l3616_361640


namespace power_of_two_representation_l3616_361626

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℕ), 2^n = 7*x^2 + y^2 ∧ Odd x ∧ Odd y :=
sorry

end power_of_two_representation_l3616_361626


namespace ant_return_probability_2006_l3616_361667

/-- A regular octahedron --/
structure Octahedron :=
  (vertices : Finset (Fin 6))
  (edges : Finset (Fin 6 × Fin 6))
  (is_regular : True)  -- This is a simplification; we're assuming it's regular

/-- An ant's position on the octahedron --/
structure AntPosition (O : Octahedron) :=
  (vertex : Fin 6)

/-- The probability distribution of the ant's position after n moves --/
def probability_distribution (O : Octahedron) (n : ℕ) : AntPosition O → ℝ := sorry

/-- The probability of the ant returning to the starting vertex after n moves --/
def return_probability (O : Octahedron) (n : ℕ) : ℝ := sorry

/-- The main theorem --/
theorem ant_return_probability_2006 (O : Octahedron) :
  return_probability O 2006 = (2^2005 + 1) / (3 * 2^2006) := by sorry

end ant_return_probability_2006_l3616_361667


namespace sinusoid_amplitude_l3616_361686

theorem sinusoid_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  a = 4 := by
sorry

end sinusoid_amplitude_l3616_361686


namespace sibling_pizza_order_l3616_361678

-- Define the siblings
inductive Sibling
| Alex
| Beth
| Cyril
| Dan
| Emma

-- Define the function that returns the fraction of pizza eaten by each sibling
def pizza_fraction (s : Sibling) : ℚ :=
  match s with
  | Sibling.Alex => 1/6
  | Sibling.Beth => 1/4
  | Sibling.Cyril => 1/5
  | Sibling.Dan => 1/3
  | Sibling.Emma => 1 - (1/6 + 1/4 + 1/5 + 1/3)

-- Define the order of siblings
def sibling_order : List Sibling :=
  [Sibling.Dan, Sibling.Beth, Sibling.Cyril, Sibling.Alex, Sibling.Emma]

-- Theorem statement
theorem sibling_pizza_order : 
  List.Pairwise (λ a b => pizza_fraction a > pizza_fraction b) sibling_order :=
sorry

end sibling_pizza_order_l3616_361678


namespace sum_of_reciprocals_of_s_max_min_l3616_361643

theorem sum_of_reciprocals_of_s_max_min (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : 
  let s := x^2 + y^2
  ∃ (s_max s_min : ℝ), (∀ (x' y' : ℝ), 4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5 → x'^2 + y'^2 ≤ s_max) ∧
                       (∀ (x' y' : ℝ), 4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5 → s_min ≤ x'^2 + y'^2) ∧
                       (1 / s_max + 1 / s_min = 8 / 5) :=
by sorry

end sum_of_reciprocals_of_s_max_min_l3616_361643


namespace lines_are_parallel_l3616_361650

/-- Two lines are parallel if they have the same slope but different y-intercepts -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * c₂ ≠ a₂ * c₁

/-- The given line: x - 2y + 1 = 0 -/
def line1 : ℝ → ℝ → ℝ := fun x y ↦ x - 2 * y + 1

/-- The line to be proved parallel: 2x - 4y + 1 = 0 -/
def line2 : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - 4 * y + 1

theorem lines_are_parallel : parallel 1 (-2) 1 2 (-4) 1 := by
  sorry

#check lines_are_parallel

end lines_are_parallel_l3616_361650


namespace consecutive_product_not_perfect_power_l3616_361623

theorem consecutive_product_not_perfect_power (a : ℤ) :
  ¬ ∃ (n : ℕ) (k : ℤ), n > 1 ∧ a * (a^2 - 1) = k^n := by
  sorry

end consecutive_product_not_perfect_power_l3616_361623


namespace log_sum_equality_l3616_361646

theorem log_sum_equality : 
  Real.sqrt (Real.log 8 / Real.log 4 + Real.log 10 / Real.log 5) = 
  Real.sqrt (5 / 2 + Real.log 2 / Real.log 5) := by
sorry

end log_sum_equality_l3616_361646


namespace firewood_sacks_filled_l3616_361694

theorem firewood_sacks_filled (sack_capacity : ℕ) (father_wood : ℕ) (ranger_wood : ℕ) (worker_wood : ℕ) (num_workers : ℕ) :
  sack_capacity = 20 →
  father_wood = 80 →
  ranger_wood = 80 →
  worker_wood = 120 →
  num_workers = 2 →
  (father_wood + ranger_wood + num_workers * worker_wood) / sack_capacity = 20 :=
by sorry

end firewood_sacks_filled_l3616_361694


namespace perimeter_is_158_l3616_361654

/-- Represents a tile in the figure -/
structure Tile where
  width : ℕ := 2
  height : ℕ := 4

/-- Represents the figure composed of tiles -/
structure Figure where
  tiles : List Tile
  horizontalEdges : ℕ := 45
  verticalEdges : ℕ := 34

/-- Calculates the perimeter of the figure -/
def calculatePerimeter (f : Figure) : ℕ :=
  (f.horizontalEdges + f.verticalEdges) * 2

/-- Theorem stating that the perimeter of the specific figure is 158 -/
theorem perimeter_is_158 (f : Figure) : calculatePerimeter f = 158 := by
  sorry

#check perimeter_is_158

end perimeter_is_158_l3616_361654


namespace square_area_from_perimeter_l3616_361636

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 32 → area = (perimeter / 4) ^ 2 → area = 64 := by
  sorry

end square_area_from_perimeter_l3616_361636


namespace equation_equivalence_l3616_361603

theorem equation_equivalence (x z : ℝ) 
  (h1 : 3 * x^2 + 4 * x + 6 * z + 2 = 0)
  (h2 : x - 2 * z + 1 = 0) :
  12 * z^2 + 2 * z + 1 = 0 := by
  sorry

end equation_equivalence_l3616_361603


namespace parallel_transitive_l3616_361660

-- Define a type for lines in a plane
variable {Line : Type}

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_transitive (A B C : Line) :
  parallel A C → parallel B C → parallel A B :=
by
  sorry

end parallel_transitive_l3616_361660


namespace no_consecutive_integers_sum_75_l3616_361645

theorem no_consecutive_integers_sum_75 : 
  ¬∃ (a n : ℕ), n ≥ 2 ∧ (n * (2 * a + n - 1) / 2 = 75) := by
  sorry

end no_consecutive_integers_sum_75_l3616_361645


namespace dodecahedron_triangle_count_l3616_361600

/-- The number of vertices in a regular dodecahedron -/
def dodecahedron_vertices : ℕ := 12

/-- The number of distinct triangles that can be formed by connecting three
    different vertices of a regular dodecahedron -/
def dodecahedron_triangles : ℕ := Nat.choose dodecahedron_vertices 3

theorem dodecahedron_triangle_count :
  dodecahedron_triangles = 220 := by
  sorry

end dodecahedron_triangle_count_l3616_361600


namespace inequality_and_equality_condition_l3616_361616

theorem inequality_and_equality_condition (a b c : ℝ) : 
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ Real.sqrt (3*a^2 + (a + b + c)^2) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) = Real.sqrt (3*a^2 + (a + b + c)^2) ↔ 
    (b = c ∨ a = 0) ∧ b*c ≥ 0) :=
by sorry

end inequality_and_equality_condition_l3616_361616


namespace fraction_1800_1809_l3616_361613

/-- The number of states that joined the union during 1800-1809. -/
def states_1800_1809 : ℕ := 4

/-- The total number of states in Walter's collection. -/
def total_states : ℕ := 30

/-- The fraction of states that joined during 1800-1809 out of the first 30 states. -/
theorem fraction_1800_1809 : (states_1800_1809 : ℚ) / total_states = 2 / 15 := by
  sorry

end fraction_1800_1809_l3616_361613


namespace min_red_balls_l3616_361607

/-- The total number of balls in the circle -/
def total_balls : ℕ := 58

/-- A type representing the color of a ball -/
inductive Color
| Red
| Blue

/-- A function that counts the number of consecutive triplets with a majority of a given color -/
def count_majority_triplets (balls : List Color) (color : Color) : ℕ := sorry

/-- A function that counts the total number of balls of a given color -/
def count_color (balls : List Color) (color : Color) : ℕ := sorry

/-- The main theorem stating the minimum number of red balls -/
theorem min_red_balls (balls : List Color) :
  balls.length = total_balls →
  count_majority_triplets balls Color.Red = count_majority_triplets balls Color.Blue →
  count_color balls Color.Red ≥ 20 := by sorry

end min_red_balls_l3616_361607


namespace diophantine_equation_solutions_l3616_361680

theorem diophantine_equation_solutions (x y z : ℤ) :
  (x * y / z : ℚ) + (y * z / x : ℚ) + (z * x / y : ℚ) = 3 ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) := by
  sorry

#check diophantine_equation_solutions

end diophantine_equation_solutions_l3616_361680


namespace percentage_addition_l3616_361611

theorem percentage_addition (x : ℝ) : x * 30 / 100 + 15 * 50 / 100 = 10.5 → x = 10 := by
  sorry

end percentage_addition_l3616_361611


namespace triangle_inequality_from_condition_l3616_361637

theorem triangle_inequality_from_condition 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : ∀ (A B C : ℝ), A > 0 → B > 0 → C > 0 → 
    A * a * (B * b + C * c) + B * b * (C * c + A * a) + C * c * (A * a + B * b) > 
    (1/2) * (A * B * c^2 + B * C * a^2 + C * A * b^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end triangle_inequality_from_condition_l3616_361637


namespace simplify_complex_root_expression_l3616_361618

theorem simplify_complex_root_expression (x : ℝ) (h : x ≥ 0) :
  (4 * x * (11 + 4 * Real.sqrt 6)) ^ (1/6) *
  (4 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) ^ (1/3) =
  (20 * x) ^ (1/3) := by
  sorry

end simplify_complex_root_expression_l3616_361618


namespace ticket_queue_arrangements_l3616_361692

/-- Represents the number of valid arrangements for a ticket queue --/
def validArrangements (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (Nat.factorial n * Nat.factorial (n + 1))

/-- Theorem stating the number of valid arrangements for a ticket queue --/
theorem ticket_queue_arrangements (n : ℕ) :
  validArrangements n = 
    let total_people := 2 * n
    let people_with_five_yuan := n
    let people_with_ten_yuan := n
    let ticket_price := 5
    -- The actual number of valid arrangements
    Nat.factorial total_people / (Nat.factorial people_with_five_yuan * Nat.factorial (people_with_ten_yuan + 1)) :=
by sorry

#check ticket_queue_arrangements

end ticket_queue_arrangements_l3616_361692


namespace parabola_focus_coordinates_l3616_361647

/-- A parabola in the cartesian coordinate plane with equation y^2 = -16x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  h : equation = fun x y ↦ y^2 = -16*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem: The coordinates of the focus of the parabola y^2 = -16x are (-4, 0) -/
theorem parabola_focus_coordinates (p : Parabola) : focus p = (-4, 0) := by sorry

end parabola_focus_coordinates_l3616_361647


namespace cube_root_64_square_root_4_power_l3616_361638

theorem cube_root_64_square_root_4_power (x y : ℝ) : 
  x^3 = 64 → y^2 = 4 → x^y = 16 := by
sorry

end cube_root_64_square_root_4_power_l3616_361638


namespace sum_of_odd_powers_divisible_by_61_l3616_361601

theorem sum_of_odd_powers_divisible_by_61 
  (a₁ a₂ a₃ a₄ : ℤ) 
  (h : a₁^3 + a₂^3 + a₃^3 + a₄^3 = 0) :
  ∀ k : ℕ, k % 2 = 1 → k > 0 → 
  (61 : ℤ) ∣ (a₁^k + a₂^k + a₃^k + a₄^k) := by
sorry

end sum_of_odd_powers_divisible_by_61_l3616_361601


namespace f_even_and_increasing_l3616_361683

def f (x : ℝ) := x^2

theorem f_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ a b, 0 ≤ a → a < b → f a ≤ f b) :=
by sorry

end f_even_and_increasing_l3616_361683


namespace product_inequality_l3616_361632

theorem product_inequality (a b c d : ℝ) : a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d := by
  sorry

end product_inequality_l3616_361632


namespace f_parity_l3616_361668

-- Define the function f(x) = x|x| + px^2
def f (p : ℝ) (x : ℝ) : ℝ := x * abs x + p * x^2

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating the parity of the function depends on p
theorem f_parity (p : ℝ) :
  (p = 0 → is_odd (f p)) ∧
  (p ≠ 0 → ¬(is_even (f p)) ∧ ¬(is_odd (f p))) :=
sorry

end f_parity_l3616_361668


namespace expand_expression_l3616_361604

theorem expand_expression (x : ℝ) : (11 * x + 17) * (3 * x) + 5 = 33 * x^2 + 51 * x + 5 := by
  sorry

end expand_expression_l3616_361604


namespace inverse_f_at_150_l3616_361695

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^4 + 6

-- State the theorem
theorem inverse_f_at_150 :
  ∃ (y : ℝ), f y = 150 ∧ y = (48 : ℝ)^(1/4) :=
sorry

end inverse_f_at_150_l3616_361695


namespace connor_date_cost_l3616_361657

/-- The cost of a movie date for Connor and his date -/
def movie_date_cost (ticket_price : ℚ) (combo_meal_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_meal_price + 2 * candy_price

/-- Theorem stating the total cost of Connor's movie date -/
theorem connor_date_cost :
  movie_date_cost 10 11 2.5 = 36 :=
by sorry

end connor_date_cost_l3616_361657


namespace problem_statement_l3616_361602

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end problem_statement_l3616_361602


namespace equation_solution_l3616_361689

theorem equation_solution (x : ℝ) : 
  (x = (((-1 + Real.sqrt 21) / 2) ^ 3) ∨ x = (((-1 - Real.sqrt 21) / 2) ^ 3)) →
  6 * x^(1/3) - 3 * (x / x^(2/3)) + 2 * x^(2/3) = 10 + x^(1/3) := by
  sorry

end equation_solution_l3616_361689


namespace book_sale_loss_percentage_l3616_361656

theorem book_sale_loss_percentage 
  (selling_price_loss : ℝ) 
  (selling_price_gain : ℝ) 
  (gain_percentage : ℝ) :
  selling_price_loss = 800 →
  selling_price_gain = 1100 →
  gain_percentage = 10 →
  (1 - selling_price_loss / (selling_price_gain / (1 + gain_percentage / 100))) * 100 = 20 := by
  sorry

end book_sale_loss_percentage_l3616_361656


namespace max_profit_at_45_l3616_361655

/-- Represents the daily profit function for selling children's toys -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 1000 * x - 21000

/-- Represents the cost price of each toy -/
def cost_price : ℝ := 30

/-- Represents the maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.5

/-- Represents the minimum selling price -/
def min_selling_price : ℝ := 35

/-- Represents the maximum selling price based on the profit margin constraint -/
def max_selling_price : ℝ := cost_price * (1 + max_profit_margin)

theorem max_profit_at_45 :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, min_selling_price ≤ x ∧ x ≤ max_selling_price →
      profit_function x ≤ profit_function max_selling_price) ∧
    profit_function max_selling_price = max_profit ∧
    max_profit = 3750 := by
  sorry

#eval profit_function max_selling_price

end max_profit_at_45_l3616_361655


namespace union_of_A_and_B_l3616_361652

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 3) / Real.log 2}
def B : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by sorry

end union_of_A_and_B_l3616_361652


namespace inverse_sum_theorem_l3616_361619

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the given condition
axiom condition : ∀ x, f (x + 1) + f (-x - 3) = 2

-- State the theorem to be proved
theorem inverse_sum_theorem : 
  ∀ x, f_inv (2009 - x) + f_inv (x - 2007) = -2 := by sorry

end inverse_sum_theorem_l3616_361619


namespace quadratic_inequality_condition_l3616_361699

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + 1 > 0) → (-2 ≤ m ∧ m ≤ 2) ∧
  ¬((-2 ≤ m ∧ m ≤ 2) → (∀ x : ℝ, x^2 - m*x + 1 > 0)) :=
by sorry

end quadratic_inequality_condition_l3616_361699


namespace regular_poly15_distance_sum_l3616_361639

/-- Regular 15-sided polygon -/
structure RegularPoly15 where
  vertices : Fin 15 → ℝ × ℝ
  is_regular : ∀ i j : Fin 15, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- Distance between two vertices -/
def dist_vertices (p : RegularPoly15) (i j : Fin 15) : ℝ :=
  dist (p.vertices i) (p.vertices j)

/-- Theorem statement -/
theorem regular_poly15_distance_sum (p : RegularPoly15) :
  1 / dist_vertices p 0 2 + 1 / dist_vertices p 0 4 + 1 / dist_vertices p 0 7 =
  1 / dist_vertices p 0 1 := by
  sorry

end regular_poly15_distance_sum_l3616_361639


namespace no_integer_solution_for_divisibility_l3616_361620

theorem no_integer_solution_for_divisibility : ¬∃ (x y : ℤ), (x^2 + y^2 + x + y) ∣ 3 := by
  sorry

end no_integer_solution_for_divisibility_l3616_361620


namespace f_equals_g_l3616_361648

-- Define the two functions
def f (x : ℝ) : ℝ := x - 1
def g (t : ℝ) : ℝ := t - 1

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end f_equals_g_l3616_361648


namespace functions_strictly_greater_iff_no_leq_l3616_361674

-- Define the functions f and g with domain ℝ
variable (f g : ℝ → ℝ)

-- State the theorem
theorem functions_strictly_greater_iff_no_leq :
  (∀ x : ℝ, f x > g x) ↔ ¬∃ x : ℝ, f x ≤ g x := by sorry

end functions_strictly_greater_iff_no_leq_l3616_361674


namespace only_one_and_four_are_propositions_l3616_361672

-- Define a type for the statements
inductive Statement
  | EmptySetSubset
  | GreaterThanImplication
  | IsThreeGreaterThanOne
  | NonIntersectingLinesParallel

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Prop :=
  match s with
  | Statement.EmptySetSubset => True
  | Statement.GreaterThanImplication => False
  | Statement.IsThreeGreaterThanOne => False
  | Statement.NonIntersectingLinesParallel => True

-- Theorem stating that only statements ① and ④ are propositions
theorem only_one_and_four_are_propositions :
  (∀ s : Statement, isProposition s ↔ (s = Statement.EmptySetSubset ∨ s = Statement.NonIntersectingLinesParallel)) :=
by sorry

end only_one_and_four_are_propositions_l3616_361672


namespace sallys_out_of_pocket_cost_l3616_361687

/-- The amount of money Sally needs to pay out of pocket to buy a reading book for each student -/
theorem sallys_out_of_pocket_cost 
  (budget : ℕ) 
  (book_cost : ℕ) 
  (num_students : ℕ) 
  (h1 : budget = 320)
  (h2 : book_cost = 12)
  (h3 : num_students = 30) :
  (book_cost * num_students - budget : ℕ) = 40 := by
  sorry

#check sallys_out_of_pocket_cost

end sallys_out_of_pocket_cost_l3616_361687


namespace smallest_value_abcd_l3616_361681

theorem smallest_value_abcd (a b c d : ℤ) 
  (sum_condition : a + b + c + d < 25)
  (a_condition : a > 8)
  (b_condition : b < 5)
  (c_odd : c % 2 = 1)
  (d_even : d % 2 = 0) :
  (∀ a' b' c' d' : ℤ, 
    a' + b' + c' + d' < 25 → 
    a' > 8 → 
    b' < 5 → 
    c' % 2 = 1 → 
    d' % 2 = 0 → 
    a' - b' + c' - d' ≥ a - b + c - d) →
  a - b + c - d = -4 :=
sorry

end smallest_value_abcd_l3616_361681


namespace matching_shoe_probability_l3616_361693

/-- The probability of selecting a matching pair of shoes from a box containing 6 pairs -/
theorem matching_shoe_probability (total_shoes : ℕ) (total_pairs : ℕ) (h1 : total_shoes = 12) (h2 : total_pairs = 6) :
  (total_pairs : ℚ) / ((total_shoes.choose 2) : ℚ) = 1 / 11 := by
  sorry

#check matching_shoe_probability

end matching_shoe_probability_l3616_361693


namespace largest_m_satisfying_inequality_l3616_361624

theorem largest_m_satisfying_inequality :
  ∃ m : ℕ, (((1 : ℚ) / 4 + m / 9 < 5 / 2) ∧
            ∀ n : ℕ, (n > m → (1 : ℚ) / 4 + n / 9 ≥ 5 / 2)) ∧
            m = 10 := by
  sorry

end largest_m_satisfying_inequality_l3616_361624


namespace max_sides_touched_l3616_361641

/-- Represents a regular hexagon -/
structure RegularHexagon where
  -- Add any necessary fields

/-- Represents a circle -/
structure Circle where
  -- Add any necessary fields

/-- Predicate to check if a circle is entirely contained within a hexagon -/
def is_contained (c : Circle) (h : RegularHexagon) : Prop :=
  sorry

/-- Predicate to check if a circle touches a side of a hexagon -/
def touches_side (c : Circle) (h : RegularHexagon) (side : Nat) : Prop :=
  sorry

/-- Predicate to check if a circle touches all sides of a hexagon -/
def touches_all_sides (c : Circle) (h : RegularHexagon) : Prop :=
  sorry

/-- The main theorem -/
theorem max_sides_touched (h : RegularHexagon) :
  ∃ (c : Circle), is_contained c h ∧ ¬touches_all_sides c h ∧
  (∃ (n : Nat), n = 2 ∧ 
    (∀ (m : Nat), (∃ (sides : Finset Nat), sides.card = m ∧ 
      (∀ (side : Nat), side ∈ sides → touches_side c h side)) → m ≤ n)) :=
sorry

end max_sides_touched_l3616_361641


namespace absolute_value_32_l3616_361612

theorem absolute_value_32 (x : ℝ) : |x| = 32 → x = 32 ∨ x = -32 := by
  sorry

end absolute_value_32_l3616_361612


namespace line_through_P_with_equal_intercepts_l3616_361644

/-- A line in the 2D plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The point P(2,3) -/
def P : ℝ × ℝ := (2, 3)

/-- A line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- A line has equal intercepts on both coordinate axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The equation of the line is y = 3/2 * x -/
def is_y_eq_3_2x (l : Line) : Prop :=
  l.a = 2 ∧ l.b = -3 ∧ l.c = 0

/-- The equation of the line is x + y - 5 = 0 -/
def is_x_plus_y_eq_5 (l : Line) : Prop :=
  l.a = 1 ∧ l.b = 1 ∧ l.c = -5

theorem line_through_P_with_equal_intercepts (l : Line) :
  passes_through l P ∧ has_equal_intercepts l →
  is_y_eq_3_2x l ∨ is_x_plus_y_eq_5 l := by
  sorry

end line_through_P_with_equal_intercepts_l3616_361644


namespace square_area_probability_l3616_361688

/-- The probability of a randomly chosen point on a line segment of length 12
    forming a square with an area between 36 and 81 -/
theorem square_area_probability : ∀ (AB : ℝ) (lower upper : ℝ),
  AB = 12 →
  lower = 6 →
  upper = 9 →
  (upper - lower) / AB = 1 / 4 :=
by sorry

end square_area_probability_l3616_361688


namespace binomial_coefficient_equality_l3616_361684

theorem binomial_coefficient_equality (a : ℝ) (ha : a ≠ 0) :
  (Nat.choose 5 4 : ℝ) * a^4 = (Nat.choose 5 3 : ℝ) * a^3 → a = 2 := by
  sorry

end binomial_coefficient_equality_l3616_361684


namespace natasha_dimes_l3616_361621

theorem natasha_dimes (n : ℕ) 
  (h1 : 100 < n ∧ n < 200)
  (h2 : n % 6 = 2)
  (h3 : n % 7 = 2)
  (h4 : n % 8 = 2) : 
  n = 170 := by sorry

end natasha_dimes_l3616_361621


namespace no_solution_triple_inequality_l3616_361679

theorem no_solution_triple_inequality :
  ¬ ∃ (x y z : ℝ), (|x| < |y - z| ∧ |y| < |z - x| ∧ |z| < |x - y|) :=
by sorry

end no_solution_triple_inequality_l3616_361679


namespace jogging_challenge_l3616_361606

theorem jogging_challenge (monday_distance : Real) (daily_increase : Real) 
  (saturday_multiplier : Real) (weekly_goal : Real) :
  let tuesday_distance := monday_distance * (1 + daily_increase)
  let thursday_distance := tuesday_distance * (1 + daily_increase)
  let saturday_distance := thursday_distance * saturday_multiplier
  let sunday_distance := weekly_goal - (monday_distance + tuesday_distance + thursday_distance + saturday_distance)
  monday_distance = 3 ∧ 
  daily_increase = 0.1 ∧ 
  saturday_multiplier = 2.5 ∧ 
  weekly_goal = 40 →
  tuesday_distance = 3.3 ∧ 
  thursday_distance = 3.63 ∧ 
  saturday_distance = 9.075 ∧ 
  sunday_distance = 21.995 := by
  sorry

end jogging_challenge_l3616_361606


namespace seashells_remaining_l3616_361605

theorem seashells_remaining (initial : ℕ) (given_joan : ℕ) (given_ali : ℕ) (given_lee : ℕ) :
  initial = 200 →
  given_joan = 43 →
  given_ali = 27 →
  given_lee = 59 →
  initial - given_joan - given_ali - given_lee = 71 :=
by sorry

end seashells_remaining_l3616_361605


namespace smallest_prime_20_less_than_square_l3616_361691

theorem smallest_prime_20_less_than_square : 
  ∃ (n : ℕ), 
    5 = n^2 - 20 ∧ 
    Prime 5 ∧ 
    (∀ (m : ℕ) (p : ℕ), p < 5 → p = m^2 - 20 → ¬ Prime p) :=
by sorry

end smallest_prime_20_less_than_square_l3616_361691


namespace circle_center_proof_l3616_361670

/-- The equation of a circle in polar coordinates -/
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The center of a circle in polar coordinates -/
def circle_center : ℝ × ℝ := (1, 0)

/-- Theorem stating that the center of the circle ρ = 2cosθ is at (1, 0) in polar coordinates -/
theorem circle_center_proof :
  ∀ ρ θ : ℝ, polar_circle_equation ρ θ → circle_center = (1, 0) :=
by
  sorry


end circle_center_proof_l3616_361670


namespace expression_evaluation_l3616_361673

theorem expression_evaluation (x y : ℝ) 
  (h : (3*x + 1)^2 + |y - 3| = 0) : 
  (x + 2*y) * (x - 2*y) + (x + 2*y)^2 - x * (2*x + 3*y) = -1 := by
  sorry

end expression_evaluation_l3616_361673


namespace smallest_upper_bound_sum_reciprocals_l3616_361608

theorem smallest_upper_bound_sum_reciprocals :
  ∃ (r s : ℕ), r ≠ 0 ∧ s ≠ 0 ∧
  (∀ (k m n : ℕ), k ≠ 0 → m ≠ 0 → n ≠ 0 →
    (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n < 1 →
    (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n ≤ r / s) ∧
  (∀ (p q : ℕ), p ≠ 0 → q ≠ 0 →
    (∀ (k m n : ℕ), k ≠ 0 → m ≠ 0 → n ≠ 0 →
      (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n < 1 →
      (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n ≤ p / q) →
    r / s ≤ p / q) ∧
  r / s = 41 / 42 := by
sorry

end smallest_upper_bound_sum_reciprocals_l3616_361608


namespace disease_cases_estimation_l3616_361661

/-- A function representing the number of disease cases over time -/
def cases (t : ℝ) : ℝ := 800000 - 19995 * (t - 1970)

theorem disease_cases_estimation :
  cases 1995 = 300125 ∧ cases 2005 = 100175 :=
by
  sorry

end disease_cases_estimation_l3616_361661


namespace correct_league_members_l3616_361635

/-- The number of members in the Valleyball Soccer League --/
def league_members : ℕ := 110

/-- The cost of a pair of socks in dollars --/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars --/
def tshirt_additional_cost : ℕ := 8

/-- The total expenditure of the league in dollars --/
def total_expenditure : ℕ := 3740

/-- Theorem stating that the number of members in the league is correct given the conditions --/
theorem correct_league_members :
  let tshirt_cost : ℕ := sock_cost + tshirt_additional_cost
  let member_cost : ℕ := sock_cost + 2 * tshirt_cost
  total_expenditure = league_members * member_cost :=
by sorry

end correct_league_members_l3616_361635


namespace prime_squares_congruence_l3616_361659

theorem prime_squares_congruence (p : ℕ) (hp : Prime p) :
  (∀ a : ℕ, ¬(p ∣ a) → a^2 % p = 1) → p = 2 ∨ p = 3 := by
  sorry

end prime_squares_congruence_l3616_361659


namespace arithmetic_geometric_ratio_l3616_361669

/-- An arithmetic sequence with a common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_nonconstant : d ≠ 0)
  (h_geom : ∃ r, 
    arithmetic_term (a 1) d 10 = r * arithmetic_term (a 1) d 5 ∧
    arithmetic_term (a 1) d 20 = r * arithmetic_term (a 1) d 10) :
  ∃ r, r = 2 ∧
    arithmetic_term (a 1) d 10 = r * arithmetic_term (a 1) d 5 ∧
    arithmetic_term (a 1) d 20 = r * arithmetic_term (a 1) d 10 :=
by sorry

end arithmetic_geometric_ratio_l3616_361669


namespace planks_per_table_is_15_l3616_361649

def planks_per_table (trees : ℕ) (planks_per_tree : ℕ) (table_price : ℕ) (labor_cost : ℕ) (profit : ℕ) : ℕ :=
  let total_planks := trees * planks_per_tree
  let total_revenue := profit + labor_cost
  let num_tables := total_revenue / table_price
  total_planks / num_tables

theorem planks_per_table_is_15 :
  planks_per_table 30 25 300 3000 12000 = 15 := by
  sorry

end planks_per_table_is_15_l3616_361649


namespace condition_necessary_not_sufficient_l3616_361662

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) :=
by sorry

end condition_necessary_not_sufficient_l3616_361662


namespace twin_brothers_age_product_difference_l3616_361609

theorem twin_brothers_age_product_difference :
  ∀ (current_age : ℕ),
  current_age = 4 →
  (current_age + 1) * (current_age + 1) - current_age * current_age = 9 :=
by
  sorry

end twin_brothers_age_product_difference_l3616_361609


namespace log_base_2_derivative_l3616_361682

open Real

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
sorry

end log_base_2_derivative_l3616_361682


namespace projection_vector_proof_l3616_361651

def line_direction : ℝ × ℝ := (3, 2)

theorem projection_vector_proof :
  ∃ (w : ℝ × ℝ), 
    w.1 + w.2 = 3 ∧ 
    w.1 * line_direction.1 + w.2 * line_direction.2 = 0 ∧
    w = (2, 1) := by
  sorry

end projection_vector_proof_l3616_361651


namespace line_translation_down_4_units_l3616_361615

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - units }

theorem line_translation_down_4_units :
  let original_line : Line := { slope := -2, intercept := 3 }
  let translated_line := translateLine original_line 4
  translated_line = { slope := -2, intercept := -1 } := by sorry

end line_translation_down_4_units_l3616_361615


namespace quadratic_inequality_condition_l3616_361627

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ |a| ≤ 2 := by sorry

end quadratic_inequality_condition_l3616_361627


namespace triangle_sin_c_max_l3616_361653

/-- Given a triangle ABC with sides a, b, c, prove that if the dot products of its vectors
    satisfy the given condition, then sin C is less than or equal to √7/3 -/
theorem triangle_sin_c_max (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_dot_product : b * c * (b^2 + c^2 - a^2) + 2 * a * c * (a^2 + c^2 - b^2) = 
                   3 * a * b * (a^2 + b^2 - c^2)) :
  Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ≤ Real.sqrt 7 / 3 := by
  sorry

end triangle_sin_c_max_l3616_361653


namespace walking_time_calculation_l3616_361634

theorem walking_time_calculation (walking_speed run_speed : ℝ) (run_time : ℝ) (h1 : walking_speed = 5) (h2 : run_speed = 15) (h3 : run_time = 36 / 60) :
  let distance := run_speed * run_time
  walking_speed * (distance / walking_speed) = 1.8 := by sorry

end walking_time_calculation_l3616_361634


namespace unique_prime_in_range_l3616_361665

def f (n : ℕ) : ℤ := n^3 - 7*n^2 + 15*n - 12

def is_prime (z : ℤ) : Prop := z > 1 ∧ ∀ m : ℕ, 1 < m → m < |z| → ¬(z % m = 0)

theorem unique_prime_in_range :
  ∃! (n : ℕ), 0 < n ∧ n ≤ 6 ∧ is_prime (f n) :=
sorry

end unique_prime_in_range_l3616_361665


namespace distance_between_points_l3616_361631

/-- The distance between two points (3, 3) and (9, 10) is √85 -/
theorem distance_between_points : Real.sqrt 85 = Real.sqrt ((9 - 3)^2 + (10 - 3)^2) := by
  sorry

end distance_between_points_l3616_361631


namespace negation_equivalence_l3616_361658

-- Define the original proposition
def original_proposition : Prop := ∀ x : ℝ, x > Real.sin x

-- Define the negation of the original proposition
def negation_proposition : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- Theorem stating that the negation of the original proposition is equivalent to the negation_proposition
theorem negation_equivalence : ¬original_proposition ↔ negation_proposition := by sorry

end negation_equivalence_l3616_361658


namespace fraction_problem_l3616_361625

theorem fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → 
  (40/100 : ℝ) * N = 384 := by
sorry

end fraction_problem_l3616_361625


namespace constant_difference_of_equal_second_derivatives_l3616_361663

theorem constant_difference_of_equal_second_derivatives 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, (deriv^[2] f) x = (deriv^[2] g) x) : 
  ∃ c : ℝ, ∀ x, f x - g x = c :=
sorry

end constant_difference_of_equal_second_derivatives_l3616_361663


namespace balance_after_six_months_l3616_361671

/-- Calculates the balance after two quarters of compound interest -/
def balance_after_two_quarters (initial_deposit : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let balance_after_first_quarter := initial_deposit * (1 + rate1)
  balance_after_first_quarter * (1 + rate2)

/-- Theorem stating the balance after two quarters with given initial deposit and interest rates -/
theorem balance_after_six_months 
  (initial_deposit : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (h1 : initial_deposit = 5000)
  (h2 : rate1 = 0.07)
  (h3 : rate2 = 0.085) :
  balance_after_two_quarters initial_deposit rate1 rate2 = 5804.25 := by
  sorry

#eval balance_after_two_quarters 5000 0.07 0.085

end balance_after_six_months_l3616_361671


namespace cost_of_traveling_specific_roads_l3616_361696

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def cost_of_traveling_roads (lawn_length lawn_width road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Theorem stating the cost of traveling two intersecting roads on a specific rectangular lawn. -/
theorem cost_of_traveling_specific_roads : 
  cost_of_traveling_roads 80 50 10 3 = 3600 := by
  sorry

end cost_of_traveling_specific_roads_l3616_361696


namespace inequality_proof_l3616_361685

theorem inequality_proof (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  abs a + abs b + abs c ≥ 3 * Real.sqrt 3 * (b^2 * c^2 + c^2 * a^2 + a^2 * b^2) := by
  sorry

end inequality_proof_l3616_361685


namespace isosceles_triangle_side_lengths_l3616_361630

/-- An isosceles triangle with centroid on the inscribed circle -/
structure IsoscelesTriangleWithCentroidOnIncircle where
  -- The lengths of the sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The triangle is isosceles with sides a and b equal
  isosceles : a = b
  -- The perimeter of the triangle is 60
  perimeter : a + b + c = 60
  -- The centroid lies on the inscribed circle
  centroid_on_incircle : True  -- We represent this condition as always true for simplicity

/-- The theorem stating the side lengths of the triangle -/
theorem isosceles_triangle_side_lengths 
  (t : IsoscelesTriangleWithCentroidOnIncircle) : 
  t.a = 25 ∧ t.b = 25 ∧ t.c = 10 := by
  sorry

end isosceles_triangle_side_lengths_l3616_361630


namespace prob_king_or_queen_is_two_thirteenths_l3616_361633

-- Define the properties of a standard deck
structure StandardDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (h_total : total_cards = 52)
  (h_ranks : num_ranks = 13)
  (h_suits : num_suits = 4)
  (h_kings : num_kings = 4)
  (h_queens : num_queens = 4)
  (h_cards_per_rank : total_cards = num_ranks * num_suits)

-- Define the probability function
def probability_king_or_queen (deck : StandardDeck) : ℚ :=
  (deck.num_kings + deck.num_queens : ℚ) / deck.total_cards

-- State the theorem
theorem prob_king_or_queen_is_two_thirteenths (deck : StandardDeck) :
  probability_king_or_queen deck = 2 / 13 := by
  sorry

end prob_king_or_queen_is_two_thirteenths_l3616_361633


namespace cloth_profit_theorem_l3616_361614

/-- Calculates the profit per meter of cloth given the total meters sold, 
    total selling price, and cost price per meter. -/
def profit_per_meter (meters_sold : ℕ) (total_selling_price : ℚ) (cost_price_per_meter : ℚ) : ℚ :=
  (total_selling_price - (meters_sold : ℚ) * cost_price_per_meter) / (meters_sold : ℚ)

/-- Theorem stating that given 85 meters of cloth sold for $8925 
    with a cost price of $90 per meter, the profit per meter is $15. -/
theorem cloth_profit_theorem :
  profit_per_meter 85 8925 90 = 15 := by
  sorry

end cloth_profit_theorem_l3616_361614


namespace two_digit_perfect_squares_divisible_by_four_l3616_361698

theorem two_digit_perfect_squares_divisible_by_four :
  (∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ 
    (10 ≤ n^2 ∧ n^2 ≤ 99) ∧ 4 ∣ n^2) ∧
  (∃ s, (∀ n, n ∈ s ↔ 
    (10 ≤ n^2 ∧ n^2 ≤ 99) ∧ 4 ∣ n^2) ∧ 
    Finset.card s = 3) :=
by sorry

end two_digit_perfect_squares_divisible_by_four_l3616_361698


namespace tan_105_degrees_l3616_361628

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l3616_361628


namespace second_quadrant_and_modulus_condition_l3616_361677

def complex_i : ℂ := Complex.I

theorem second_quadrant_and_modulus_condition (a : ℝ) : 
  let z₁ : ℂ := a + 2 / (1 - complex_i)
  let z₂ : ℂ := a - complex_i
  (z₁.re < 0 ∧ z₁.im > 0) → Complex.abs z₂ = 2 → a = -Real.sqrt 3 := by
  sorry

end second_quadrant_and_modulus_condition_l3616_361677


namespace parallel_vectors_x_value_l3616_361675

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -9)
  parallel a b → x = -6 :=
by
  sorry

end parallel_vectors_x_value_l3616_361675


namespace intersection_of_A_and_B_l3616_361697

def set_A : Set ℝ := {x | -1 ≤ 2*x+1 ∧ 2*x+1 ≤ 3}
def set_B : Set ℝ := {x | x ≠ 0 ∧ (x-2)/x ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l3616_361697


namespace remaining_area_of_19x11_rectangle_l3616_361622

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of the remaining rectangle after placing the largest possible squares inside a given rectangle -/
def remainingArea (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that the remaining area of a 19x11 rectangle after placing four squares is 6 -/
theorem remaining_area_of_19x11_rectangle : 
  remainingArea ⟨19, 11⟩ = 6 := by sorry

end remaining_area_of_19x11_rectangle_l3616_361622


namespace least_sum_of_exponents_l3616_361642

theorem least_sum_of_exponents (h : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h) 
  (h_div_216 : 216 ∣ h) 
  (h_eq : h = 2^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ)) : 
  (∀ a' b' c' : ℕ+, 
    (225 ∣ (2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ))) → 
    (216 ∣ (2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ))) → 
    (a:ℕ) + (b:ℕ) + (c:ℕ) ≤ (a':ℕ) + (b':ℕ) + (c':ℕ)) ∧ 
  (a:ℕ) + (b:ℕ) + (c:ℕ) = 10 :=
sorry

end least_sum_of_exponents_l3616_361642


namespace investment_problem_l3616_361676

/-- Proves that the amount invested in the first account is approximately $2336.36 --/
theorem investment_problem (total_interest : ℝ) (second_account_investment : ℝ) 
  (interest_rate_difference : ℝ) (first_account_rate : ℝ) :
  total_interest = 1282 →
  second_account_investment = 8200 →
  interest_rate_difference = 0.015 →
  first_account_rate = 0.11 →
  ∃ x : ℝ, (x * first_account_rate + 
    second_account_investment * (first_account_rate + interest_rate_difference) = total_interest) ∧ 
    (abs (x - 2336.36) < 0.01) :=
by sorry

end investment_problem_l3616_361676


namespace cost_price_calculation_cost_price_is_15000_l3616_361629

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem cost_price_is_15000 : 
  cost_price_calculation 18000 0.1 0.08 = 15000 := by
  sorry

end cost_price_calculation_cost_price_is_15000_l3616_361629


namespace parabola_intercepts_sum_l3616_361666

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define the x-intercept
def a : ℝ := parabola 0

-- Define the y-intercepts
def b_and_c : Set ℝ := {y | parabola y = 0}

-- Theorem statement
theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ b_and_c ∧ c ∈ b_and_c ∧ b ≠ c ∧ a + b + c = 7 :=
sorry

end parabola_intercepts_sum_l3616_361666
