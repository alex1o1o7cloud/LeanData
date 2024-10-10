import Mathlib

namespace zero_not_necessarily_in_2_5_l240_24020

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f having its only zero in (1,3)
def has_only_zero_in_open_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 3 ∧ f x = 0 ∧ ∀ y, f y = 0 → (1 < y ∧ y < 3)

-- State the theorem
theorem zero_not_necessarily_in_2_5 
  (h : has_only_zero_in_open_interval f) : 
  ¬(∀ f, has_only_zero_in_open_interval f → ∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end zero_not_necessarily_in_2_5_l240_24020


namespace functional_equation_characterization_l240_24032

theorem functional_equation_characterization
  (a : ℤ) (ha : a ≠ 0)
  (f g : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + g y) = g x + f y + a * y) :
  (∃ n : ℤ, n ≠ 0 ∧ n ≠ 1 ∧ a = n^2 - n) ∧
  (∃ n : ℤ, ∃ v : ℚ, (n ≠ 0 ∧ n ≠ 1) ∧
    ((∀ x : ℚ, f x = n * x + v ∧ g x = n * x) ∨
     (∀ x : ℚ, f x = (1 - n) * x + v ∧ g x = (1 - n) * x))) :=
by sorry

end functional_equation_characterization_l240_24032


namespace angle_with_parallel_sides_l240_24031

-- Define the concept of parallel angles
def parallel_angles (A B : Real) : Prop := sorry

-- Theorem statement
theorem angle_with_parallel_sides (A B : Real) :
  parallel_angles A B → A = 45 → (B = 45 ∨ B = 135) := by
  sorry

end angle_with_parallel_sides_l240_24031


namespace functional_equation_solutions_l240_24058

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, 0 < x → 0 < f x}

/-- The functional equation property -/
def SatisfiesFunctionalEquation (f : PositiveRealFunction) : Prop :=
  ∀ x y, 0 < x → 0 < y → f.val (x^y) = (f.val x)^(f.val y)

/-- The main theorem -/
theorem functional_equation_solutions (f : PositiveRealFunction) 
  (h : SatisfiesFunctionalEquation f) :
  (∀ x, 0 < x → f.val x = 1) ∨ (∀ x, 0 < x → f.val x = x) :=
sorry

end functional_equation_solutions_l240_24058


namespace whale_population_growth_l240_24025

/-- Proves that given the conditions of whale population growth, 
    the initial number of whales was 4000 -/
theorem whale_population_growth (w : ℕ) 
  (h1 : 2 * w = w + w)  -- The number of whales doubles each year
  (h2 : 2 * (2 * w) + 800 = 8800)  -- Prediction for third year
  : w = 4000 := by
  sorry

end whale_population_growth_l240_24025


namespace parking_arrangement_l240_24042

/-- The number of ways to park cars in a row with empty spaces -/
def park_cars (total_spaces : ℕ) (cars : ℕ) (empty_spaces : ℕ) : ℕ :=
  (total_spaces - empty_spaces + 1) * (cars.factorial)

theorem parking_arrangement :
  park_cars 8 4 4 = 120 :=
by sorry

end parking_arrangement_l240_24042


namespace max_rooms_less_than_55_l240_24079

/-- Represents the number of rooms with different combinations of bouquets -/
structure RoomCounts where
  chrysOnly : ℕ
  carnOnly : ℕ
  roseOnly : ℕ
  chrysCarn : ℕ
  chrysRose : ℕ
  carnRose : ℕ
  allThree : ℕ

/-- The conditions of the mansion and its bouquets -/
def MansionConditions (r : RoomCounts) : Prop :=
  r.chrysCarn = 2 ∧
  r.chrysRose = 3 ∧
  r.carnRose = 4 ∧
  r.chrysOnly + r.chrysCarn + r.chrysRose + r.allThree = 10 ∧
  r.carnOnly + r.chrysCarn + r.carnRose + r.allThree = 20 ∧
  r.roseOnly + r.chrysRose + r.carnRose + r.allThree = 30

/-- The total number of rooms in the mansion -/
def totalRooms (r : RoomCounts) : ℕ :=
  r.chrysOnly + r.carnOnly + r.roseOnly + r.chrysCarn + r.chrysRose + r.carnRose + r.allThree

/-- Theorem stating that the maximum number of rooms is less than 55 -/
theorem max_rooms_less_than_55 (r : RoomCounts) (h : MansionConditions r) : 
  totalRooms r < 55 := by
  sorry


end max_rooms_less_than_55_l240_24079


namespace product_of_numbers_l240_24027

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := by
  sorry

end product_of_numbers_l240_24027


namespace right_triangle_circle_theorem_l240_24022

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle with vertices P, Q, and R -/
structure Triangle :=
  (P : Point)
  (Q : Point)
  (R : Point)

/-- Checks if a triangle is right-angled at Q -/
def isRightTriangle (t : Triangle) : Prop :=
  -- Definition of right triangle at Q
  sorry

/-- Checks if a point S lies on the circle with diameter QR -/
def isOnCircle (t : Triangle) (S : Point) : Prop :=
  -- Definition of S being on the circle with diameter QR
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Definition of distance between two points
  sorry

/-- Main theorem -/
theorem right_triangle_circle_theorem (t : Triangle) (S : Point) :
  isRightTriangle t →
  isOnCircle t S →
  distance t.P S = 3 →
  distance t.Q S = 9 →
  distance t.R S = 27 :=
by
  sorry

end right_triangle_circle_theorem_l240_24022


namespace consecutive_ranks_probability_l240_24084

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Number of possible consecutive rank sets (A-2-3 to J-Q-K) -/
def ConsecutiveRankSets : ℕ := 10

/-- Number of suits in a standard deck -/
def Suits : ℕ := 4

/-- The probability of drawing three cards of consecutive ranks from a standard deck -/
theorem consecutive_ranks_probability :
  (ConsecutiveRankSets * Suits^CardsDrawn) / (StandardDeck.choose CardsDrawn) = 32 / 1105 := by
  sorry

#check consecutive_ranks_probability

end consecutive_ranks_probability_l240_24084


namespace parabola_directrix_l240_24015

/-- The directrix of a parabola x^2 + 12y = 0 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, x^2 + 12*y = 0) → (∃ k : ℝ, k = 3 ∧ y = k) :=
by sorry

end parabola_directrix_l240_24015


namespace only_rainbow_statement_correct_l240_24010

/-- Represents the conditions for seeing a rainbow --/
structure RainbowConditions :=
  (sunlight : Bool)
  (rain : Bool)
  (observer_position : ℝ × ℝ × ℝ)

/-- Represents the outcome of a coin flip --/
inductive CoinFlip
  | Heads
  | Tails

/-- Represents the precipitation data for a city --/
structure PrecipitationData :=
  (average : ℝ)
  (variance : ℝ)

/-- The set of statements about random events and statistical concepts --/
inductive Statement
  | RainbowRandomEvent
  | AircraftRandomSampling
  | CoinFlipDeterministic
  | PrecipitationStability

/-- Function to determine if seeing a rainbow is random given the conditions --/
def is_rainbow_random (conditions : RainbowConditions) : Prop :=
  ∃ (c1 c2 : RainbowConditions), c1 ≠ c2 ∧ 
    (c1.sunlight ∧ c1.rain) ∧ (c2.sunlight ∧ c2.rain) ∧ 
    c1.observer_position ≠ c2.observer_position

/-- Function to determine if a statement is correct --/
def is_correct_statement (s : Statement) : Prop :=
  match s with
  | Statement.RainbowRandomEvent => ∀ c, is_rainbow_random c
  | _ => False

/-- Theorem stating that only the rainbow statement is correct --/
theorem only_rainbow_statement_correct :
  ∀ s, is_correct_statement s ↔ s = Statement.RainbowRandomEvent :=
sorry

end only_rainbow_statement_correct_l240_24010


namespace danny_bottle_caps_l240_24011

theorem danny_bottle_caps (initial : ℕ) (found : ℕ) (current : ℕ) (thrown_away : ℕ) : 
  initial = 69 → found = 58 → current = 67 → 
  thrown_away = initial + found - current →
  thrown_away = 60 := by
sorry

end danny_bottle_caps_l240_24011


namespace amaya_total_marks_l240_24028

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  socialStudies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks across all subjects -/
def totalMarks (m : Marks) : ℕ := m.music + m.socialStudies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored given the conditions -/
theorem amaya_total_marks :
  ∀ (m : Marks),
    m.music = 70 →
    m.socialStudies = m.music + 10 →
    m.maths = m.arts - 20 →
    m.maths = (9 : ℕ) * m.arts / 10 →
    totalMarks m = 530 := by
  sorry

#check amaya_total_marks

end amaya_total_marks_l240_24028


namespace system_of_equations_l240_24048

theorem system_of_equations (p t j x y : ℝ) : 
  j = 0.75 * p →
  j = 0.8 * t →
  t = p * (1 - t / 100) →
  x = 0.1 * t →
  y = 0.5 * j →
  x + y = 12 →
  t = 24 := by
sorry

end system_of_equations_l240_24048


namespace both_selected_l240_24040

-- Define the probabilities of selection for Ram and Ravi
def prob_ram : ℚ := 1/7
def prob_ravi : ℚ := 1/5

-- Define the probability of both being selected
def prob_both : ℚ := prob_ram * prob_ravi

-- Theorem: The probability of both Ram and Ravi being selected is 1/35
theorem both_selected : prob_both = 1/35 := by
  sorry

end both_selected_l240_24040


namespace smallest_solution_congruence_l240_24007

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (6 * x) % 31 = 19 % 31 ∧ 
  ∀ (y : ℕ), y > 0 → (6 * y) % 31 = 19 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l240_24007


namespace sons_age_l240_24094

/-- Given a man and his son, where the man is 32 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 30 years. -/
theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 32 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 30 := by
sorry

end sons_age_l240_24094


namespace lucy_money_problem_l240_24076

theorem lucy_money_problem (initial_money : ℚ) : 
  (initial_money * (2/3) * (3/4) = 15) → initial_money = 30 := by
  sorry

end lucy_money_problem_l240_24076


namespace mark_height_in_feet_l240_24066

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Height to total inches -/
def Height.toInches (h : Height) : ℕ := h.feet * 12 + h.inches

/-- The height difference between Mike and Mark in inches -/
def heightDifference : ℕ := 10

/-- Mike's height -/
def mikeHeight : Height := ⟨6, 1, by sorry⟩

/-- Mark's height in inches -/
def markHeightInches : ℕ := mikeHeight.toInches - heightDifference

theorem mark_height_in_feet :
  ∃ (h : Height), h.toInches = markHeightInches ∧ h.feet = 5 ∧ h.inches = 3 := by
  sorry

end mark_height_in_feet_l240_24066


namespace logarithm_expression_equality_l240_24091

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the logarithm with arbitrary base
noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem logarithm_expression_equality :
  (lg 5)^2 + lg 2 * lg 50 - log 8 9 * log 27 32 = -1/9 := by
  sorry

end logarithm_expression_equality_l240_24091


namespace complex_calculation_l240_24069

theorem complex_calculation (c d : ℂ) (hc : c = 3 + 2*I) (hd : d = 2 - 3*I) :
  3*c + 4*d = 17 - 6*I :=
by sorry

end complex_calculation_l240_24069


namespace region_location_l240_24090

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Theorem statement
theorem region_location :
  ∀ (x y : ℝ), region x y → 
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ x < x₀ ∧ y > y₀ :=
sorry

end region_location_l240_24090


namespace root_nature_depends_on_k_l240_24085

theorem root_nature_depends_on_k :
  ∀ k : ℝ, ∃ Δ : ℝ, 
    (Δ = 1 + 4*k) ∧ 
    (Δ < 0 → (∀ x : ℝ, (x - 1) * (x - 2) ≠ k)) ∧
    (Δ = 0 → (∃! x : ℝ, (x - 1) * (x - 2) = k)) ∧
    (Δ > 0 → (∃ x y : ℝ, x ≠ y ∧ (x - 1) * (x - 2) = k ∧ (y - 1) * (y - 2) = k)) :=
by sorry


end root_nature_depends_on_k_l240_24085


namespace quadratic_non_real_roots_l240_24036

/-- A quadratic equation x^2 + bx + 16 has two non-real roots if and only if b is in the open interval (-8, 8) -/
theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
sorry

end quadratic_non_real_roots_l240_24036


namespace remainder_theorem_l240_24030

theorem remainder_theorem (y : ℤ) : 
  ∃ (P : ℤ → ℤ), y^50 = (y^2 - 5*y + 6) * P y + (2^50*(y-3) - 3^50*(y-2)) := by
  sorry

end remainder_theorem_l240_24030


namespace expansion_properties_l240_24052

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x+2)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The coefficient of x^k in the expansion of (x+2)^n -/
def coeff (n k : ℕ) : ℕ := sorry

theorem expansion_properties :
  let n : ℕ := 8
  let a₀ : ℕ := coeff n 0
  let a₁ : ℕ := coeff n 1
  let a₂ : ℕ := coeff n 2
  -- a₀, a₁, a₂ form an arithmetic sequence
  (a₁ - a₀ = a₂ - a₁) →
  -- The middle (5th) term is 1120x⁴
  (coeff n 4 = 1120) ∧
  -- The sum of coefficients of odd powers is 3280
  (coeff n 1 + coeff n 3 + coeff n 5 + coeff n 7 = 3280) :=
by sorry

end expansion_properties_l240_24052


namespace currency_exchange_problem_l240_24061

/-- 
Proves the existence of a positive integer d that satisfies the conditions
of the currency exchange problem and has a digit sum of 3.
-/
theorem currency_exchange_problem : ∃ d : ℕ+, 
  (8 : ℚ) / 5 * d.val - 72 = d.val ∧ 
  (d.val.repr.toList.map (λ c => c.toString.toNat!)).sum = 3 := by
  sorry

end currency_exchange_problem_l240_24061


namespace multiply_powers_with_coefficient_l240_24078

theorem multiply_powers_with_coefficient (a : ℝ) : 2 * (a^2 * a^4) = 2 * a^6 := by
  sorry

end multiply_powers_with_coefficient_l240_24078


namespace intersection_A_B_l240_24081

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end intersection_A_B_l240_24081


namespace solution_set_properties_l240_24043

def M : Set ℝ := {x : ℝ | 3 - 2*x < 0}

theorem solution_set_properties : (0 ∉ M) ∧ (2 ∈ M) := by
  sorry

end solution_set_properties_l240_24043


namespace r_to_s_conversion_l240_24062

/-- Given a linear relationship between r and s scales, prove that r = 48 corresponds to s = 100 -/
theorem r_to_s_conversion (r s : ℝ → ℝ) : 
  (∃ a b : ℝ, ∀ x, s x = a * x + b) →  -- Linear relationship
  s 6 = 30 →                          -- First given point
  s 24 = 60 →                         -- Second given point
  s 48 = 100 :=                       -- Conclusion to prove
by sorry

end r_to_s_conversion_l240_24062


namespace unique_fixed_point_l240_24012

-- Define the type for points in the plane
variable (Point : Type)

-- Define the type for lines in the plane
variable (Line : Type)

-- Define the set of all lines in the plane
variable (L : Set Line)

-- Define the function f that assigns a point to each line
variable (f : Line → Point)

-- Define a predicate to check if a point is on a line
variable (on_line : Point → Line → Prop)

-- Define a predicate to check if points are on a circle
variable (on_circle : Point → Point → Point → Point → Prop)

-- Axiom: f(l) is on l for all lines l
axiom f_on_line : ∀ l : Line, on_line (f l) l

-- Axiom: For any point X and any three lines l1, l2, l3 passing through X,
--        the points f(l1), f(l2), f(l3), and X lie on a circle
axiom circle_property : 
  ∀ (X : Point) (l1 l2 l3 : Line),
  on_line X l1 → on_line X l2 → on_line X l3 →
  on_circle X (f l1) (f l2) (f l3)

-- Theorem: There exists a unique point P such that f(l) = P for any line l passing through P
theorem unique_fixed_point :
  ∃! P : Point, ∀ l : Line, on_line P l → f l = P :=
sorry

end unique_fixed_point_l240_24012


namespace sale_price_calculation_l240_24063

def ticket_price : ℝ := 25
def discount_rate : ℝ := 0.25

theorem sale_price_calculation :
  ticket_price * (1 - discount_rate) = 18.75 := by
  sorry

end sale_price_calculation_l240_24063


namespace pet_store_gerbils_l240_24024

/-- The number of gerbils left in a pet store after some are sold -/
def gerbils_left (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Theorem: Given 85 initial gerbils and 69 sold, 16 gerbils are left -/
theorem pet_store_gerbils : gerbils_left 85 69 = 16 := by
  sorry

end pet_store_gerbils_l240_24024


namespace difference_of_cubes_factorization_l240_24071

theorem difference_of_cubes_factorization (a b c d e : ℚ) :
  (∀ x, 512 * x^3 - 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 102 := by
sorry

end difference_of_cubes_factorization_l240_24071


namespace polygon_angles_l240_24056

theorem polygon_angles (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end polygon_angles_l240_24056


namespace solution_set_subset_interval_l240_24016

def solution_set (a : ℝ) : Set ℝ :=
  {x | x^2 - 2*a*x + a + 2 ≤ 0}

theorem solution_set_subset_interval (a : ℝ) :
  solution_set a ⊆ Set.Icc 1 4 ↔ a ∈ Set.Ioo (-1) (18/7) :=
sorry

end solution_set_subset_interval_l240_24016


namespace ben_whitewashed_length_l240_24064

theorem ben_whitewashed_length (total_length : ℝ) (remaining_length : ℝ)
  (h1 : total_length = 100)
  (h2 : remaining_length = 48)
  (h3 : ∃ x : ℝ, 
    remaining_length = total_length - x - 
    (1/5) * (total_length - x) - 
    (1/3) * (total_length - x - (1/5) * (total_length - x))) :
  ∃ x : ℝ, x = 10 ∧ 
    remaining_length = total_length - x - 
    (1/5) * (total_length - x) - 
    (1/3) * (total_length - x - (1/5) * (total_length - x)) :=
by sorry

end ben_whitewashed_length_l240_24064


namespace geese_duck_difference_l240_24053

/-- The number of more geese than ducks remaining at the duck park after a series of events --/
theorem geese_duck_difference : ℕ := by
  -- Define initial numbers
  let initial_ducks : ℕ := 25
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let initial_swans : ℕ := 3 * initial_ducks + 8

  -- Define changes in population
  let arriving_ducks : ℕ := 4
  let arriving_geese : ℕ := 7
  let leaving_swans : ℕ := 9
  let leaving_geese : ℕ := 5
  let returning_geese : ℕ := 15
  let returning_swans : ℕ := 11

  -- Calculate intermediate populations
  let ducks_after_arrival : ℕ := initial_ducks + arriving_ducks
  let geese_after_arrival : ℕ := initial_geese + arriving_geese
  let swans_after_leaving : ℕ := initial_swans - leaving_swans
  let geese_after_leaving : ℕ := geese_after_arrival - leaving_geese
  let final_geese : ℕ := geese_after_leaving + returning_geese
  let final_swans : ℕ := swans_after_leaving + returning_swans

  -- Calculate birds leaving
  let leaving_ducks : ℕ := 2 * ducks_after_arrival
  let leaving_swans : ℕ := final_swans / 2

  -- Calculate final populations
  let remaining_ducks : ℕ := ducks_after_arrival - leaving_ducks
  let remaining_geese : ℕ := final_geese

  -- Prove the difference
  have h : remaining_geese - remaining_ducks = 57 := by sorry

  exact 57

end geese_duck_difference_l240_24053


namespace existence_of_homomorphism_l240_24083

variable {G : Type*} [Group G]

def special_function (φ : G → G) : Prop :=
  ∀ a b c d e f : G, a * b * c = 1 ∧ d * e * f = 1 → φ a * φ b * φ c = φ d * φ e * φ f

theorem existence_of_homomorphism (φ : G → G) (h : special_function φ) :
  ∃ k : G, ∀ x y : G, k * φ (x * y) = (k * φ x) * (k * φ y) := by
  sorry

end existence_of_homomorphism_l240_24083


namespace miguels_wall_paint_area_l240_24092

/-- The area to be painted on a wall with given dimensions and a window -/
def area_to_paint (wall_height wall_length window_side : ℝ) : ℝ :=
  wall_height * wall_length - window_side * window_side

/-- Theorem stating the area to be painted for Miguel's wall -/
theorem miguels_wall_paint_area :
  area_to_paint 10 15 3 = 141 := by
  sorry

end miguels_wall_paint_area_l240_24092


namespace arrangement_count_is_2028_l240_24003

/-- Represents the set of files that can be arranged after lunch -/
def RemainingFiles : Finset ℕ := Finset.range 9 ∪ {12}

/-- The number of ways to arrange a subset of files from {1,2,...,9,12} -/
def ArrangementCount : ℕ := sorry

/-- Theorem stating that the number of different arrangements is 2028 -/
theorem arrangement_count_is_2028 : ArrangementCount = 2028 := by sorry

end arrangement_count_is_2028_l240_24003


namespace cricketer_average_score_l240_24002

theorem cricketer_average_score (avg1 avg2 overall_avg : ℚ) (n1 n2 : ℕ) : 
  avg1 = 30 → avg2 = 40 → overall_avg = 36 → n1 = 2 → n2 = 3 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = overall_avg →
  n1 + n2 = 5 := by sorry

end cricketer_average_score_l240_24002


namespace regular_octagon_diagonal_ratio_regular_octagon_diagonal_ratio_proof_l240_24037

/-- The ratio of the shorter diagonal to the longer diagonal in a regular octagon -/
theorem regular_octagon_diagonal_ratio : ℝ :=
  1 / Real.sqrt 2

/-- Proof that the ratio of the shorter diagonal to the longer diagonal in a regular octagon is 1 / √2 -/
theorem regular_octagon_diagonal_ratio_proof :
  regular_octagon_diagonal_ratio = 1 / Real.sqrt 2 := by
  sorry

end regular_octagon_diagonal_ratio_regular_octagon_diagonal_ratio_proof_l240_24037


namespace hotel_rate_problem_l240_24067

theorem hotel_rate_problem (f n : ℝ) 
  (h1 : f + 3 * n = 210)  -- 4-night stay cost
  (h2 : f + 6 * n = 350)  -- 7-night stay cost
  : f = 70 := by
  sorry

end hotel_rate_problem_l240_24067


namespace kitchen_clock_correct_time_bedroom_clock_correct_time_clocks_same_time_l240_24098

-- Constants
def minutes_per_hour : ℚ := 60
def hours_per_day : ℚ := 24
def clock_cycle_minutes : ℚ := 720

-- Clock rates
def kitchen_clock_advance_rate : ℚ := 1.5
def bedroom_clock_slow_rate : ℚ := 0.5

-- Theorem for kitchen clock
theorem kitchen_clock_correct_time (t : ℚ) :
  t * kitchen_clock_advance_rate = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 20 := by sorry

-- Theorem for bedroom clock
theorem bedroom_clock_correct_time (t : ℚ) :
  t * bedroom_clock_slow_rate = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 60 := by sorry

-- Theorem for both clocks showing the same time
theorem clocks_same_time (t : ℚ) :
  t * (kitchen_clock_advance_rate + bedroom_clock_slow_rate) = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 15 := by sorry

end kitchen_clock_correct_time_bedroom_clock_correct_time_clocks_same_time_l240_24098


namespace greater_than_implies_half_greater_than_l240_24095

theorem greater_than_implies_half_greater_than (a b : ℝ) (h : a > b) : a / 2 > b / 2 := by
  sorry

end greater_than_implies_half_greater_than_l240_24095


namespace intersection_complement_theorem_l240_24004

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 4}

-- Define set N
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- The theorem to prove
theorem intersection_complement_theorem :
  N ∩ (Set.compl M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_complement_theorem_l240_24004


namespace basketball_substitutions_l240_24035

/-- Represents the number of ways to make exactly n substitutions -/
def b (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 5 * (11 - n) * b n

/-- The total number of ways to make substitutions -/
def m : ℕ := (b 0) + (b 1) + (b 2) + (b 3) + (b 4) + (b 5)

theorem basketball_substitutions :
  m % 1000 = 301 := by sorry

end basketball_substitutions_l240_24035


namespace a_n_bounds_l240_24006

variable (n : ℕ+)

noncomputable def a : ℕ → ℚ
  | 0 => 1/2
  | k + 1 => a k + (1/n) * (a k)^2

theorem a_n_bounds : 1 - 1/n < a n n ∧ a n n < 1 := by sorry

end a_n_bounds_l240_24006


namespace factorization_equality_l240_24059

theorem factorization_equality (x : ℝ) : 9*x^3 - 18*x^2 + 9*x = 9*x*(x-1)^2 := by
  sorry

end factorization_equality_l240_24059


namespace sqrt_two_irrational_and_greater_than_one_l240_24068

theorem sqrt_two_irrational_and_greater_than_one :
  ∃ x : ℝ, Irrational x ∧ x > 1 :=
by
  use Real.sqrt 2
  sorry

end sqrt_two_irrational_and_greater_than_one_l240_24068


namespace saltwater_volume_proof_l240_24057

theorem saltwater_volume_proof (x : ℝ) : 
  (0.20 * x + 12) / (0.75 * x + 18) = 1/3 → x = 120 := by
  sorry

end saltwater_volume_proof_l240_24057


namespace shifted_line_not_in_third_quadrant_l240_24041

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.slope * shift + l.intercept }

/-- Checks if a line passes through the third quadrant -/
def passes_through_third_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = l.slope * x + l.intercept

/-- The original line y = -2x - 1 -/
def original_line : Line :=
  { slope := -2, intercept := -1 }

/-- The amount of right shift -/
def shift_amount : ℝ := 3

theorem shifted_line_not_in_third_quadrant :
  ¬ passes_through_third_quadrant (shift_line original_line shift_amount) := by
  sorry

end shifted_line_not_in_third_quadrant_l240_24041


namespace intersection_A_complement_B_find_m_for_intersection_l240_24086

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem for part (1)
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B 3) = Set.Icc 3 5 := by sorry

-- Theorem for part (2)
theorem find_m_for_intersection : 
  ∃ m : ℝ, A ∩ B m = Set.Ioo (-1) 4 → m = 8 := by sorry

end intersection_A_complement_B_find_m_for_intersection_l240_24086


namespace divisible_by_nine_l240_24029

/-- The eight-digit number in the form 973m2158 -/
def eight_digit_number (m : ℕ) : ℕ := 973000000 + m * 10000 + 2158

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100000000) + ((n / 10000000) % 10) + ((n / 1000000) % 10) + 
  ((n / 100000) % 10) + ((n / 10000) % 10) + ((n / 1000) % 10) + 
  ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem divisible_by_nine (m : ℕ) : 
  (eight_digit_number m) % 9 = 0 ↔ m = 1 :=
sorry

end divisible_by_nine_l240_24029


namespace initial_number_of_girls_l240_24070

/-- The initial number of girls -/
def n : ℕ := sorry

/-- The initial average weight of the girls -/
def A : ℝ := sorry

/-- The weight of the replaced girl -/
def replaced_weight : ℝ := 40

/-- The weight of the new girl -/
def new_weight : ℝ := 80

/-- The increase in average weight -/
def avg_increase : ℝ := 2

theorem initial_number_of_girls :
  (n : ℝ) * (A + avg_increase) - n * A = new_weight - replaced_weight →
  n = 20 := by sorry

end initial_number_of_girls_l240_24070


namespace equation_solution_l240_24097

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by sorry

end equation_solution_l240_24097


namespace sqrt_sum_square_condition_l240_24073

theorem sqrt_sum_square_condition (a b : ℝ) :
  Real.sqrt (a^2 + b^2 + 2*a*b) = a + b ↔ a + b ≥ 0 := by sorry

end sqrt_sum_square_condition_l240_24073


namespace subset_intersection_iff_bounds_l240_24026

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem subset_intersection_iff_bounds (a : ℝ) :
  (A a).Nonempty → (A a ⊆ A a ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by
  sorry

#check subset_intersection_iff_bounds

end subset_intersection_iff_bounds_l240_24026


namespace trapezium_other_side_length_l240_24001

theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 15 → area = 285 → area = (a + b) * h / 2 → b = 18 := by
  sorry

end trapezium_other_side_length_l240_24001


namespace power_of_power_negative_l240_24096

theorem power_of_power_negative (a : ℝ) : -(a^3)^4 = -a^12 := by
  sorry

end power_of_power_negative_l240_24096


namespace incenter_characterization_l240_24088

/-- Triangle ABC with point P inside -/
structure Triangle :=
  (A B C P : ℝ × ℝ)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Perpendicular distance from a point to a line segment -/
def perpDistance (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Length of a line segment -/
def segmentLength (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

theorem incenter_characterization (t : Triangle) :
  let l := perimeter t
  let s := area t
  let PD := perpDistance t.P (t.A, t.B)
  let PE := perpDistance t.P (t.B, t.C)
  let PF := perpDistance t.P (t.C, t.A)
  let AB := segmentLength (t.A, t.B)
  let BC := segmentLength (t.B, t.C)
  let CA := segmentLength (t.C, t.A)
  AB / PD + BC / PE + CA / PF ≤ l^2 / (2 * s) →
  t.P = incenter t :=
by sorry

end incenter_characterization_l240_24088


namespace probability_two_red_one_black_l240_24077

def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls
def num_draws : ℕ := 3

def prob_red : ℚ := num_red_balls / total_balls
def prob_black : ℚ := num_black_balls / total_balls

def prob_two_red_one_black : ℚ := 3 * (prob_red * prob_red * prob_black)

theorem probability_two_red_one_black : 
  prob_two_red_one_black = 144 / 343 := by
  sorry

end probability_two_red_one_black_l240_24077


namespace quadratic_discriminant_l240_24089

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x + 1/2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 81/4 := by sorry

end quadratic_discriminant_l240_24089


namespace square_minus_product_identity_l240_24018

theorem square_minus_product_identity (x y : ℝ) :
  (2*x - 3*y)^2 - (2*x + 3*y)*(2*x - 3*y) = -12*x*y + 18*y^2 := by
  sorry

end square_minus_product_identity_l240_24018


namespace teacher_friends_count_l240_24060

theorem teacher_friends_count (total_students : ℕ) 
  (both_friends : ℕ) (neither_friends : ℕ) (friend_difference : ℕ) :
  total_students = 50 →
  both_friends = 30 →
  neither_friends = 1 →
  friend_difference = 7 →
  ∃ (zhang_friends : ℕ),
    zhang_friends = 43 ∧
    zhang_friends + (zhang_friends - friend_difference) - both_friends + neither_friends = total_students :=
by sorry

end teacher_friends_count_l240_24060


namespace wang_hao_height_l240_24039

/-- Given Yao Ming's height and the difference between Yao Ming's and Wang Hao's heights,
    prove that Wang Hao's height is 1.58 meters. -/
theorem wang_hao_height (yao_ming_height : ℝ) (height_difference : ℝ) 
  (h1 : yao_ming_height = 2.29)
  (h2 : height_difference = 0.71) :
  yao_ming_height - height_difference = 1.58 := by
  sorry

end wang_hao_height_l240_24039


namespace winning_candidate_percentage_l240_24023

/-- Given an election with a total of 5200 votes where the winning candidate
    has a majority of 1040 votes, prove that the winning candidate received 60% of the votes. -/
theorem winning_candidate_percentage (total_votes : ℕ) (majority : ℕ) 
  (h_total : total_votes = 5200)
  (h_majority : majority = 1040) :
  (majority : ℚ) / total_votes * 100 + 50 = 60 := by
  sorry

end winning_candidate_percentage_l240_24023


namespace arithmetic_sequence_sum_l240_24065

/-- An arithmetic sequence with its partial sums. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem. -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.S 4 = 8)
    (h2 : seq.S 8 = 20) :
    seq.a 11 + seq.a 12 + seq.a 13 + seq.a 14 = 18 := by
  sorry

end arithmetic_sequence_sum_l240_24065


namespace quadratic_one_solution_find_m_l240_24021

/-- A quadratic equation ax² + bx + c = 0 has exactly one solution if and only if its discriminant b² - 4ac = 0 -/
theorem quadratic_one_solution (a b c : ℝ) (h : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0 := by sorry

theorem find_m : ∃ m : ℚ, (∃! x : ℝ, 3 * x^2 - 7 * x + m = 0) → m = 49/12 := by
  sorry

end quadratic_one_solution_find_m_l240_24021


namespace day_15_net_income_l240_24045

/-- Calculate the net income on a given day of business -/
def net_income (initial_income : ℝ) (daily_multiplier : ℝ) (daily_expenses : ℝ) (tax_rate : ℝ) (day : ℕ) : ℝ :=
  let gross_income := initial_income * daily_multiplier^(day - 1)
  let tax := tax_rate * gross_income
  let after_tax := gross_income - tax
  after_tax - daily_expenses

/-- The net income on the 15th day of business is $12,913,916.3 -/
theorem day_15_net_income :
  net_income 3 3 100 0.1 15 = 12913916.3 := by
  sorry

end day_15_net_income_l240_24045


namespace one_ton_equals_2000_pounds_l240_24014

/-- The weight of a blue whale's tongue in pounds -/
def tongue_weight_pounds : ℕ := 6000

/-- The weight of a blue whale's tongue in tons -/
def tongue_weight_tons : ℕ := 3

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := tongue_weight_pounds / tongue_weight_tons

theorem one_ton_equals_2000_pounds : pounds_per_ton = 2000 := by sorry

end one_ton_equals_2000_pounds_l240_24014


namespace survey_response_rate_change_l240_24082

theorem survey_response_rate_change 
  (original_customers : Nat) 
  (original_responses : Nat)
  (final_customers : Nat)
  (final_responses : Nat)
  (h1 : original_customers = 100)
  (h2 : original_responses = 10)
  (h3 : final_customers = 90)
  (h4 : final_responses = 27) :
  ((final_responses : ℝ) / final_customers - (original_responses : ℝ) / original_customers) / 
  ((original_responses : ℝ) / original_customers) * 100 = 200 := by
sorry

end survey_response_rate_change_l240_24082


namespace proposition_implication_l240_24038

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k ≥ 1 → (P k → P (k + 1)))
  (h2 : ¬ P 10) : 
  ¬ P 9 := by sorry

end proposition_implication_l240_24038


namespace johns_journey_length_l240_24072

theorem johns_journey_length :
  ∀ (total_length : ℝ),
  (total_length / 4 : ℝ) + 30 + (1/3 : ℝ) * (total_length - total_length / 4 - 30) = total_length →
  total_length = 160 := by
sorry

end johns_journey_length_l240_24072


namespace sin_150_degrees_l240_24049

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l240_24049


namespace star_polygon_n_is_24_l240_24054

/-- A n-pointed regular star polygon -/
structure StarPolygon (n : ℕ) where
  edges : Fin (2 * n) → ℝ
  angleA : Fin n → ℝ
  angleB : Fin n → ℝ
  edges_congruent : ∀ i j, edges i = edges j
  angleA_congruent : ∀ i j, angleA i = angleA j
  angleB_congruent : ∀ i j, angleB i = angleB j
  angle_difference : ∀ i, angleA i = angleB i - 15

/-- The theorem stating that n = 24 for the given star polygon -/
theorem star_polygon_n_is_24 (n : ℕ) (star : StarPolygon n) : n = 24 := by
  sorry

end star_polygon_n_is_24_l240_24054


namespace modified_triathlon_speed_l240_24008

theorem modified_triathlon_speed (total_time : ℝ) (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ) (kayak_distance kayak_speed : ℝ)
  (bike_distance : ℝ) :
  total_time = 3 ∧
  swim_distance = 1/2 ∧ swim_speed = 2 ∧
  run_distance = 5 ∧ run_speed = 10 ∧
  kayak_distance = 1 ∧ kayak_speed = 3 ∧
  bike_distance = 20 →
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed + kayak_distance / kayak_speed))) = 240/23 :=
by sorry

end modified_triathlon_speed_l240_24008


namespace alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two_l240_24017

theorem alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two
  (α : ℝ) (h1 : α ≠ 0) (h2 : α + Real.tan α = 0) :
  (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := by
  sorry

end alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two_l240_24017


namespace select_two_from_five_assign_prizes_l240_24074

/-- The number of ways to select 2 people from n employees and assign them distinct prizes -/
def select_and_assign (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: For 5 employees, there are 20 ways to select 2 and assign distinct prizes -/
theorem select_two_from_five_assign_prizes :
  select_and_assign 5 = 20 := by
  sorry

end select_two_from_five_assign_prizes_l240_24074


namespace bobs_smallest_number_l240_24019

def is_valid_bob_number (alice_num bob_num : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ alice_num → p ∣ bob_num

def has_additional_prime_factor (alice_num bob_num : ℕ) : Prop :=
  ∃ q : ℕ, q.Prime ∧ q ∣ bob_num ∧ ¬(q ∣ alice_num)

theorem bobs_smallest_number (alice_num : ℕ) (bob_num : ℕ) :
  alice_num = 36 →
  is_valid_bob_number alice_num bob_num →
  has_additional_prime_factor alice_num bob_num →
  (∀ n : ℕ, n < bob_num →
    ¬(is_valid_bob_number alice_num n ∧ has_additional_prime_factor alice_num n)) →
  bob_num = 30 :=
sorry

end bobs_smallest_number_l240_24019


namespace solutions_for_20_l240_24034

/-- The number of distinct integer solutions (x, y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := sorry

/-- The theorem stating the number of solutions for n = 20 -/
theorem solutions_for_20 :
  num_solutions 1 = 4 →
  num_solutions 2 = 8 →
  num_solutions 3 = 12 →
  num_solutions 20 = 80 := by sorry

end solutions_for_20_l240_24034


namespace discount_profit_theorem_l240_24093

/-- Given a discount percentage and a profit percentage without discount,
    calculate the profit percentage with the discount. -/
def profit_with_discount (discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  (1 + profit_without_discount) * (1 - discount) - 1

/-- Theorem stating that with a 4% discount and 25% profit without discount,
    the profit percentage with discount is 20%. -/
theorem discount_profit_theorem :
  profit_with_discount 0.04 0.25 = 0.20 := by
  sorry

#eval profit_with_discount 0.04 0.25

end discount_profit_theorem_l240_24093


namespace congruence_solutions_count_l240_24013

theorem congruence_solutions_count : 
  ∃! (s : Finset ℤ), 
    (∀ x ∈ s, (x^3 + 3*x^2 + x + 3) % 25 = 0) ∧ 
    (∀ x, (x^3 + 3*x^2 + x + 3) % 25 = 0 → x % 25 ∈ s) ∧ 
    s.card = 6 :=
by sorry

end congruence_solutions_count_l240_24013


namespace ellipse_m_range_collinearity_AGN_l240_24000

-- Define the curve C
def C (m : ℝ) (x y : ℝ) : Prop := (5 - m) * x^2 + (m - 2) * y^2 = 8

-- Define the condition for C to be an ellipse with foci on x-axis
def is_ellipse_x_foci (m : ℝ) : Prop :=
  (8 / (5 - m) > 8 / (m - 2)) ∧ (8 / (5 - m) > 0) ∧ (8 / (m - 2) > 0)

-- Define the line y = kx + 4
def line_k (k : ℝ) (x y : ℝ) : Prop := y = k * x + 4

-- Define the line y = 1
def line_one (x y : ℝ) : Prop := y = 1

-- Theorem for part 1
theorem ellipse_m_range (m : ℝ) :
  is_ellipse_x_foci m → (7/2 < m) ∧ (m < 5) := by sorry

-- Theorem for part 2
theorem collinearity_AGN (k : ℝ) (xA yA xB yB xM yM xN yN xG : ℝ) :
  C 4 0 yA ∧ C 4 0 yB ∧ yA > yB ∧
  C 4 xM yM ∧ C 4 xN yN ∧
  line_k k xM yM ∧ line_k k xN yN ∧
  line_one xG 1 ∧
  (yM - yB) / (xM - xB) = (1 - yB) / (xG - xB) →
  ∃ (t : ℝ), xG = t * xA + (1 - t) * xN ∧ 1 = t * yA + (1 - t) * yN := by sorry

end ellipse_m_range_collinearity_AGN_l240_24000


namespace tan_theta_two_implies_expression_equals_negative_two_l240_24087

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end tan_theta_two_implies_expression_equals_negative_two_l240_24087


namespace sum_9000_eq_1355_l240_24080

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  /-- The sum of the first 3000 terms -/
  sum_3000 : ℝ
  /-- The sum of the first 6000 terms -/
  sum_6000 : ℝ
  /-- The sum of the first 3000 terms is 500 -/
  sum_3000_eq : sum_3000 = 500
  /-- The sum of the first 6000 terms is 950 -/
  sum_6000_eq : sum_6000 = 950

/-- The sum of the first 9000 terms of the geometric sequence is 1355 -/
theorem sum_9000_eq_1355 (seq : GeometricSequence) : ℝ := by
  sorry

end sum_9000_eq_1355_l240_24080


namespace smallest_representable_integer_l240_24044

theorem smallest_representable_integer :
  ∃ (m n : ℕ+), 11 = 36 * m - 5 * n ∧
  ∀ (k : ℕ+) (m' n' : ℕ+), k < 11 → k ≠ 36 * m' - 5 * n' :=
sorry

end smallest_representable_integer_l240_24044


namespace all_statements_false_l240_24099

theorem all_statements_false : 
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧ 
  (¬ ∀ (x y : ℚ), x = -y → x = y⁻¹) ∧ 
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) := by
  sorry

end all_statements_false_l240_24099


namespace average_speed_calculation_l240_24009

/-- Calculates the average speed of a trip given the conditions specified in the problem -/
theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

end average_speed_calculation_l240_24009


namespace maximum_marks_calculation_l240_24005

theorem maximum_marks_calculation (passing_threshold : ℝ) (scored_marks : ℕ) (shortfall : ℕ) : 
  passing_threshold = 30 / 100 →
  scored_marks = 212 →
  shortfall = 16 →
  ∃ (total_marks : ℕ), total_marks = 760 ∧ 
    (scored_marks + shortfall : ℝ) / total_marks = passing_threshold :=
by sorry

end maximum_marks_calculation_l240_24005


namespace tournament_games_theorem_l240_24046

/-- A single-elimination tournament with no ties. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games needed to declare a winner in a single-elimination tournament. -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games played to declare a winner is 22. -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.no_ties = true) : 
  games_to_winner t = 22 := by
  sorry


end tournament_games_theorem_l240_24046


namespace picnic_watermelon_slices_l240_24075

/-- The number of watermelons Danny brings -/
def danny_watermelons : ℕ := 3

/-- The number of slices Danny cuts each watermelon into -/
def danny_slices_per_watermelon : ℕ := 10

/-- The number of watermelons Danny's sister brings -/
def sister_watermelons : ℕ := 1

/-- The number of slices Danny's sister cuts her watermelon into -/
def sister_slices_per_watermelon : ℕ := 15

/-- The total number of watermelon slices at the picnic -/
def total_slices : ℕ := danny_watermelons * danny_slices_per_watermelon + sister_watermelons * sister_slices_per_watermelon

theorem picnic_watermelon_slices : total_slices = 45 := by
  sorry

end picnic_watermelon_slices_l240_24075


namespace trigonometric_identity_l240_24050

theorem trigonometric_identity (α : Real) (m : Real) (h : Real.tan α = m) :
  Real.sin (π/4 + α)^2 - Real.sin (π/6 - α)^2 - Real.cos (5*π/12) * Real.sin (5*π/12 - 2*α) = 2*m / (1 + m^2) := by
  sorry

end trigonometric_identity_l240_24050


namespace q_div_p_equals_225_l240_24055

def total_cards : ℕ := 50
def num_range : Set ℕ := Finset.range 10
def cards_per_num : ℕ := 5
def drawn_cards : ℕ := 5

def p : ℚ := 10 / Nat.choose total_cards drawn_cards
def q : ℚ := (10 * 9 * cards_per_num * cards_per_num) / Nat.choose total_cards drawn_cards

theorem q_div_p_equals_225 : q / p = 225 := by sorry

end q_div_p_equals_225_l240_24055


namespace min_value_at_four_l240_24033

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- Theorem stating that f(x) achieves its minimum when x = 4 -/
theorem min_value_at_four :
  ∀ x : ℝ, f x ≥ f 4 := by sorry

end min_value_at_four_l240_24033


namespace negative_fraction_multiplication_l240_24051

theorem negative_fraction_multiplication :
  ((-144 : ℤ) / (-36 : ℤ)) * 3 = 12 := by sorry

end negative_fraction_multiplication_l240_24051


namespace rikshaw_charge_theorem_l240_24047

/-- Represents the rikshaw charging system in Mumbai -/
structure RikshawCharge where
  base_charge : ℝ  -- Charge for the first 1 km
  rate_1_5 : ℝ     -- Rate per km for 1-5 km
  rate_5_10 : ℝ    -- Rate per 1/3 km for 5-10 km
  rate_10_plus : ℝ -- Rate per 1/3 km beyond 10 km
  wait_rate : ℝ    -- Waiting charge per hour after first 10 minutes

/-- Calculates the total charge for a rikshaw ride -/
def calculate_charge (c : RikshawCharge) (distance : ℝ) (wait_time : ℝ) : ℝ :=
  sorry

/-- The theorem stating the total charge for the given ride -/
theorem rikshaw_charge_theorem (c : RikshawCharge) 
  (h1 : c.base_charge = 18.5)
  (h2 : c.rate_1_5 = 3)
  (h3 : c.rate_5_10 = 2.5)
  (h4 : c.rate_10_plus = 4)
  (h5 : c.wait_rate = 20) :
  calculate_charge c 16 1.5 = 170 :=
sorry

end rikshaw_charge_theorem_l240_24047
