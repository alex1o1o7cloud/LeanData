import Mathlib

namespace tetrahedron_volume_formula_l82_8255

/-- A tetrahedron with an inscribed sphere. -/
structure TetrahedronWithInscribedSphere where
  R : ℝ  -- Radius of the inscribed sphere
  S₁ : ℝ  -- Area of face 1
  S₂ : ℝ  -- Area of face 2
  S₃ : ℝ  -- Area of face 3
  S₄ : ℝ  -- Area of face 4

/-- The volume of a tetrahedron with an inscribed sphere. -/
def volume (t : TetrahedronWithInscribedSphere) : ℝ :=
  t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄)

/-- Theorem: The volume of a tetrahedron with an inscribed sphere
    is equal to the radius of the inscribed sphere multiplied by
    the sum of the areas of its four faces. -/
theorem tetrahedron_volume_formula (t : TetrahedronWithInscribedSphere) :
  volume t = t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄) := by
  sorry

end tetrahedron_volume_formula_l82_8255


namespace pauls_money_duration_l82_8256

/-- 
Given Paul's earnings and weekly spending, prove how long the money will last.
-/
theorem pauls_money_duration (lawn_mowing : ℕ) (weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : lawn_mowing = 44)
  (h2 : weed_eating = 28)
  (h3 : weekly_spending = 9) :
  (lawn_mowing + weed_eating) / weekly_spending = 8 := by
  sorry

end pauls_money_duration_l82_8256


namespace beatles_collection_theorem_l82_8221

/-- The number of albums in either Andrew's or John's collection, but not both -/
def unique_albums (shared : ℕ) (andrew_total : ℕ) (john_unique : ℕ) : ℕ :=
  (andrew_total - shared) + john_unique

theorem beatles_collection_theorem :
  unique_albums 9 17 6 = 14 := by
  sorry

end beatles_collection_theorem_l82_8221


namespace hyperbola_eccentricity_l82_8284

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 2

-- Define the eccentricity
def eccentricity (e a c : ℝ) : Prop := e = c / a

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : focal_length 2) 
  (hp : hyperbola a b 2 3) : 
  ∃ e, eccentricity e a 2 ∧ e = 2 := by
sorry

end hyperbola_eccentricity_l82_8284


namespace regular_octagon_interior_angle_is_135_l82_8235

/-- The measure of each interior angle of a regular octagon in degrees. -/
def regular_octagon_interior_angle : ℝ := 135

/-- Theorem stating that the measure of each interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by sorry

end regular_octagon_interior_angle_is_135_l82_8235


namespace range_of_m_l82_8292

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 14 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ≤ 4 :=
by sorry

end range_of_m_l82_8292


namespace f_max_min_in_interval_l82_8215

-- Define the function f
def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f satisfies f(1 + x) = f(1 - x) for all x
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)

-- f is monotonically increasing in [-1, 1]
axiom f_monotone : ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

-- Theorem statement
theorem f_max_min_in_interval :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ f 1) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f 3 ≤ f x) :=
sorry

end f_max_min_in_interval_l82_8215


namespace equation_solution_l82_8243

def solution_set : Set ℝ := {-Real.sqrt 10, -Real.pi, -1, 1, Real.pi, Real.sqrt 10}

def domain (x : ℝ) : Prop :=
  (-Real.sqrt 10 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ Real.sqrt 10)

theorem equation_solution :
  ∀ x : ℝ, domain x →
    ((Real.sin (2 * x) - Real.pi * Real.sin x) * Real.sqrt (11 * x^2 - x^4 - 10) = 0 ↔ x ∈ solution_set) :=
by sorry

end equation_solution_l82_8243


namespace g_difference_l82_8213

/-- The function g(x) = 3x^3 + 4x^2 - 3x + 2 -/
def g (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 - 3 * x + 2

/-- Theorem stating that g(x + h) - g(x) = h(9x^2 + 8x + 9xh + 4h + 3h^2 - 3) for all x and h -/
theorem g_difference (x h : ℝ) : 
  g (x + h) - g x = h * (9 * x^2 + 8 * x + 9 * x * h + 4 * h + 3 * h^2 - 3) := by
  sorry

end g_difference_l82_8213


namespace sin_product_equality_l82_8236

theorem sin_product_equality : 
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) * Real.sin (60 * π / 180) * Real.sin (85 * π / 180) =
  Real.sin (20 * π / 180) * Real.sin (40 * π / 180) * Real.sin (75 * π / 180) * Real.sin (80 * π / 180) := by
sorry

end sin_product_equality_l82_8236


namespace intercept_sum_l82_8222

theorem intercept_sum : ∃ (x₀ y₀ : ℕ), 
  x₀ < 25 ∧ y₀ < 25 ∧
  (4 * x₀) % 25 = 2 % 25 ∧
  (5 * y₀ + 2) % 25 = 0 ∧
  x₀ + y₀ = 28 := by
sorry

end intercept_sum_l82_8222


namespace partial_fraction_sum_l82_8240

theorem partial_fraction_sum (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_l82_8240


namespace oil_depth_in_specific_tank_l82_8216

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in a horizontal cylindrical tank --/
def oilDepth (tank : HorizontalCylindricalTank) (surfaceArea : ℝ) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The depth of oil in the specified tank with given surface area --/
theorem oil_depth_in_specific_tank :
  let tank : HorizontalCylindricalTank := ⟨12, 8⟩
  let surfaceArea : ℝ := 48
  oilDepth tank surfaceArea = 4 + 2 * Real.sqrt 3 := by
  sorry

end oil_depth_in_specific_tank_l82_8216


namespace rod_and_rope_problem_l82_8200

theorem rod_and_rope_problem (x y : ℝ) : 
  (x = y + 5 ∧ x / 2 = y - 5) ↔ 
  (x - y = 5 ∧ y - x / 2 = 5) := by sorry

end rod_and_rope_problem_l82_8200


namespace hash_3_8_l82_8252

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem to prove
theorem hash_3_8 : hash 3 8 = 80 := by
  sorry

end hash_3_8_l82_8252


namespace partial_fraction_decomposition_l82_8266

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1) ∧
    P = -6 ∧ Q = 8 ∧ R = -5 := by
  sorry

end partial_fraction_decomposition_l82_8266


namespace five_digit_number_product_l82_8297

theorem five_digit_number_product (a b c d e : Nat) : 
  a ≠ 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
  (10 * a + b + 10 * b + c) * 
  (10 * b + c + 10 * c + d) * 
  (10 * c + d + 10 * d + e) = 157605 →
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 ∧ e = 5) ∨
  (a = 2 ∧ b = 1 ∧ c = 4 ∧ d = 3 ∧ e = 6) := by
  sorry

end five_digit_number_product_l82_8297


namespace line_vector_to_slope_intercept_l82_8276

/-- Given a line in vector form, prove its slope-intercept form --/
theorem line_vector_to_slope_intercept :
  let vector_form : ℝ × ℝ → Prop := λ p => (3 : ℝ) * (p.1 - 2) + (-4 : ℝ) * (p.2 + 3) = 0
  ∃ m b : ℝ, m = 3/4 ∧ b = -9/2 ∧ ∀ x y : ℝ, vector_form (x, y) ↔ y = m * x + b :=
by sorry

end line_vector_to_slope_intercept_l82_8276


namespace Q_subset_complement_P_l82_8207

-- Define the sets P and Q
def P : Set ℝ := {x | x > 4}
def Q : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the theorem
theorem Q_subset_complement_P : Q ⊆ (Set.univ \ P) := by sorry

end Q_subset_complement_P_l82_8207


namespace milk_selling_price_l82_8220

/-- Calculates the selling price of a milk-water mixture given the initial milk price, water percentage, and desired gain percentage. -/
def calculate_selling_price (milk_price : ℚ) (water_percentage : ℚ) (gain_percentage : ℚ) : ℚ :=
  let total_volume : ℚ := 1 + water_percentage
  let cost_price : ℚ := milk_price
  let selling_price : ℚ := cost_price * (1 + gain_percentage)
  selling_price / total_volume

/-- Proves that the selling price of the milk-water mixture is 15 rs per liter under the given conditions. -/
theorem milk_selling_price :
  calculate_selling_price 12 (20/100) (50/100) = 15 := by
  sorry

end milk_selling_price_l82_8220


namespace correct_outfits_l82_8294

-- Define the colors
inductive Color
| Red
| Blue

-- Define the clothing types
inductive ClothingType
| Tshirt
| Shorts

-- Define a structure for a child's outfit
structure Outfit :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

def outfit : Child → Outfit
| Child.Alyna => ⟨Color.Red, Color.Red⟩
| Child.Bohdan => ⟨Color.Red, Color.Blue⟩
| Child.Vika => ⟨Color.Blue, Color.Blue⟩
| Child.Grysha => ⟨Color.Red, Color.Blue⟩

theorem correct_outfits :
  (outfit Child.Alyna).tshirt = Color.Red ∧
  (outfit Child.Bohdan).tshirt = Color.Red ∧
  (outfit Child.Alyna).shorts ≠ (outfit Child.Bohdan).shorts ∧
  (outfit Child.Vika).tshirt ≠ (outfit Child.Grysha).tshirt ∧
  (outfit Child.Vika).shorts = Color.Blue ∧
  (outfit Child.Grysha).shorts = Color.Blue ∧
  (outfit Child.Alyna).tshirt ≠ (outfit Child.Vika).tshirt ∧
  (outfit Child.Alyna).shorts ≠ (outfit Child.Vika).shorts ∧
  (∀ c : Child, (outfit c).tshirt = Color.Red ∨ (outfit c).tshirt = Color.Blue) ∧
  (∀ c : Child, (outfit c).shorts = Color.Red ∨ (outfit c).shorts = Color.Blue) :=
by sorry

#check correct_outfits

end correct_outfits_l82_8294


namespace smallest_number_of_purple_marbles_l82_8203

theorem smallest_number_of_purple_marbles :
  ∀ (n : ℕ),
  (n ≥ 10) →  -- Ensuring n is at least 10 to satisfy all conditions
  (n % 10 = 0) →  -- n must be a multiple of 10
  (n / 2 : ℕ) + (n / 5 : ℕ) + 7 < n →  -- Ensuring there's at least one purple marble
  (∃ (blue red green purple : ℕ),
    blue = n / 2 ∧
    red = n / 5 ∧
    green = 7 ∧
    purple = n - (blue + red + green) ∧
    purple > 0) →
  (∀ (m : ℕ),
    m < n →
    ¬(∃ (blue red green purple : ℕ),
      blue = m / 2 ∧
      red = m / 5 ∧
      green = 7 ∧
      purple = m - (blue + red + green) ∧
      purple > 0)) →
  (n - (n / 2 + n / 5 + 7) = 2) :=
by sorry

end smallest_number_of_purple_marbles_l82_8203


namespace dodecagon_diagonals_plus_sides_l82_8271

/-- The number of sides in a regular dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The sum of the number of diagonals and sides in a regular dodecagon is 66 -/
theorem dodecagon_diagonals_plus_sides :
  num_diagonals dodecagon_sides + dodecagon_sides = 66 := by
  sorry

end dodecagon_diagonals_plus_sides_l82_8271


namespace nell_card_difference_l82_8244

/-- Given Nell's card collection information, prove the difference between
    her final Ace cards and baseball cards. -/
theorem nell_card_difference
  (initial_baseball : ℕ)
  (initial_ace : ℕ)
  (final_baseball : ℕ)
  (final_ace : ℕ)
  (h1 : initial_baseball = 239)
  (h2 : initial_ace = 38)
  (h3 : final_baseball = 111)
  (h4 : final_ace = 376) :
  final_ace - final_baseball = 265 := by
  sorry

end nell_card_difference_l82_8244


namespace dried_mushroom_weight_l82_8228

/-- 
Given:
- Fresh mushrooms contain 90% water by weight
- Dried mushrooms contain 12% water by weight
- We start with 22 kg of fresh mushrooms

Prove that the weight of dried mushrooms obtained is 2.5 kg
-/
theorem dried_mushroom_weight (fresh_water_content : ℝ) (dried_water_content : ℝ) 
  (fresh_weight : ℝ) (dried_weight : ℝ) :
  fresh_water_content = 0.90 →
  dried_water_content = 0.12 →
  fresh_weight = 22 →
  dried_weight = 2.5 →
  dried_weight = (1 - fresh_water_content) * fresh_weight / (1 - dried_water_content) :=
by sorry

end dried_mushroom_weight_l82_8228


namespace quiz_competition_participants_l82_8269

theorem quiz_competition_participants (total : ℕ) 
  (h1 : (total : ℝ) * (1 - 0.6) * 0.25 = 16) : total = 160 := by
  sorry

end quiz_competition_participants_l82_8269


namespace four_machines_completion_time_l82_8209

/-- A machine with a given work rate in jobs per hour -/
structure Machine where
  work_rate : ℚ

/-- The time taken for multiple machines to complete one job when working together -/
def time_to_complete (machines : List Machine) : ℚ :=
  1 / (machines.map (λ m => m.work_rate) |>.sum)

theorem four_machines_completion_time :
  let machine_a : Machine := ⟨1/4⟩
  let machine_b : Machine := ⟨1/2⟩
  let machine_c : Machine := ⟨1/6⟩
  let machine_d : Machine := ⟨1/3⟩
  let machines := [machine_a, machine_b, machine_c, machine_d]
  time_to_complete machines = 4/5 := by
  sorry

end four_machines_completion_time_l82_8209


namespace quadratic_inequality_solution_l82_8263

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a*x^2 + 2*x + 20 > 0) → a = -12 := by
  sorry

end quadratic_inequality_solution_l82_8263


namespace stickers_given_to_lucy_l82_8259

/-- Given that Gary initially had 99 stickers, gave 26 stickers to Alex, 
    and had 31 stickers left afterwards, prove that Gary gave 42 stickers to Lucy. -/
theorem stickers_given_to_lucy (initial_stickers : ℕ) (stickers_to_alex : ℕ) (stickers_left : ℕ) :
  initial_stickers = 99 →
  stickers_to_alex = 26 →
  stickers_left = 31 →
  initial_stickers - stickers_to_alex - stickers_left = 42 :=
by sorry

end stickers_given_to_lucy_l82_8259


namespace largest_equilateral_triangle_l82_8214

/-- Represents a square piece of paper -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- The folding process that creates the largest equilateral triangle from a square -/
noncomputable def foldLargestTriangle (s : Square) : EquilateralTriangle :=
  sorry

/-- Theorem stating that the triangle produced by foldLargestTriangle is the largest possible -/
theorem largest_equilateral_triangle (s : Square) :
  ∀ t : EquilateralTriangle, t.side ≤ (foldLargestTriangle s).side :=
  sorry

end largest_equilateral_triangle_l82_8214


namespace f_m_plus_one_positive_l82_8254

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_one_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := by
  sorry

end f_m_plus_one_positive_l82_8254


namespace pirate_treasure_l82_8277

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by sorry

end pirate_treasure_l82_8277


namespace ace_ten_king_of_hearts_probability_l82_8234

/-- The probability of drawing an Ace, then a 10, then the King of Hearts from a standard deck of 52 cards without replacement -/
theorem ace_ten_king_of_hearts_probability :
  let total_cards : ℕ := 52
  let aces : ℕ := 4
  let tens : ℕ := 4
  let king_of_hearts : ℕ := 1
  (aces / total_cards) * (tens / (total_cards - 1)) * (king_of_hearts / (total_cards - 2)) = 4 / 33150 := by
sorry

end ace_ten_king_of_hearts_probability_l82_8234


namespace sum_of_reciprocals_shifted_roots_l82_8233

theorem sum_of_reciprocals_shifted_roots (a b c : ℂ) : 
  (a^3 - 2*a + 4 = 0) → 
  (b^3 - 2*b + 4 = 0) → 
  (c^3 - 2*c + 4 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = -5/4) := by
  sorry

end sum_of_reciprocals_shifted_roots_l82_8233


namespace sum_of_a_and_b_l82_8283

theorem sum_of_a_and_b (a b : ℕ+) (h : a.val^2 - b.val^4 = 2009) : a.val + b.val = 47 := by
  sorry

end sum_of_a_and_b_l82_8283


namespace prob_zhong_guo_meng_correct_l82_8206

/-- The number of cards labeled "中" -/
def num_zhong : ℕ := 2

/-- The number of cards labeled "国" -/
def num_guo : ℕ := 2

/-- The number of cards labeled "梦" -/
def num_meng : ℕ := 1

/-- The total number of cards -/
def total_cards : ℕ := num_zhong + num_guo + num_meng

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing cards that form "中国梦" -/
def prob_zhong_guo_meng : ℚ := 2 / 5

theorem prob_zhong_guo_meng_correct :
  (num_zhong * num_guo * num_meng : ℚ) / (total_cards.choose cards_drawn) = prob_zhong_guo_meng := by
  sorry

end prob_zhong_guo_meng_correct_l82_8206


namespace oplus_k_oplus_k_l82_8201

-- Define the ⊕ operation
def oplus (x y : ℝ) : ℝ := x^3 - 2*y + x

-- Theorem statement
theorem oplus_k_oplus_k (k : ℝ) : oplus k (oplus k k) = -k^3 + 3*k := by
  sorry

end oplus_k_oplus_k_l82_8201


namespace area_between_concentric_circles_l82_8227

theorem area_between_concentric_circles :
  let r₁ : ℝ := 12  -- radius of larger circle
  let r₂ : ℝ := 7   -- radius of smaller circle
  let A₁ := π * r₁^2  -- area of larger circle
  let A₂ := π * r₂^2  -- area of smaller circle
  A₁ - A₂ = 95 * π := by sorry

end area_between_concentric_circles_l82_8227


namespace bus_stop_optimal_location_l82_8239

/-- Represents the distance between two buildings in meters -/
def building_distance : ℝ := 250

/-- Represents the number of students in the first building -/
def students_building1 : ℕ := 100

/-- Represents the number of students in the second building -/
def students_building2 : ℕ := 150

/-- Calculates the total walking distance for all students given the bus stop location -/
def total_walking_distance (bus_stop_location : ℝ) : ℝ :=
  students_building2 * bus_stop_location + students_building1 * (building_distance - bus_stop_location)

/-- Theorem stating that the total walking distance is minimized when the bus stop is at the second building -/
theorem bus_stop_optimal_location :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ building_distance →
    total_walking_distance 0 ≤ total_walking_distance x :=
by sorry

end bus_stop_optimal_location_l82_8239


namespace lcm_18_30_l82_8281

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l82_8281


namespace simplest_form_product_l82_8289

theorem simplest_form_product (a b : ℕ) (h : a = 45 ∧ b = 75) : 
  let g := Nat.gcd a b
  (a / g) * (b / g) = 15 := by
sorry

end simplest_form_product_l82_8289


namespace sum_of_squares_with_hcf_lcm_constraint_l82_8299

theorem sum_of_squares_with_hcf_lcm_constraint 
  (a b c : ℕ+) 
  (sum_of_squares : a^2 + b^2 + c^2 = 2011)
  (x : ℕ) 
  (hx : x = Nat.gcd a (Nat.gcd b c))
  (y : ℕ) 
  (hy : y = Nat.lcm a (Nat.lcm b c))
  (hxy : x + y = 388) : 
  a + b + c = 61 := by
sorry

end sum_of_squares_with_hcf_lcm_constraint_l82_8299


namespace domain_exclusion_sum_l82_8262

theorem domain_exclusion_sum (C D : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 8 * x + 6 = 0 ↔ (x = C ∨ x = D)) →
  C + D = 4 := by
  sorry

end domain_exclusion_sum_l82_8262


namespace adult_ticket_cost_l82_8248

theorem adult_ticket_cost (child_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (adult_count : ℕ)
  (h1 : child_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_revenue = 1875)
  (h4 : adult_count = 175) :
  (total_revenue - child_cost * (total_tickets - adult_count)) / adult_count = 9 := by
  sorry

#eval (1875 - 6 * (225 - 175)) / 175  -- Should output 9

end adult_ticket_cost_l82_8248


namespace orthogonal_vectors_l82_8286

theorem orthogonal_vectors (y : ℚ) : 
  ((-4 : ℚ) * 3 + 7 * y = 0) → y = 12/7 := by
  sorry

end orthogonal_vectors_l82_8286


namespace inequality_proof_l82_8223

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) :=
by sorry

end inequality_proof_l82_8223


namespace unique_x_intercept_l82_8237

/-- The parabola equation: x = -3y^2 + 2y + 4 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- X-intercept occurs when y = 0 -/
def x_intercept : ℝ := parabola 0

theorem unique_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end unique_x_intercept_l82_8237


namespace expression_evaluation_l82_8210

theorem expression_evaluation : -4 / (4 / 9) * (9 / 4) = -81 / 4 := by
  sorry

end expression_evaluation_l82_8210


namespace binomial_seven_two_l82_8211

theorem binomial_seven_two : Nat.choose 7 2 = 21 := by
  sorry

end binomial_seven_two_l82_8211


namespace dairy_factory_profit_comparison_l82_8224

/-- Represents the profit calculation for a dairy factory --/
theorem dairy_factory_profit_comparison :
  let total_milk : ℝ := 20
  let fresh_milk_profit : ℝ := 500
  let yogurt_profit : ℝ := 1000
  let milk_powder_profit : ℝ := 1800
  let yogurt_capacity : ℝ := 6
  let milk_powder_capacity : ℝ := 2
  let days : ℝ := 4

  let plan_one_profit : ℝ := 
    (milk_powder_capacity * days * milk_powder_profit) + 
    ((total_milk - milk_powder_capacity * days) * fresh_milk_profit)

  let plan_two_milk_powder_days : ℝ := 
    (total_milk - yogurt_capacity * days) / (yogurt_capacity - milk_powder_capacity)
  
  let plan_two_yogurt_days : ℝ := days - plan_two_milk_powder_days

  let plan_two_profit : ℝ := 
    (plan_two_milk_powder_days * milk_powder_capacity * milk_powder_profit) + 
    (plan_two_yogurt_days * yogurt_capacity * yogurt_profit)

  plan_two_profit > plan_one_profit := by sorry

end dairy_factory_profit_comparison_l82_8224


namespace smallest_prime_with_digit_sum_23_l82_8291

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 599 :=
sorry

end smallest_prime_with_digit_sum_23_l82_8291


namespace smallest_addition_for_divisibility_l82_8245

def sum_of_two_digit_pairs (n : ℕ) : ℕ :=
  (n % 100) + ((n / 100) % 100) + ((n / 10000) % 100)

def alternating_sum_of_three_digit_groups (n : ℕ) : ℤ :=
  (n % 1000 : ℤ) - ((n / 1000) % 1000 : ℤ)

theorem smallest_addition_for_divisibility (n : ℕ) (k : ℕ) :
  (∀ m < k, ¬(456 ∣ (987654 + m))) ∧
  (456 ∣ (987654 + k)) ∧
  (19 ∣ sum_of_two_digit_pairs (987654 + k)) ∧
  (8 ∣ alternating_sum_of_three_digit_groups (987654 + k)) →
  k = 22 := by
  sorry

end smallest_addition_for_divisibility_l82_8245


namespace at_least_one_geq_two_l82_8226

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end at_least_one_geq_two_l82_8226


namespace max_product_constrained_sum_l82_8275

theorem max_product_constrained_sum (x y : ℕ+) (h : 7 * x + 5 * y = 140) :
  x * y ≤ 140 ∧ ∃ (a b : ℕ+), 7 * a + 5 * b = 140 ∧ a * b = 140 := by
  sorry

end max_product_constrained_sum_l82_8275


namespace ellipse_implies_a_greater_than_one_l82_8265

/-- Represents the condition that the curve is an ellipse with foci on the x-axis -/
def is_ellipse (t : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (3 - t) + y^2 / (t + 1) = 1 → -1 < t ∧ t < 1

/-- Represents the inequality condition -/
def satisfies_inequality (t a : ℝ) : Prop :=
  t^2 - (a - 1) * t - a < 0

/-- The main theorem statement -/
theorem ellipse_implies_a_greater_than_one :
  (∀ t : ℝ, is_ellipse t → (∃ a : ℝ, satisfies_inequality t a)) ∧
  (∃ t a : ℝ, satisfies_inequality t a ∧ ¬is_ellipse t) →
  ∀ a : ℝ, (∃ t : ℝ, is_ellipse t → satisfies_inequality t a) → a > 1 :=
sorry

end ellipse_implies_a_greater_than_one_l82_8265


namespace predicted_height_at_10_l82_8250

/-- Represents the regression model for height prediction -/
def height_model (age : ℝ) : ℝ := 7.19 * age + 73.93

/-- Theorem stating that the predicted height at age 10 is approximately 145.83 cm -/
theorem predicted_height_at_10 :
  ∃ ε > 0, |height_model 10 - 145.83| < ε :=
sorry

end predicted_height_at_10_l82_8250


namespace sum_of_binary_numbers_l82_8257

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def num1 : List Bool := [true, true, true, true, true, true, true, true, true]
def num2 : List Bool := [true, false, false, false, false, false, true]

theorem sum_of_binary_numbers :
  binary_to_decimal num1 + binary_to_decimal num2 = 576 := by
  sorry

end sum_of_binary_numbers_l82_8257


namespace bernardo_wins_l82_8241

def game_winner (M : ℕ) : Prop :=
  M ≤ 999 ∧
  3 * M < 1000 ∧
  3 * M + 100 < 1000 ∧
  3 * (3 * M + 100) < 1000 ∧
  3 * (3 * M + 100) + 100 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  ∃ M : ℕ, game_winner M ∧
    (∀ N : ℕ, N < M → ¬game_winner N) ∧
    M = 67 ∧
    sum_of_digits M = 13 := by
  sorry

end bernardo_wins_l82_8241


namespace expression_simplification_l82_8217

theorem expression_simplification (a b x y : ℝ) 
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -a * y / b)
  (h3 : x ≠ b * y / a) :
  ((a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) * ((a * x + b * y)^2 - 4 * a * b * x * y)) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = a^2 * x^2 - b^2 * y^2 := by
sorry

end expression_simplification_l82_8217


namespace jonas_current_socks_l82_8204

-- Define the wardrobe items
def shoes : ℕ := 5
def pants : ℕ := 10
def tshirts : ℕ := 10
def socks_to_buy : ℕ := 35

-- Define the function to calculate individual items
def individual_items (socks : ℕ) : ℕ :=
  2 * shoes + 2 * pants + tshirts + 2 * socks

-- Theorem to prove
theorem jonas_current_socks :
  ∃ current_socks : ℕ,
    individual_items (current_socks + socks_to_buy) = 2 * individual_items current_socks ∧
    current_socks = 15 := by
  sorry


end jonas_current_socks_l82_8204


namespace perfect_squares_between_100_and_500_l82_8285

theorem perfect_squares_between_100_and_500 : 
  (Finset.filter (fun n => 100 < n^2 ∧ n^2 < 500) (Finset.range 23)).card = 12 := by
  sorry

end perfect_squares_between_100_and_500_l82_8285


namespace rick_ironing_time_l82_8202

/-- Represents the rate at which Rick irons dress shirts per hour -/
def shirts_per_hour : ℕ := 4

/-- Represents the rate at which Rick irons dress pants per hour -/
def pants_per_hour : ℕ := 3

/-- Represents the number of hours Rick spent ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- Represents the total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

/-- Proves that Rick spent 3 hours ironing dress shirts given the conditions -/
theorem rick_ironing_time :
  ∃ (h : ℕ), h * shirts_per_hour + hours_ironing_pants * pants_per_hour = total_pieces ∧ h = 3 :=
by sorry

end rick_ironing_time_l82_8202


namespace smallest_deletion_for_order_l82_8293

theorem smallest_deletion_for_order (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, k = n - Int.ceil (Real.sqrt n) ∧
    (∀ perm : List ℕ, perm.length = n → perm.toFinset = Finset.range n →
      ∃ subseq : List ℕ, subseq.length = n - k ∧ 
        (subseq.Sorted (·<·) ∨ subseq.Sorted (·>·)) ∧
        subseq.toFinset ⊆ perm.toFinset) ∧
    (∀ k' : ℕ, k' < k →
      ∃ perm : List ℕ, perm.length = n ∧ perm.toFinset = Finset.range n ∧
        ∀ subseq : List ℕ, subseq.length > n - k' →
          subseq.toFinset ⊆ perm.toFinset →
            ¬(subseq.Sorted (·<·) ∨ subseq.Sorted (·>·))) :=
by
  sorry

end smallest_deletion_for_order_l82_8293


namespace x_equals_one_sufficient_not_necessary_for_x_squared_equals_one_l82_8231

theorem x_equals_one_sufficient_not_necessary_for_x_squared_equals_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) →
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  ¬(∀ x : ℝ, x^2 = 1 → x = 1) :=
by sorry

end x_equals_one_sufficient_not_necessary_for_x_squared_equals_one_l82_8231


namespace largest_possible_value_l82_8242

theorem largest_possible_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 1) :
  let M := 3 / (Real.sqrt 3 + 1)
  (x / (1 + y * z / x)) + (y / (1 + z * x / y)) + (z / (1 + x * y / z)) ≥ M ∧ 
  ∀ N > M, ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b + b * c + c * a = 1 ∧
    (a / (1 + b * c / a)) + (b / (1 + c * a / b)) + (c / (1 + a * b / c)) < N :=
by sorry

end largest_possible_value_l82_8242


namespace alice_winning_strategy_l82_8260

theorem alice_winning_strategy (x : ℕ) (h : x ≤ 2020) :
  ∃ k : ℤ, (2021 - x)^2 - x^2 = 2021 * k := by
  sorry

end alice_winning_strategy_l82_8260


namespace both_arithmetic_and_geometric_is_geometric_with_ratio_one_l82_8288

/-- A sequence that is both arithmetic and geometric -/
def BothArithmeticAndGeometric (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)

/-- Theorem: A sequence that is both arithmetic and geometric is a geometric sequence with common ratio 1 -/
theorem both_arithmetic_and_geometric_is_geometric_with_ratio_one 
  (a : ℕ → ℝ) (h : BothArithmeticAndGeometric a) : 
  ∃ r : ℝ, r = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r :=
by sorry

end both_arithmetic_and_geometric_is_geometric_with_ratio_one_l82_8288


namespace phi_value_l82_8212

theorem phi_value : ∃! (Φ : ℕ), Φ < 10 ∧ 504 / Φ = 40 + 3 * Φ :=
  sorry

end phi_value_l82_8212


namespace equation_solution_l82_8225

theorem equation_solution :
  ∃ x : ℚ, (2 / 3 + 1 / x = 7 / 9) ∧ (x = 9) :=
by
  sorry

end equation_solution_l82_8225


namespace cookie_distribution_l82_8219

theorem cookie_distribution (people : ℕ) (cookies_per_person : ℕ) 
  (h1 : people = 6) (h2 : cookies_per_person = 4) : 
  people * cookies_per_person = 24 := by
  sorry

end cookie_distribution_l82_8219


namespace min_sum_of_squares_l82_8246

theorem min_sum_of_squares (x y : ℝ) (h : (x + 4) * (y - 4) = 0) :
  ∃ (m : ℝ), m = 16 ∧ ∀ (a b : ℝ), (a + 4) * (b - 4) = 0 → a^2 + b^2 ≥ m :=
sorry

end min_sum_of_squares_l82_8246


namespace quadrilaterals_on_circle_l82_8258

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quad : ℕ := 4

/-- The number of points to choose from after fixing two points -/
def remaining_points : ℕ := num_points - 2

/-- The number of additional vertices needed after fixing two points -/
def additional_vertices : ℕ := vertices_per_quad - 2

theorem quadrilaterals_on_circle :
  choose num_points vertices_per_quad - choose remaining_points additional_vertices = 450 := by
  sorry

end quadrilaterals_on_circle_l82_8258


namespace switches_in_A_after_process_l82_8270

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 5

/-- The set of all switches -/
def switches : Finset Switch := sorry

/-- The process of advancing switches for 1000 steps -/
def advance_switches : Finset Switch → Finset Switch := sorry

/-- Counts switches in position A -/
def count_switches_in_A : Finset Switch → Nat := sorry

/-- Main theorem: After 1000 steps, 725 switches are in position A -/
theorem switches_in_A_after_process : 
  count_switches_in_A (advance_switches switches) = 725 := by sorry

end switches_in_A_after_process_l82_8270


namespace pats_password_length_l82_8230

/-- Represents the structure of Pat's computer password -/
structure PasswordStructure where
  lowercase_count : ℕ
  uppercase_and_numbers_count : ℕ
  symbol_count : ℕ

/-- Calculates the total number of characters in Pat's password -/
def total_characters (p : PasswordStructure) : ℕ :=
  p.lowercase_count + p.uppercase_and_numbers_count + p.symbol_count

/-- Theorem stating the total number of characters in Pat's password -/
theorem pats_password_length :
  ∃ (p : PasswordStructure),
    p.lowercase_count = 8 ∧
    p.uppercase_and_numbers_count = p.lowercase_count / 2 ∧
    p.symbol_count = 2 ∧
    total_characters p = 14 := by
  sorry

end pats_password_length_l82_8230


namespace gcf_lcm_360_270_l82_8273

theorem gcf_lcm_360_270 :
  (Nat.gcd 360 270 = 90) ∧ (Nat.lcm 360 270 = 1080) := by
  sorry

end gcf_lcm_360_270_l82_8273


namespace divisible_by_27_l82_8280

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) - 9*n^2 + 21*n - 14 = 27*k := by
  sorry

end divisible_by_27_l82_8280


namespace equivalence_of_statements_l82_8232

theorem equivalence_of_statements (P Q : Prop) :
  (¬P → Q) ↔ (¬Q → P) :=
by sorry

end equivalence_of_statements_l82_8232


namespace inequality_preservation_l82_8278

theorem inequality_preservation (m n : ℝ) (h : m > n) : m / 5 > n / 5 := by
  sorry

end inequality_preservation_l82_8278


namespace chad_savings_theorem_l82_8261

def calculate_savings (mowing_yards : ℝ) (birthday_holidays : ℝ) (video_games : ℝ) (odd_jobs : ℝ) : ℝ :=
  let total_earnings := mowing_yards + birthday_holidays + video_games + odd_jobs
  let tax_rate := 0.1
  let taxes := tax_rate * total_earnings
  let after_tax := total_earnings - taxes
  let mowing_savings := 0.5 * mowing_yards
  let birthday_savings := 0.3 * birthday_holidays
  let video_games_savings := 0.4 * video_games
  let odd_jobs_savings := 0.2 * odd_jobs
  mowing_savings + birthday_savings + video_games_savings + odd_jobs_savings

theorem chad_savings_theorem :
  calculate_savings 600 250 150 150 = 465 := by
  sorry

end chad_savings_theorem_l82_8261


namespace quadratic_equation_solution_l82_8251

theorem quadratic_equation_solution :
  ∀ x : ℝ, (x - 6) * (x + 2) = 0 ↔ x = 6 ∨ x = -2 := by sorry

end quadratic_equation_solution_l82_8251


namespace leap_day_2024_is_sunday_l82_8290

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Calculates the day of the week for a given number of days after a Sunday -/
def dayAfterSunday (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- The number of days between February 29, 2000, and February 29, 2024 -/
def daysBetween2000And2024 : Nat := 8766

theorem leap_day_2024_is_sunday :
  dayAfterSunday daysBetween2000And2024 = DayOfWeek.Sunday := by
  sorry

#check leap_day_2024_is_sunday

end leap_day_2024_is_sunday_l82_8290


namespace child_wage_is_eight_l82_8295

/-- Represents the daily wage structure and worker composition of a building contractor. -/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage of a child worker given the contractor's data. -/
def child_worker_wage (data : ContractorData) : ℕ :=
  ((data.average_wage * (data.male_workers + data.female_workers + data.child_workers)) -
   (data.male_wage * data.male_workers + data.female_wage * data.female_workers)) / data.child_workers

/-- Theorem stating that given the specific conditions, the child worker's daily wage is 8 rupees. -/
theorem child_wage_is_eight (data : ContractorData)
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.child_workers = 5)
  (h4 : data.male_wage = 25)
  (h5 : data.female_wage = 20)
  (h6 : data.average_wage = 21) :
  child_worker_wage data = 8 := by
  sorry


end child_wage_is_eight_l82_8295


namespace sum_always_positive_l82_8218

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f (x + 4)) ∧
  (∀ x ≥ 2, Monotone (fun y ↦ f y))

/-- Theorem statement -/
theorem sum_always_positive
  (f : ℝ → ℝ)
  (hf : special_function f)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ > 4)
  (h_prod : (x₁ - 2) * (x₂ - 2) < 0) :
  f x₁ + f x₂ > 0 :=
by sorry

end sum_always_positive_l82_8218


namespace fraction_difference_simplest_form_l82_8249

theorem fraction_difference_simplest_form :
  let a := 5
  let b := 19
  let c := 2
  let d := 23
  let numerator := a * d - c * b
  let denominator := b * d
  (numerator : ℚ) / denominator = 77 / 437 ∧
  ∀ (x y : ℤ), x ≠ 0 → (77 : ℚ) / 437 = (x : ℚ) / y → (x = 77 ∧ y = 437 ∨ x = -77 ∧ y = -437) :=
by sorry

end fraction_difference_simplest_form_l82_8249


namespace intersection_of_A_and_B_l82_8247

-- Define sets A and B
def A : Set ℝ := {x | 2 * x + 1 < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l82_8247


namespace remainder_problem_l82_8279

theorem remainder_problem (n : ℤ) (k : ℤ) (h : n = 25 * k - 2) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end remainder_problem_l82_8279


namespace problem_statement_l82_8282

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y - 6 = 0) 
  (h2 : z^2 + 9 = x*y) : 
  x^2 + (1/3)*y^2 = 12 := by
sorry

end problem_statement_l82_8282


namespace judy_spending_l82_8229

def carrot_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 5
def ice_cream_price : ℕ := 7

def carrot_quantity : ℕ := 5
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 2
def flour_quantity : ℕ := 2

def coupon_discount : ℕ := 10
def coupon_threshold : ℕ := 40

def shopping_total : ℕ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  pineapple_price * (pineapple_quantity / 2) +
  flour_price * flour_quantity +
  ice_cream_price

theorem judy_spending :
  shopping_total = 38 :=
by sorry

end judy_spending_l82_8229


namespace tournament_size_l82_8296

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players not in the weakest 15
  total_players : ℕ := n + 15
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  weak_player_games : ℕ := 15 * 14 / 2
  strong_player_games : ℕ := n * (n - 1) / 2
  cross_games : ℕ := 15 * n

/-- The theorem stating that the tournament must have 36 players -/
theorem tournament_size (t : Tournament) : t.total_players = 36 := by
  sorry

end tournament_size_l82_8296


namespace at_least_one_leq_neg_two_l82_8208

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
  sorry

end at_least_one_leq_neg_two_l82_8208


namespace train_crossing_time_l82_8238

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (h1 : train_length = 130) (h2 : train_speed_kmh = 144) : 
  train_length / (train_speed_kmh * 1000 / 3600) = 3.25 := by
  sorry

#check train_crossing_time

end train_crossing_time_l82_8238


namespace largest_y_coordinate_l82_8264

theorem largest_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end largest_y_coordinate_l82_8264


namespace bakers_cakes_l82_8272

/-- The number of cakes Baker made is equal to the sum of cakes sold and cakes left. -/
theorem bakers_cakes (total sold left : ℕ) (h1 : sold = 145) (h2 : left = 72) (h3 : total = sold + left) :
  total = 217 := by
  sorry

end bakers_cakes_l82_8272


namespace probability_x_plus_y_less_than_5_l82_8287

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A point inside the square --/
structure PointInSquare (s : Square) where
  point : ℝ × ℝ
  inside : point.1 ≥ s.bottomLeft.1 ∧ point.1 ≤ s.topRight.1 ∧
           point.2 ≥ s.bottomLeft.2 ∧ point.2 ≤ s.topRight.2

/-- The probability of an event for a uniformly distributed point in the square --/
def probability (s : Square) (event : PointInSquare s → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_5 :
  let s : Square := ⟨(0, 0), (4, 4)⟩
  probability s (fun p => p.point.1 + p.point.2 < 5) = 29 / 32 := by
  sorry

end probability_x_plus_y_less_than_5_l82_8287


namespace wallace_existing_bags_l82_8267

/- Define the problem parameters -/
def batch_size : ℕ := 10
def order_size : ℕ := 60
def days_to_fulfill : ℕ := 4

/- Define the function to calculate the number of bags Wallace can make in given days -/
def bags_made_in_days (days : ℕ) : ℕ := days * batch_size

/- Theorem: Wallace has already made 20 bags of jerky -/
theorem wallace_existing_bags : 
  order_size - bags_made_in_days days_to_fulfill = 20 := by
  sorry

#eval order_size - bags_made_in_days days_to_fulfill

end wallace_existing_bags_l82_8267


namespace smallest_n_for_probability_threshold_l82_8298

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def P (n : ℕ) : ℚ := 1 / (n * (n^2 + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 3000

theorem smallest_n_for_probability_threshold :
  ∀ n : ℕ, n > 0 → n < 15 → P n ≥ 1 / num_boxes ∧
  P 15 < 1 / num_boxes :=
sorry

end smallest_n_for_probability_threshold_l82_8298


namespace dogs_food_consumption_l82_8268

/-- The amount of dog food eaten by one dog per day -/
def dog_food_per_day : ℝ := 0.12

/-- The number of dogs -/
def num_dogs : ℕ := 2

/-- The total amount of dog food eaten by all dogs per day -/
def total_dog_food : ℝ := dog_food_per_day * num_dogs

theorem dogs_food_consumption :
  total_dog_food = 0.24 := by sorry

end dogs_food_consumption_l82_8268


namespace fn_equals_de_l82_8205

-- Define the circle
variable (O : Point) (A B : Point)
variable (circle : Circle O)

-- Define other points
variable (C D E F M N : Point)

-- Define the conditions
variable (h1 : C ∈ circle)
variable (h2 : Diameter circle A B)
variable (h3 : Perpendicular CD AB D)
variable (h4 : E ∈ Segment B D)
variable (h5 : AE = AC)
variable (h6 : Square D E F M)
variable (h7 : N ∈ circle ∩ Line A M)

-- State the theorem
theorem fn_equals_de : FN = DE := by
  sorry

end fn_equals_de_l82_8205


namespace sum_of_seventh_eighth_ninth_l82_8274

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_first_three : a 1 + a 2 + a 3 = 30
  sum_next_three : a 4 + a 5 + a 6 = 120

/-- The sum of the 7th, 8th, and 9th terms equals 480 -/
theorem sum_of_seventh_eighth_ninth (seq : GeometricSequence) : 
  seq.a 7 + seq.a 8 + seq.a 9 = 480 := by
  sorry

end sum_of_seventh_eighth_ninth_l82_8274


namespace multiplication_sum_equality_l82_8253

theorem multiplication_sum_equality : 45 * 58 + 45 * 42 = 4500 := by
  sorry

end multiplication_sum_equality_l82_8253
